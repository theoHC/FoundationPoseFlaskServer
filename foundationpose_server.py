# app.py
import io
import json
import time
import numpy as np
import cv2
from flask import Flask, request, jsonify
import os
import traceback
import time

from estimater import *
from datareader import *
import argparse

app = Flask(__name__)

def read_image_from_upload(file_storage, flags=cv2.IMREAD_UNCHANGED):
    data = file_storage.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), flags)
    if img is None:
        raise ValueError("Failed to decode uploaded image")
    return img

def parse_intrinsics(k_json):
    """
    Accept either:
      - {"fx":..., "fy":..., "cx":..., "cy":...}
      - {"K":[[...],[...],[...]]}
    """
    if "K" in k_json:
        K = np.array(k_json["K"], dtype=np.float64)
        assert K.shape == (3,3)
        return K
    fx, fy, cx, cy = (k_json["fx"], k_json["fy"], k_json["cx"], k_json["cy"])
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float64)
    return K

class FoundationPoseService:
    def __init__(self):
        # TODO: load FoundationPose here once (weights, meshes registry, etc.)
        # self.estimator = ...
        # self.objects = {"mustard": "/path/to/mustard.obj", ...}

        set_seed(0)

        code_dir = os.path.dirname(os.path.realpath(__file__))
        mesh_path = os.path.join(
            code_dir,
            "meshes",
            "tello_detail.obj"
        )
        mesh = trimesh.load(mesh_path)

        self.to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        self.bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()
        self.est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=self.scorer, refiner=self.refiner, debug_dir="debug", debug=0, glctx=self.glctx)
        logging.info("estimator initialization done")

        self.pose = None

        self.last_call_time = None
        
        pass

    def estimate_pose(self, color, depth, K, mask=None):
        """
        color: HxWx3 uint8 (OpenCV RGB)
        depth: HxW (uint16 mm or float32 meters)
        K: 3x3 float32
        bbox: [x,y,w,h] optional
        mask: HxW bool optional
        depth_scale: if depth is uint16 in mm -> meters scale=0.001
        """

        now = time.time()
        
        reregister = False

        if self.last_call_time is None:
            reregister = True
        elif now - self.last_call_time > 2.0:
            print("TIMEOUT-BASED REREGISTER")
            reregister = True

        if (self.pose is None or reregister) and mask is not None:
            self.pose = self.est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=2)
        else:
            if self.est.pose_last is None:
                return None
            self.pose = self.est.track_one(rgb=color, depth=depth, K=K, iteration=2)

        self.last_call_time = time.time()

        return self.pose
    
    def score_current_pose(self, rgb, depth, K):
        """
        Runs the FoundationPose score network on the current pose (tracking case).
        Returns a python float.
        """
        import torch  # estimater.py uses torch; safe to import here

        # scorer expects depth processed similarly to estimater.py
        depth_t = torch.as_tensor(depth, device="cuda", dtype=torch.float)
        depth_t = erode_depth(depth_t, radius=2, device="cuda")
        depth_t = bilateral_filter_depth(depth_t, radius=2, device="cuda")

        scores, _vis = self.est.scorer.predict(
            mesh=self.est.mesh,
            rgb=rgb,
            depth=depth_t,
            K=K,
            ob_in_cams=self.est.pose_last.reshape(1, 4, 4).data.cpu().numpy(),
            normal_map=None,
            mesh_tensors=self.est.mesh_tensors,
            glctx=self.est.glctx,
            mesh_diameter=self.est.diameter,
            get_vis=False,
        )

        # scores is typically a torch tensor or numpy array of shape (N,)
        try:
            return float(scores[0].item())
        except Exception:
            return float(scores[0])

service = FoundationPoseService()
print("Initialization Complete - FoundationPose Online!")

@app.post("/pose")
def pose():
    t0 = time.time()

    # multipart/form-data: rgb, depth, optional mask
    if "rgb" not in request.files or "depth" not in request.files:
        print("RGB + D not found.")
        return jsonify({"error": "Expected files: rgb, depth"}), 400
    
    rgb_shape_s = request.form["rgb_shape"]      # string like '[480, 640, 3]'
    depth_shape_s = request.form["depth_shape"]

    rgb_shape = tuple(json.loads(rgb_shape_s))
    depth_shape = tuple(json.loads(depth_shape_s))

    if not rgb_shape or not depth_shape:
        print("Shapes not found.")
        return jsonify({"error": "Expected form fields: rgb_shape, depth_shape"}), 400

    rgb = np.frombuffer(request.files["rgb"].read(), dtype=np.uint8).reshape(rgb_shape)
    depth = np.frombuffer(request.files["depth"].read(), dtype=np.float32).reshape(depth_shape)

    # JSON fields (either form fields or a single json part)
    # object_id = request.form.get("object_id", default=None, type=str)
    # if not object_id:
    #     return jsonify({"error": "Missing form field: object_id"}), 400

    k_raw = request.form.get("K") or request.form.get("intrinsics")
    if not k_raw:
        print("bad intrinsics")
        return jsonify({"error": "Missing intrinsics (form field K or intrinsics)"}), 400
    K = parse_intrinsics(json.loads(k_raw))

    # bbox_raw = request.form.get("bbox")
    # bbox = json.loads(bbox_raw) if bbox_raw else None  # [x,y,w,h]

    mask = None
    if "masknp" in request.files:
        mask_shape_s = request.form["mask_shape"]
        mask_shape = tuple(json.loads(mask_shape_s))
        mask = np.frombuffer(request.files["masknp"].read(), dtype=np.uint8).reshape(mask_shape)

    if "mask" in request.files:
        mask_img = read_image_from_upload(request.files["mask"], flags=cv2.IMREAD_UNCHANGED)
        # accept 0/255 or 0/1
        if mask_img.ndim == 3:
            mask_img = mask_img[..., 0]
        mask = mask_img > 0

    try:
        pose = service.estimate_pose(rgb, depth, K, mask=mask)
    except Exception as e:
        print(f"estimation error {e}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

    if pose is not None:
        return jsonify({"pose": pose.tolist()})
    else:
        print("Error with pose estimation - no pose returned")
        return jsonify({"error": "Error with pose estimation - no pose returned"}), 500

@app.post("/reset")
def reset():
    service.pose = None
    return jsonify({"status": "ok"})

@app.route("/hello")
def hello_world():
    return "<p>Hello, World!</p>"

# if __name__ == "__main__":
#     # For local testing only. For GPU inference use gunicorn in practice.
#     app.run(host="0.0.0.0", port=8000, debug=False)
