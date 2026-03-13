# FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects
[[Paper]](https://arxiv.org/abs/2312.08344) [[Website]](https://nvlabs.github.io/FoundationPose/)

This is a fork of [Nvidia's FoundationPose](https://github.com/NVlabs/FoundationPose), incorporating streamlined setup and a http server for remote inference. This readme has been modified from the original.

# Data prepare


1) Download all network weights from [here](https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i?usp=sharing) and put them under the folder `weights/`. For the refiner, you will need `2023-10-28-18-33-37`. For scorer, you will need `2024-01-11-20-02-45`.

1) [optional] [Download demo data](https://drive.google.com/drive/folders/1pRyFmxYXmAnpku7nGRioZaKrVJtIsroP?usp=sharing) and extract them under the folder `demo_data/`

# Docker setup
build a docker container from dockerfile.flask file in docker.


If it's the first time you launch the container, you need to build extensions. Run this command *inside* the Docker container.
```
bash build_all.sh
```

Later you can execute into the container without re-build.
```
docker exec -it foundationpose bash
```

Outisde the docker container, create a python virtual environment exposed to your system packages and install Flask

```
python3 venv .venv --system-site-packages
source .venv/bin/activate
pip install flask
```


# Run model-based demo
The paths have been set in argparse by default. If you need to change the scene, you can pass the args accordingly. By running on the demo data, you should be able to see the robot manipulating the mustard bottle. Pose estimation is conducted on the first frame, then it automatically switches to tracking mode for the rest of the video. The resulting visualizations will be saved to the `debug_dir` specified in the argparse. (Note the first time running could be slower due to online compilation)
```
python run_demo.py
```


<img src="assets/demo.jpg" width="50%">


Feel free to try on other objects (**no need to retrain**) such as driller, by changing the paths in argparse.

<img src="assets/demo_driller.jpg" width="50%">

# Run Flask Server
Source the Flask virtual environment inside the container, then run 
```
bash runfpflask.bash
```
This will make FoundationPose available at port 4242. This script will do for testing; Flask warns you loudly and prominently that you shouldn't use this script in production environments - I forward this warning to you in the readme to do with as you will.

Before you run the server, make sure to provide your 3d model, and modify foundationpose_server.py to point it at the location of your .obj

The server provides 3 interactions: /pose, /reset, and /hello.

/hello returns a simple "hello world" text, and is useful for testing your connection

/pose expects properly formatted rgb, depth, and camera intrinsics information. Images should be included as numpy arrays in the request.files, under the keys 'rgb' and 'depth'. The shape (jsonified numpy shape tuple) of these images should be included in the .data under the key 'rgb_shape' and 'depth_shape' keys. Intrinsics are accepted under the key 'K' or 'intrinsics' and should be either a 3x3 numpy array or fx, fy, cx, cy map. If this is the first request, a mask containing only the region of your image with the object of interest in it, transmitted similarly to color and depth, is also required.

/pose returns a json containing the 3x3 transform matrix and the score assigned by the foundationpose scorer. By default, score is only populated on the first request, and a default value of -1.0 is returned otherwise. Including a 'rescore' key in your request.data with the associated value of 'true' will cause the server to run the scorer on the currently tracked pose. Note that this increases inference time.

# Contact
For questions, please contact [Theo Coulson](https://www.theocoulson.com).
