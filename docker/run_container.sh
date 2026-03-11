docker rm -f /foundationpose_flask
DIR=$(pwd)/../
docker run --gpus all --env NVIDIA_DISABLE_REQUIRE=1 -it -p 4242:4242 --name foundationpose_flask --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $DIR:$DIR -v /home:/home -v /mnt:/mnt -v /tmp:/tmp  --ipc=host  -e GIT_INDEX_FILE foundationpose_flask bash -c "unset DISPLAY && cd $DIR && bash"
