sudo docker run --rm -it \
    -e DISPLAY=":0" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
    -v /$(pwd):/home/developer/lrauv_ws/src/lrauv_application \
    -w /home/developer/lrauv_ws/src/lrauv_application \
    -e MESA_GL_VERSION_OVERRIDE=3.3 \
    --gpus all \
    ros2_lrauv:latest \
    bash