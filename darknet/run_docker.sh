nvidia-docker run -it --rm \
-v ./cfg/bdd:/opt/darknet/cfg/bdd \
-v ./cfg/aiedge/:/opt/darknet/cfg/aiedge \
darknet:latest
