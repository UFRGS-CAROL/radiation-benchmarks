#!/bin/bash

set -e

# this script will download and check darknet V1 and V2
# weights consistency

DIR=.
YOLOV1=yolo_v1.weights
YOLOV2=yolo_v2.weights

V1MD5SUM=25a56f52d3d536df443989a465c35370
V2MD5SUM=70d89ba2e180739a1c700a9ff238e354


if [ ! -f $DIR/$YOLOV1 ] || [ ! "$(echo "$V1MD5SUM $DIR/$YOLOV1" | md5sum -c)" == "$DIR/$YOLOV1: OK" ];
then
    wget https://www.dropbox.com/s/389dxz0l2c7l3i0/yolo.weights -O $DIR/$YOLOV1
fi


if [ ! -f $DIR/$YOLOV2 ] || [ ! "$(echo "$V2MD5SUM $DIR/$YOLOV2" | md5sum -c)" == "$DIR/$YOLOV2: OK" ];
then
    wget https://pjreddie.com/media/files/yolo.weights -O $DIR/$YOLOV2
fi

#check m5sum
if [ ! "$(echo "$V1MD5SUM $DIR/$YOLOV1" | md5sum -c)" == "$DIR/$YOLOV1: OK" ];
then
    echo "ERROR: MD5SUM is not the same for YoloV1 weights, exiting..."
    exit -1
fi

if [ ! "$(echo "$V2MD5SUM $DIR/$YOLOV2" | md5sum -c)" == "$DIR/$YOLOV2: OK" ];
then
    echo "ERROR: MD5SUM is not the same for YoloV2 weights, exiting..."
    exit -1
fi

echo "If you are reading this all darknet weight files were downloaded and OK"

exit 0
