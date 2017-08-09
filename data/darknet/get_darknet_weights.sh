#!/bin/bash

set -e

# this script will download and check darknet V1 and V2
# weights consistency

DIR=.
YOLOV1=yolo_v1.weights
YOLOV2=yolo_v2.weights


if [ ! -f $DIR/$YOLOV1 ];
then
    wget http://pjreddie.com/media/files/yolov1.weights -O $DIR/$YOLOV1
fi


if [ ! -f $DIR/$YOLOV2 ];
then
    wget https://pjreddie.com/media/files/yolo.weights -O $DIR/$YOLOV2
fi

#check m5sum
if [ ! "$(echo "f90aa062b70d10e46649e1f1bd69edce $DIR/$YOLOV1" | md5sum -c)" == "$DIR/$YOLOV1: OK" ];
then
    echo "ERROR: MD5SUM is not the same for YoloV1 weights, exiting..."
    exit -1
fi

if [ ! "$(echo "70d89ba2e180739a1c700a9ff238e354 $DIR/$YOLOV2" | md5sum -c)" == "$DIR/$YOLOV2: OK" ];
then
    echo "ERROR: MD5SUM is not the same for YoloV2 weights, exiting..."
    exit -1
fi

echo "If you are reading this all darknet weight files were downloaded and OK"

exit 0
