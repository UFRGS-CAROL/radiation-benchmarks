#!/bin/bash

set -e

# this script will download and check darknet V1 and V2
# weights consistency

DIR=.
YOLOV1=yolo_v1.weights
YOLOV2=yolo_v2.weights

TINY_YOLOV1=tiny-yolo_v1.weights
TINY_YOLOV2=tiny-yolo-voc_v2.weights


V1MD5SUM=25a56f52d3d536df443989a465c35370
V2MD5SUM=70d89ba2e180739a1c700a9ff238e354

TINYV1MD5SUM=04904b18f0e0835d8e1d6e75d3fa93fe
TINYV2MD5SUM=fca33deaff44dec1750a34df42d2807e


#V1-------------------------
if [ ! -f $DIR/$YOLOV1 ] || [ ! "$(echo "$V1MD5SUM $DIR/$YOLOV1" | md5sum -c)" == "$DIR/$YOLOV1: OK" ];
then
    wget https://www.dropbox.com/s/389dxz0l2c7l3i0/yolo.weights -O $DIR/$YOLOV1 --no-check-certificate
fi

#check m5sum
if [ ! "$(echo "$V1MD5SUM $DIR/$YOLOV1" | md5sum -c)" == "$DIR/$YOLOV1: OK" ];
then
    echo "ERROR: MD5SUM is not the same for YoloV1 weights, exiting..."
    exit -1
fi

#~ #TINY V1
#~ if [ ! -f $DIR/$TINY_YOLOV1 ] || [ ! "$(echo "$V1MD5SUM $DIR/$TINY_YOLOV1" | md5sum -c)" == "$DIR/$TINY_YOLOV1: OK" ];
#~ then
    #~ wget https://www.dropbox.com/s/acakz3i4ee0tp70/tiny-yolo_v1.weights -O $DIR/$TINY_YOLOV1 --no-check-certificate
#~ fi

#~ #check m5sum
#~ if [ ! "$(echo "$TINYV1MD5SUM $DIR/$TINY_YOLOV1" | md5sum -c)" == "$DIR/$TINY_YOLOV1: OK" ];
#~ then
    #~ echo "ERROR: MD5SUM is not the same for Tiny YoloV1 weights, exiting..."
    #~ exit -1
#~ fi

########################################################################
#V2-------------------------
if [ ! -f $DIR/$YOLOV2 ] || [ ! "$(echo "$V2MD5SUM $DIR/$YOLOV2" | md5sum -c)" == "$DIR/$YOLOV2: OK" ];
then
    wget https://pjreddie.com/media/files/yolo.weights -O $DIR/$YOLOV2 --no-check-certificate
fi


#check m5sum
if [ ! "$(echo "$V2MD5SUM $DIR/$YOLOV2" | md5sum -c)" == "$DIR/$YOLOV2: OK" ];
then
    echo "ERROR: MD5SUM is not the same for YoloV2 weights, exiting..."
    exit -1
fi

#TINY V2
if [ ! -f $DIR/$TINY_YOLOV2 ] || [ ! "$(echo "$V1MD5SUM $DIR/$TINY_YOLOV2" | md5sum -c)" == "$DIR/$TINY_YOLOV2: OK" ];
then
    wget https://www.dropbox.com/s/8ncf5lroyluudvs/tiny-yolo-voc_v2.weights -O $DIR/$TINY_YOLOV2 --no-check-certificate
fi

#check m5sum
if [ ! "$(echo "$TINYV2MD5SUM $DIR/$TINY_YOLOV2" | md5sum -c)" == "$DIR/$TINY_YOLOV2: OK" ];
then
    echo "ERROR: MD5SUM is not the same for Tiny YoloV2 weights, exiting..."
    exit -1
fi

echo "If you are reading this all darknet weight files were downloaded and OK"

exit 0
