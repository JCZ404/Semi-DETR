#!/usr/bin/env bash
TYPE=$1
GPUS=$2


rangeStart=29590
rangeEnd=29690

PORT=0
# 判断当前端口是否被占用，没被占用返回0，反之1
function Listening {
   TCPListeningnum=`netstat -an | grep ":$1 " | awk '$1 == "tcp" && $NF == "LISTEN" {print $0}' | wc -l`
   UDPListeningnum=`netstat -an | grep ":$1 " | awk '$1 == "udp" && $NF == "0.0.0.0:*" {print $0}' | wc -l`
   (( Listeningnum = TCPListeningnum + UDPListeningnum ))
   if [ $Listeningnum == 0 ]; then
       echo "0"
   else
       echo "1"
   fi
}

# 指定区间随机数
function random_range {
   shuf -i $1-$2 -n1
}

# 得到随机端口
function get_random_port {
   templ=0
   while [ $PORT == 0 ]; do
       temp1=`random_range $1 $2`
       if [ `Listening $temp1` == 0 ] ; then
              PORT=$temp1
       fi
   done
   echo "Using Port=$PORT"
}

# main
get_random_port ${rangeStart} ${rangeEnd};


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
if [[ ${TYPE} == 'dino_detr' ]]; then
    WORK_DIR='work_dirs/dino_detr_r50_8x2_12e_coco'
    mkdir -p $WORK_DIR
    python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port $PORT \
    $(dirname "$0")/train_detr_od.py configs/dino_detr/dino_detr_r50_8x2_12e_coco.py \
    --launcher pytorch \
    --work-dir $WORK_DIR

fi