#!/usr/bin/env bash
GPUS=$1

rangeStart=29620
rangeEnd=29630

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

WORK_DIR="work_dirs/detr_ssod_dino_detr_r50_coco_full_240k/"
mkdir -p $WORK_DIR
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port $PORT \
    $(dirname "$0")/train_detr_ssod.py configs/detr_ssod/detr_ssod_dino_detr_r50_coco_full_240k.py \
    --work-dir $WORK_DIR \
    --launcher pytorch
  

