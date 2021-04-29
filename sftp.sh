#! /bin/bash

rm -rf /Users/limengfan/PycharmProjects/210414_CfgYoloV3Sftp

mkdir /Users/limengfan/PycharmProjects/210414_CfgYoloV3Sftp
cp /Users/limengfan/PycharmProjects/210414_CfgYoloV3/run.sh /Users/limengfan/PycharmProjects/210414_CfgYoloV3Sftp
cp /Users/limengfan/PycharmProjects/210414_CfgYoloV3/run_pennfudan.sh /Users/limengfan/PycharmProjects/210414_CfgYoloV3Sftp
cp /Users/limengfan/PycharmProjects/210414_CfgYoloV3/kill.sh /Users/limengfan/PycharmProjects/210414_CfgYoloV3Sftp
cp /Users/limengfan/PycharmProjects/210414_CfgYoloV3/map.sh /Users/limengfan/PycharmProjects/210414_CfgYoloV3Sftp
cp /Users/limengfan/PycharmProjects/210414_CfgYoloV3/map_pennfudan.sh /Users/limengfan/PycharmProjects/210414_CfgYoloV3Sftp

cp /Users/limengfan/PycharmProjects/210414_CfgYoloV3/view_process.sh /Users/limengfan/PycharmProjects/210414_CfgYoloV3Sftp
cp /Users/limengfan/PycharmProjects/210414_CfgYoloV3/view_run.sh /Users/limengfan/PycharmProjects/210414_CfgYoloV3Sftp
cp /Users/limengfan/PycharmProjects/210414_CfgYoloV3/view_gpu.sh /Users/limengfan/PycharmProjects/210414_CfgYoloV3Sftp

cp /Users/limengfan/PycharmProjects/210414_CfgYoloV3/train_utils.py /Users/limengfan/PycharmProjects/210414_CfgYoloV3Sftp
cp /Users/limengfan/PycharmProjects/210414_CfgYoloV3/train.py /Users/limengfan/PycharmProjects/210414_CfgYoloV3Sftp
cp /Users/limengfan/PycharmProjects/210414_CfgYoloV3/train_pennfudan.py /Users/limengfan/PycharmProjects/210414_CfgYoloV3Sftp

cp /Users/limengfan/PycharmProjects/210414_CfgYoloV3/predict_pennfudan_with_eval.py /Users/limengfan/PycharmProjects/210414_CfgYoloV3Sftp
cp /Users/limengfan/PycharmProjects/210414_CfgYoloV3/predict_pennfudan_with_loss.py /Users/limengfan/PycharmProjects/210414_CfgYoloV3Sftp
cp /Users/limengfan/PycharmProjects/210414_CfgYoloV3/predict_voc_with_loss.py /Users/limengfan/PycharmProjects/210414_CfgYoloV3Sftp
cp /Users/limengfan/PycharmProjects/210414_CfgYoloV3/predict_voc_with_eval.py /Users/limengfan/PycharmProjects/210414_CfgYoloV3Sftp

cp /Users/limengfan/PycharmProjects/210414_CfgYoloV3/compute_map.py /Users/limengfan/PycharmProjects/210414_CfgYoloV3Sftp

mkdir /Users/limengfan/PycharmProjects/210414_CfgYoloV3Sftp/conf
cp /Users/limengfan/PycharmProjects/210414_CfgYoloV3/conf/coco.names /Users/limengfan/PycharmProjects/210414_CfgYoloV3Sftp/conf
cp /Users/limengfan/PycharmProjects/210414_CfgYoloV3/conf/config.py /Users/limengfan/PycharmProjects/210414_CfgYoloV3Sftp/conf
cp /Users/limengfan/PycharmProjects/210414_CfgYoloV3/conf/voc.names /Users/limengfan/PycharmProjects/210414_CfgYoloV3Sftp/conf

mkdir /Users/limengfan/PycharmProjects/210414_CfgYoloV3Sftp/dataset
cp /Users/limengfan/PycharmProjects/210414_CfgYoloV3/dataset/transform.py /Users/limengfan/PycharmProjects/210414_CfgYoloV3Sftp/dataset
cp /Users/limengfan/PycharmProjects/210414_CfgYoloV3/dataset/pennfudan_dataset.py /Users/limengfan/PycharmProjects/210414_CfgYoloV3Sftp/dataset
cp /Users/limengfan/PycharmProjects/210414_CfgYoloV3/dataset/voc_dataset.py /Users/limengfan/PycharmProjects/210414_CfgYoloV3Sftp/dataset

mkdir /Users/limengfan/PycharmProjects/210414_CfgYoloV3Sftp/model
cp /Users/limengfan/PycharmProjects/210414_CfgYoloV3/model/base_model.py /Users/limengfan/PycharmProjects/210414_CfgYoloV3Sftp/model
cp /Users/limengfan/PycharmProjects/210414_CfgYoloV3/model/darknet.py /Users/limengfan/PycharmProjects/210414_CfgYoloV3Sftp/model
cp /Users/limengfan/PycharmProjects/210414_CfgYoloV3/model/yolov3.py /Users/limengfan/PycharmProjects/210414_CfgYoloV3Sftp/model
cp /Users/limengfan/PycharmProjects/210414_CfgYoloV3/model/yolov3decode.py /Users/limengfan/PycharmProjects/210414_CfgYoloV3Sftp/model
cp /Users/limengfan/PycharmProjects/210414_CfgYoloV3/model/yolov3loss.py /Users/limengfan/PycharmProjects/210414_CfgYoloV3Sftp/model
cp /Users/limengfan/PycharmProjects/210414_CfgYoloV3/model/yolov3net.py /Users/limengfan/PycharmProjects/210414_CfgYoloV3Sftp/model

#mkdir /Users/limengfan/PycharmProjects/210414_CfgYoloV3Sftp/outer_weights
#cp /Users/limengfan/PycharmProjects/210414_CfgYoloV3/outer_weights/demo_darknet53_weights.pth /Users/limengfan/PycharmProjects/210414_CfgYoloV3Sftp/outer_weights

mkdir /Users/limengfan/PycharmProjects/210414_CfgYoloV3Sftp/logs

mkdir /Users/limengfan/PycharmProjects/210414_CfgYoloV3Sftp/outer_map_input
mkdir /Users/limengfan/PycharmProjects/210414_CfgYoloV3Sftp/outer_map_input/detection_result
mkdir /Users/limengfan/PycharmProjects/210414_CfgYoloV3Sftp/outer_map_input/ground_truth

sftp lab2 << EOF
put -r /Users/limengfan/PycharmProjects/210414_CfgYoloV3Sftp /home/lenovo/data/lmf
EOF
