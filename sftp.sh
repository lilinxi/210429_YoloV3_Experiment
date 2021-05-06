#! /bin/bash

rm -rf /Users/limengfan/PycharmProjects/210429_YoloV3_ExperimentSftp

mkdir /Users/limengfan/PycharmProjects/210429_YoloV3_ExperimentSftp
cp /Users/limengfan/PycharmProjects/210429_YoloV3_Experiment/run.sh /Users/limengfan/PycharmProjects/210429_YoloV3_ExperimentSftp
cp /Users/limengfan/PycharmProjects/210429_YoloV3_Experiment/kill.sh /Users/limengfan/PycharmProjects/210429_YoloV3_ExperimentSftp
cp /Users/limengfan/PycharmProjects/210429_YoloV3_Experiment/map.sh /Users/limengfan/PycharmProjects/210429_YoloV3_ExperimentSftp

cp /Users/limengfan/PycharmProjects/210429_YoloV3_Experiment/view_process.sh /Users/limengfan/PycharmProjects/210429_YoloV3_ExperimentSftp
cp /Users/limengfan/PycharmProjects/210429_YoloV3_Experiment/view_run.sh /Users/limengfan/PycharmProjects/210429_YoloV3_ExperimentSftp
cp /Users/limengfan/PycharmProjects/210429_YoloV3_Experiment/view_gpu.sh /Users/limengfan/PycharmProjects/210429_YoloV3_ExperimentSftp

cp /Users/limengfan/PycharmProjects/210429_YoloV3_Experiment/train_utils.py /Users/limengfan/PycharmProjects/210429_YoloV3_ExperimentSftp
cp /Users/limengfan/PycharmProjects/210429_YoloV3_Experiment/train.py /Users/limengfan/PycharmProjects/210429_YoloV3_ExperimentSftp

cp /Users/limengfan/PycharmProjects/210429_YoloV3_Experiment/predict_pennfudan_with_eval.py /Users/limengfan/PycharmProjects/210429_YoloV3_ExperimentSftp
cp /Users/limengfan/PycharmProjects/210429_YoloV3_Experiment/predict_pennfudan_with_loss.py /Users/limengfan/PycharmProjects/210429_YoloV3_ExperimentSftp
cp /Users/limengfan/PycharmProjects/210429_YoloV3_Experiment/predict_voc_with_loss.py /Users/limengfan/PycharmProjects/210429_YoloV3_ExperimentSftp
cp /Users/limengfan/PycharmProjects/210429_YoloV3_Experiment/predict_voc_with_eval.py /Users/limengfan/PycharmProjects/210429_YoloV3_ExperimentSftp

cp /Users/limengfan/PycharmProjects/210429_YoloV3_Experiment/compute_map.py /Users/limengfan/PycharmProjects/210429_YoloV3_ExperimentSftp

mkdir /Users/limengfan/PycharmProjects/210429_YoloV3_ExperimentSftp/conf
cp /Users/limengfan/PycharmProjects/210429_YoloV3_Experiment/conf/coco.names /Users/limengfan/PycharmProjects/210429_YoloV3_ExperimentSftp/conf
cp /Users/limengfan/PycharmProjects/210429_YoloV3_Experiment/conf/config.py /Users/limengfan/PycharmProjects/210429_YoloV3_ExperimentSftp/conf
cp /Users/limengfan/PycharmProjects/210429_YoloV3_Experiment/conf/voc.names /Users/limengfan/PycharmProjects/210429_YoloV3_ExperimentSftp/conf

mkdir /Users/limengfan/PycharmProjects/210429_YoloV3_ExperimentSftp/dataset
cp /Users/limengfan/PycharmProjects/210429_YoloV3_Experiment/dataset/transform.py /Users/limengfan/PycharmProjects/210429_YoloV3_ExperimentSftp/dataset
cp /Users/limengfan/PycharmProjects/210429_YoloV3_Experiment/dataset/pennfudan_dataset.py /Users/limengfan/PycharmProjects/210429_YoloV3_ExperimentSftp/dataset
cp /Users/limengfan/PycharmProjects/210429_YoloV3_Experiment/dataset/voc_dataset.py /Users/limengfan/PycharmProjects/210429_YoloV3_ExperimentSftp/dataset

mkdir /Users/limengfan/PycharmProjects/210429_YoloV3_ExperimentSftp/model
cp /Users/limengfan/PycharmProjects/210429_YoloV3_Experiment/model/base_model.py /Users/limengfan/PycharmProjects/210429_YoloV3_ExperimentSftp/model
cp /Users/limengfan/PycharmProjects/210429_YoloV3_Experiment/model/darknet.py /Users/limengfan/PycharmProjects/210429_YoloV3_ExperimentSftp/model
cp /Users/limengfan/PycharmProjects/210429_YoloV3_Experiment/model/yolov3.py /Users/limengfan/PycharmProjects/210429_YoloV3_ExperimentSftp/model
cp /Users/limengfan/PycharmProjects/210429_YoloV3_Experiment/model/yolov3decode.py /Users/limengfan/PycharmProjects/210429_YoloV3_ExperimentSftp/model
cp /Users/limengfan/PycharmProjects/210429_YoloV3_Experiment/model/yolov3loss.py /Users/limengfan/PycharmProjects/210429_YoloV3_ExperimentSftp/model
cp /Users/limengfan/PycharmProjects/210429_YoloV3_Experiment/model/yolov3net.py /Users/limengfan/PycharmProjects/210429_YoloV3_ExperimentSftp/model

#mkdir /Users/limengfan/PycharmProjects/210429_YoloV3_ExperimentSftp/outer_weights
#cp /Users/limengfan/PycharmProjects/210429_YoloV3_Experiment/outer_weights/demo_darknet53_weights.pth /Users/limengfan/PycharmProjects/210429_YoloV3_ExperimentSftp/outer_weights

mkdir /Users/limengfan/PycharmProjects/210429_YoloV3_ExperimentSftp/logs

mkdir /Users/limengfan/PycharmProjects/210429_YoloV3_ExperimentSftp/outer_map_input
mkdir /Users/limengfan/PycharmProjects/210429_YoloV3_ExperimentSftp/outer_map_input/detection_result
mkdir /Users/limengfan/PycharmProjects/210429_YoloV3_ExperimentSftp/outer_map_input/ground_truth

sftp lab2 << EOF
put -r /Users/limengfan/PycharmProjects/210429_YoloV3_ExperimentSftp /home/lenovo/data/lmf
EOF
