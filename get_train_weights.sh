#! /bin/bash

sftp lab2 << EOF
get -r /home/lenovo/data/lmf/210429_YoloV3_ExperimentSftp/logs/Pennfudan_Test_3_2_Epoch697-Train_Loss0.8573-Val_Loss4.8586.pth /Users/limengfan/PycharmProjects/210429_YoloV3_Experiment/logs
# get -r /home/lenovo/data/lmf/210429_YoloV3_ExperimentSftp_Cpy/logs/Voc_Test4Epoch259-Train_Loss4.2447-Val_Loss4.6763.pth /Users/limengfan/PycharmProjects/210429_YoloV3_Experiment/logs
EOF
