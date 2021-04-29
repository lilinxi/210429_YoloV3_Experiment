import conf.config
import dataset.pennfudan_dataset
import model.yolov3

"""
（1）demo weights: 

67.91% = person AP 	||	score_threhold=0.5 : F1=0.80 ; Recall=99.09% ; Precision=66.87%
mAP = 67.91%

（2）trained weights(Pennfudan_Test1_Epoch88-Train_Loss5.0447-Val_Loss2.9787.pth): 

69.02% = person AP 	||	score_threhold=0.5 : F1=0.82 ; Recall=79.09% ; Precision=85.29%
mAP = 69.02%
"""

if __name__ == "__main__":
    # 1. 配置文件
    Config = conf.config.PennFudanConfig

    # 2. 验证集
    BATCH_SIZE = 256
    pennfudan_dataloader = dataset.pennfudan_dataset.PennFudanDataset.EvalDataloader(
        config=Config,
        batch_size=BATCH_SIZE
    )

    # 3. 初始化模型
    yolov3 = model.yolov3.YoloV3(Config)

    # 4. 遍历数据集
    EPOCH = 1
    for epoch in range(EPOCH):
        print("Epoch:", epoch)
        for batch_index, (tensord_images, truth_annotation_list) in enumerate(pennfudan_dataloader):
            print("batch_index:", batch_index)
            if Config["cuda"]:
                tensord_images = tensord_images.cuda()
            for step in range(BATCH_SIZE):
                print("step:", step)
                # 4. 预测结果并记录
                yolov3.predict_with_eval(
                    tensord_images[step],
                    truth_annotation_list[step],
                )
