import conf.config
import dataset.pennfudan_dataset
import model.yolov3

"""
（1）demo weights: 

67.91% = person AP 	||	score_threhold=0.5 : F1=0.80 ; Recall=99.09% ; Precision=66.87%
mAP = 67.91%

---------------------------------------
tensor(2.9181) tensor(2.8104)
tensor(0.1682) tensor(0.1877)
tensor(20.8298) tensor(0.0001)
---------------------------------------

loss: tensor(13.4571)

（2）trained weights(Pennfudan_Test1_Epoch88-Train_Loss5.0447-Val_Loss2.9787.pth): 

---------------------------------------
tensor(0.0108) tensor(0.0102)
tensor(0.0013) tensor(0.0012)
tensor(0.0731) tensor(0.)
---------------------------------------

loss: tensor(0.0483)
"""

if __name__ == "__main__":
    # 1. 配置文件
    Config = conf.config.PennFudanConfig

    # 2. 验证集
    BATCH_SIZE = 8
    pennfudan_dataloader = dataset.pennfudan_dataset.PennFudanDataset.TrainDataloader(
        config=Config,
        batch_size=BATCH_SIZE
    )

    # 3. 初始化模型
    yolov3 = model.yolov3.YoloV3(Config)

    # 4. 遍历数据集
    EPOCH = 1
    for epoch in range(EPOCH):
        print("Epoch:", epoch)
        for batch_index, (tensord_images, tensord_boxes_list) in enumerate(pennfudan_dataloader):
            print("batch_index:", batch_index)
            for step in range(BATCH_SIZE):
                print("step:", step)
                # 4. 预测结果并记录
                image = yolov3.predict_with_loss(
                    tensord_images[step],
                    tensord_boxes_list[step],
                )
                image.show()
            exit(-1)
