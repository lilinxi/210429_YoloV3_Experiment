import numpy

import torch

import conf.config
import dataset.voc_dataset
import model.yolov3

if __name__ == "__main__":
    # 0. 确保每次的伪随机数相同以便于问题的复现
    numpy.random.seed(0)
    torch.manual_seed(1)

    # 1. 配置文件
    Config = conf.config.VocConfig

    # 2. 验证集
    BATCH_SIZE = 16
    voc_dataloader = dataset.voc_dataset.VOCDataset.TrainDataloader(
        config=Config,
        batch_size=BATCH_SIZE
    )

    # 3. 初始化模型
    yolov3 = model.yolov3.YoloV3(Config)

    # 4. 遍历数据集
    EPOCH = 1
    for epoch in range(EPOCH):
        print("Epoch:", epoch)
        for batch_index, (tensord_images, tensord_boxes_list) in enumerate(voc_dataloader):
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
