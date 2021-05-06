import os

import numpy

import torch.utils.data

import conf.config
import model.yolov3net, model.yolov3loss, model.yolov3
import dataset.voc_dataset
import train_utils

if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # torch.cuda.set_device(1)
    torch.multiprocessing.set_sharing_strategy('file_system')  # https://www.cnblogs.com/zhengbiqing/p/10478311.html
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    # 0. 确保每次的伪随机数相同以便于问题的复现
    numpy.random.seed(0)
    torch.manual_seed(1)

    # 1. 训练参数
    Config = conf.config.VocConfig

    print("config:\n", Config)

    # 提示 OOM 或者显存不足请调小 Batch_size
    Freeze_Train_Batch_Size = 128
    Freeze_Eval_Batch_Size = 64

    Unfreeze_Train_Batch_Size = 32
    Unfreeze_Eval_Batch_Size = 16

    Map_Batch_Size = 256

    Init_Epoch = 0  # 起始世代
    Freeze_Epoch = 50  # 冻结训练的世代
    Unfreeze_Epoch = 2000  # 总训练世代

    Freeze_Epoch_LR = 1e-3
    Unfreeze_Epoch_LR = 1e-4

    # lr warm up
    Freeze_Epoch_Gamma = 0.96  # 0.96 ^ 50 = 0.12988
    Unfreeze_Epoch_Gamma = 0.97  # 0.97 ^ 50 = 0.22, 0.97 ^ 250 = 0.0005

    Num_Workers = 12
    Suffle = True

    Image_Set = "trainval"
    Validation_Split = 0.05  # 验证集大小

    Parallel = True and Config["cuda"]

    Test_Name = "Voc_Experiment_1_4"

    #################################################################################################################

    # 2. 创建 yolo 模型，训练前一定要修改 Config 里面的 classes 参数，训练的是 YoloNet 不是 Yolo
    yolov3_net = model.yolov3net.YoloV3Net(Config)

    # 3. 开启训练模式
    yolov3_net = yolov3_net.train()

    if Config["cuda"]:
        if Parallel:
            yolov3_net = torch.nn.DataParallel(yolov3_net, device_ids=[0, 1])
        yolov3_net = yolov3_net.cuda()

    print("yolov3_net in cuda") if Config["cuda"] else print("yolov3_net not in cuda")

    # 4. 加载 darknet53 的权值作为预训练权值（加载的是 gpu 权重，要在 DataParallel 之后）
    train_utils.load_pretrained_weights(yolov3_net, Config["pretrained_weights_path"], Config["cuda"])

    # 5. 建立 loss 函数
    yolov3_loss = model.yolov3loss.YoloV3Loss(Config)

    # 6. 加载训练数据集和测试数据集
    # 6.0 划分数据集
    dataset_size = 11540  # voc trainval 长度
    indices = list(range(dataset_size))
    split = int(numpy.floor(Validation_Split * dataset_size))
    if Suffle:
        numpy.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    # Creating PT data samplers and loaders:
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    # 6.1 冻结训练数据集
    freeze_train_data_loader = dataset.voc_dataset.VOCDataset.Dataloader(
        config=Config,
        image_set=Image_Set,
        batch_size=Freeze_Train_Batch_Size,
        train=True,
        shuffle=False,  # Suffle 和 Sampler 只能有一个，Sampler 已经 Suffle 了
        num_workers=Num_Workers,
        drop_last=False,
        sampler=train_sampler,
        enhancement=True,
    )

    freeze_validate_data_loader = dataset.voc_dataset.VOCDataset.Dataloader(
        config=Config,
        image_set=Image_Set,
        batch_size=Freeze_Eval_Batch_Size,
        train=True,
        shuffle=False,  # Suffle 和 Sampler 只能有一个，Sampler 已经 Suffle 了
        num_workers=Num_Workers,
        drop_last=False,
        sampler=valid_sampler,
        enhancement=False,
    )

    # 6.2 解冻训练数据集
    unfreeze_train_data_loader = dataset.voc_dataset.VOCDataset.Dataloader(
        config=Config,
        image_set=Image_Set,
        batch_size=Unfreeze_Train_Batch_Size,
        train=True,
        shuffle=False,  # Suffle 和 Sampler 只能有一个，Sampler 已经 Suffle 了
        num_workers=Num_Workers,
        drop_last=False,
        sampler=train_sampler,
        enhancement=True,
    )

    unfreeze_validate_data_loader = dataset.voc_dataset.VOCDataset.Dataloader(
        config=Config,
        image_set=Image_Set,
        batch_size=Unfreeze_Eval_Batch_Size,
        train=True,
        shuffle=False,  # Suffle 和 Sampler 只能有一个，Sampler 已经 Suffle 了
        num_workers=Num_Workers,
        drop_last=False,
        sampler=valid_sampler,
        enhancement=False,
    )

    # 6.3 mAP 计算数据集
    mAP_train_data_loader = dataset.voc_dataset.VOCDataset.Dataloader(
        config=Config,
        image_set=Image_Set,
        batch_size=Map_Batch_Size,
        train=False,
        shuffle=False,  # Suffle 和 Sampler 只能有一个，Sampler 已经 Suffle 了
        num_workers=Num_Workers,
        drop_last=False,
        sampler=train_sampler,
        enhancement=True,
    )

    mAP_validate_data_loader = dataset.voc_dataset.VOCDataset.Dataloader(
        config=Config,
        image_set=Image_Set,
        batch_size=Map_Batch_Size,
        train=False,
        shuffle=False,  # Suffle 和 Sampler 只能有一个，Sampler 已经 Suffle 了
        num_workers=Num_Workers,
        drop_last=False,
        sampler=valid_sampler,
        enhancement=False,
    )

    # 初始 map
    yolov3 = model.yolov3.YoloV3(Config, with_net=False)
    yolov3.net = yolov3_net.eval()

    # print("mAP_train_data_loader:")
    # train_utils.compute_map(yolov3, mAP_train_data_loader, Config["cuda"])
    # print("mAP_validate_data_loader:")
    # train_utils.compute_map(yolov3, mAP_validate_data_loader, Config["cuda"])

    # 7. 粗略训练预测头

    # 7.1 优化器和学习率调整器
    optimizer = torch.optim.Adam(yolov3_net.parameters(), Freeze_Epoch_LR)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=Freeze_Epoch_Gamma)

    # 7.2 冻结特征网络
    if Parallel:
        for param in yolov3_net.module.backbone.parameters():
            param.requires_grad = False
    else:
        for param in yolov3_net.backbone.parameters():
            param.requires_grad = False

    # 7.3 训练若干 Epoch
    for epoch in range(Init_Epoch, Freeze_Epoch):
        train_utils.train_one_epoch(
            Test_Name,
            yolov3_net,  # 网络模型
            yolov3_loss,  # 损失函数
            optimizer,  # 优化器
            epoch,  # 当前 epoch
            len(freeze_train_data_loader),  # 训练集批次数
            len(freeze_validate_data_loader),  # 验证集批次数
            Freeze_Epoch,  # 总批次
            freeze_train_data_loader,  # 训练集
            freeze_validate_data_loader,  # 验证集
            Config["cuda"],
        )
        lr_scheduler.step()  # 更新步长
        # 计算 mAP
        if (epoch + 1) % 10 == 0:
            yolov3.net = yolov3_net.eval()
            # print("\nmAP_train_data_loader:")
            train_utils.compute_map(yolov3, mAP_train_data_loader, Config["cuda"])
            # print("\nmAP_validate_data_loader:")
            train_utils.compute_map(yolov3, mAP_validate_data_loader, Config["cuda"])

    # 8. 精细训练预测头和特征网络

    # 8.1 优化器和学习率调整器
    optimizer = torch.optim.Adam(yolov3_net.parameters(), Unfreeze_Epoch_LR)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=Unfreeze_Epoch_Gamma)

    # 8.2 解冻特征网络
    if Parallel:
        for param in yolov3_net.module.backbone.parameters():
            param.requires_grad = True
    else:
        for param in yolov3_net.backbone.parameters():
            param.requires_grad = True

    # 8.3 训练若干 Epoch
    for epoch in range(Freeze_Epoch, Unfreeze_Epoch):
        train_utils.train_one_epoch(
            Test_Name,
            yolov3_net,  # 网络模型
            yolov3_loss,  # 损失函数
            optimizer,  # 优化器
            epoch,  # 当前 epoch
            len(unfreeze_train_data_loader),  # 训练集批次数
            len(unfreeze_validate_data_loader),  # 验证集批次数
            Unfreeze_Epoch,  # 总批次
            unfreeze_train_data_loader,  # 训练集
            unfreeze_validate_data_loader,  # 验证集
            Config["cuda"],
        )
        lr_scheduler.step()  # 更新步长
        # 计算 mAP
        if (epoch + 1) % 5 == 0:
            yolov3.net = yolov3_net.eval()
            # print("\nmAP_train_data_loader:")
            train_utils.compute_map(yolov3, mAP_train_data_loader, Config["cuda"])
            # print("\nmAP_validate_data_loader:")
            train_utils.compute_map(yolov3, mAP_validate_data_loader, Config["cuda"])
