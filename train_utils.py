import os

import tqdm

import torch
import torch.optim
import torch.utils.data.dataloader

import model.yolov3net, model.yolov3loss


def compute_map(
        yolov3: model.yolov3net.YoloV3Net,
        validate_data_loader: torch.utils.data.dataloader.DataLoader,  # 验证集
        cuda: bool,
):
    os.system("rm -rf ./outer_map_input")
    os.system("mkdir outer_map_input")
    os.system("mkdir outer_map_input/detection_result")
    os.system("mkdir outer_map_input/ground_truth")

    yolov3 = yolov3.eval()
    for batch_index, (tensord_images, truth_annotation_list) in enumerate(validate_data_loader):
        print("batch_index:", batch_index)
        if cuda:
            tensord_images = tensord_images.cuda()
        for step in range(tensord_images.shape[0]):
            print("step:", step)
            # 4. 预测结果并记录
            yolov3.predict_with_eval(
                tensord_images[step],
                truth_annotation_list[step],
            )

    os.system("python3 ./compute_map.py")
    os.system("rm -rf ./outer_map_input")


def train_one_epoch(
        test_name: str,  # 实验名称
        yolov3_net: model.yolov3net.YoloV3Net,  # 网络模型
        yolov3_losses: model.yolov3loss.YoloV3Loss,  # 损失函数
        optimizer: torch.optim.Optimizer,  # 优化器
        epoch: int,  # 当前 epoch
        train_batch_num: int,  # 训练集的批次数，即为训练集大小除以批次大小
        validate_batch_num: int,  # 验证集的批次数，即为验证集大小除以批次大小
        total_epoch: int,  # 总批次
        train_data_loader: torch.utils.data.dataloader.DataLoader,  # 训练集
        validate_data_loader: torch.utils.data.dataloader.DataLoader,  # 验证集
        cuda: bool,
) -> None:
    """
    训练一个 epoch
    :return:
    """

    # -----------------------------------------------------------------------------------------------------------#
    # step1. 训练
    # -----------------------------------------------------------------------------------------------------------#
    total_train_loss = 0  # 当前 epoch 的训练总损失

    # 1. 打开网络训练模式
    yolov3_net = yolov3_net.train()

    # torch.save(yolov3_net.state_dict(), "logs/" + "begin" + ".pth")

    # 2. 加载 tadm 进度条，
    with tqdm.tqdm(total=train_batch_num, desc=f"Epoch {epoch + 1}/{total_epoch}", postfix=dict) as pbar:
        # 3. 批次遍历数据集
        for iteration, (tensord_images, tensord_target_list) in enumerate(train_data_loader):
            if cuda:
                tensord_images = tensord_images.cuda()

            # print("train in cuda") if cuda else print("train not in cuda")

            # 4. 清零梯度
            optimizer.zero_grad()

            # 5. 前向传播
            predict_feature_list = yolov3_net(tensord_images)

            # 6. 计算损失
            loss = yolov3_losses(predict_feature_list, tensord_target_list)

            # 7. 反向传播
            loss.backward()

            # 8. 优化器优化参数
            optimizer.step()

            # 9. 进度条更新
            total_train_loss += loss.item()
            pbar.set_postfix(
                **{
                    "lr": optimizer.param_groups[0]["lr"],  # 优化器的当前学习率
                    "train_loss": total_train_loss / (iteration + 1),  # 当前 epoch 的训练总损失 / 迭代次数
                }
            )
            pbar.update(1)  # 进度条更新

    # -----------------------------------------------------------------------------------------------------------#
    # step2. 验证
    # -----------------------------------------------------------------------------------------------------------#
    total_validate_loss = 0  # 当前 epoch 的验证总损失

    # 1. 打开网络验证模式
    yolov3_net = yolov3_net.eval()

    # 2. 加载 tadm 进度条，
    with tqdm.tqdm(total=validate_batch_num, desc=f"Epoch {epoch + 1}/{total_epoch}", postfix=dict) as pbar:
        # 3. 批次遍历数据集
        for iteration, (tensord_images, tensord_target_list) in enumerate(validate_data_loader):
            if cuda:
                tensord_images = tensord_images.cuda()

            # print("eval in cuda") if cuda else print("eval not in cuda")

            # 4. 清零梯度
            optimizer.zero_grad()

            # 5. 前向传播
            predict_feature_list = yolov3_net(tensord_images)

            # 6. 计算损失
            loss = yolov3_losses(predict_feature_list, tensord_target_list)

            # 7. 进度条更新
            total_validate_loss += loss.item()
            pbar.set_postfix(
                **{
                    "validate_loss": total_validate_loss / (iteration + 1),  # 当前 epoch 的验证总损失 / 迭代次数
                }
            )
            pbar.update(1)  # 进度条更新

    # -----------------------------------------------------------------------------------------------------------#
    # step3. 结果
    # -----------------------------------------------------------------------------------------------------------#
    # 1. 计算平均损失
    train_loss = total_train_loss / train_batch_num
    validate_loss = total_validate_loss / validate_batch_num

    # 2. 显示结果
    ret = "Epoch%04d-Train_Loss%.4f-Val_Loss%.4f" % (epoch + 1, train_loss, validate_loss)
    # print(ret)

    # 3. 保存权重
    torch.save(
        yolov3_net.state_dict(),
        os.path.join(os.path.join(os.getcwd(), "logs"), test_name + "_" + ret + ".pth")
    )

    # step4. 计算 mAP
    compute_map(yolov3_net, validate_data_loader, cuda)


def load_pretrained_weights(net: torch.nn.Module, weights_path: str, cuda: bool):
    """
    加载预训练权重中名称相符的部分

    :param net: 网络
    :param weights_path: 预训练权重路径
    :param cuda: 是否使用 gpu
    :return:
    """
    print("Loading weights into state dict...", weights_path)
    print("weights in cuda") if cuda else print("weights not in cuda")

    # 1. 确定设备
    device = torch.device("cuda" if cuda else "cpu")

    # 2. 获取网络权重字典和预训练权重字典
    net_dict = net.state_dict()
    pretrained_dict = torch.load(weights_path, map_location=device)

    # 3. 将 pretrained_dict 里不属于 net_dict 的键剔除掉
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict}

    # 4. 将 pretrained_dict 的键值更新到 net_dict
    net_dict.update(pretrained_dict)

    # 5. net 加载 net_dict
    net.load_state_dict(net_dict)

    print("Loading weights into state dict Success！")
