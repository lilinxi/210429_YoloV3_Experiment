import os
from typing import List
import collections

import numpy
import PIL.Image, PIL.ImageDraw, PIL.ImageFont

import torch
import torchvision

import model.yolov3net
import model.yolov3decode
import model.yolov3loss
import dataset.transform


# -----------------------------------------------------------------------------------------------------------#
# class YoloV3(object) # YoloV3 预测网络
# -----------------------------------------------------------------------------------------------------------#

class YoloV3(object):
    """
    对 YoloV3 的三层预测结果提供解析
    """

    def __init__(self, config: dict) -> None:
        super().__init__()

        self.config = config
        self.cuda = self.config["cuda"]
        self.generate()

    def generate(self):
        """
        初始化预测模型和解析工具
        :return:
        """
        print("YoloV3 generate...")
        # 1. 生成模型
        self.net = model.yolov3net.YoloV3Net(self.config)
        if self.cuda:
            self.net = self.net.cuda()
        # 2. 加载模型权重
        device = torch.device("cuda") if self.config["cuda"] else torch.device("cpu")
        state_dict = torch.load(self.config["weights_path"], map_location=device)

        # 3. 加载多卡并行 gpu 模型
        new_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            if k[:6] == "module":
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict

        self.net.load_state_dict(state_dict)
        # 3. 网络开启 eval 模式
        self.net = self.net.eval()
        # 4. 初始化特征层解码器
        self.yolov3_decode = model.yolov3decode.YoloV3Decode(config=self.config)
        # 5. 预测结果恢复变换
        self.rescale_boxes = dataset.transform.RescaleBoxes(config=self.config)
        # 6. 训练结果恢复变换
        self.renorm_and_reform_boxes = dataset.transform.RenormAndReformBoxes(config=self.config)
        # 7. 损失函数
        self.yolov3_loss = model.yolov3loss.YoloV3Loss(config=self.config)
        print("YoloV3 generate Success")

    def predict(self, tensord_image: torch.Tensor) -> (numpy.ndarray, List[torch.Tensor]):
        """
        检测图片
        """
        with torch.no_grad():  # 1. 没有梯度传递，进行图像检测
            # 2. 将图像输入网络当中进行预测
            predict_feature_list = self.net(tensord_image.unsqueeze(dim=0))
            predict_bbox_attrs_list = []  # 预测框列表

            # 3. 对三种大小的预测框进行解析
            for index in range(3):  # 顺序：大，中，小
                predict_bbox_attrs_list.append(self.yolov3_decode(predict_feature_list[index]))

            # 4. 将预测框进行堆叠
            predict_bbox_attrs = torch.cat(predict_bbox_attrs_list, 1)  # 按第一个维度拼接：13*13*3 + 26*26*3 + 52*52*3 = 10647

            # 5. 进行非极大抑制
            predict_bbox_attrs_after_nms = model.yolov3decode.non_max_suppression(
                predict_bbox_attrs,  # 预测框列表
                conf_threshold=self.config["conf_threshold"],  # 置信度
                nms_iou_threshold=self.config["nms_iou_threshold"]  # iou 阈值
            )

        if predict_bbox_attrs_after_nms[0] == None:
            return None, predict_feature_list
        else:
            return numpy.around(predict_bbox_attrs_after_nms[0].cpu().numpy()).astype(numpy.int), predict_feature_list

    def predict_with_eval(self, tensord_image: torch.Tensor, truth_annotation: dict) -> None:
        # 1. 记录真值框
        ground_truth = open(
            os.path.join(
                os.getcwd(),
                "outer_map_input",
                "ground_truth",
                truth_annotation["filename"].split(".")[0] + ".txt"
            ),
            "w"
        )
        for truth_box in truth_annotation["boxes"]:
            (xmin, ymin, xmax, ymax, label) = truth_box
            ground_truth.write("%s %s %s %s %s\n" % (self.config["labels"][label], xmin, ymin, xmax, ymax))

        # 2. 记录预测框
        detection_result = open(
            os.path.join(
                os.getcwd(),
                "outer_map_input",
                "detection_result",
                truth_annotation["filename"].split(".")[0] + ".txt"
            ),
            "w"
        )
        # 3. 获取预测框
        predict_boxes, _ = self.predict(tensord_image)
        if predict_boxes is None:
            print("predict_boxes is None")
            return
        # 4. 解析预测框
        predict_boxes = self.rescale_boxes(truth_annotation["raw_image"], predict_boxes)
        # 5. 写入预测框
        for predict_box in predict_boxes:
            (xmin, ymin, xmax, ymax, conf, label) = predict_box
            detection_result.write(
                "%s %s %s %s %s %s\n" % (self.config["labels"][label], conf, xmin, ymin, xmax, ymax))

    def predict_with_loss(self, tensord_image: torch.Tensor, tensord_boxes: torch.Tensor) -> PIL.Image.Image:
        # 1. 转为 PIL.Image.Image
        image = torchvision.transforms.ToPILImage()(tensord_image)
        # 2. 绘制真值框（绿）
        scaled_boxes = self.renorm_and_reform_boxes(tensord_boxes.cpu())
        for truth_box in scaled_boxes:
            (xmin, ymin, xmax, ymax, label) = truth_box
            draw = PIL.ImageDraw.Draw(image)
            draw.rectangle([xmin, ymin, xmax, ymax], outline="#00FF00")
            # 绘制标签
            font = PIL.ImageFont.truetype("/Users/limengfan/PycharmProjects/210414_CfgYoloV3/assets/simhei.ttf", 32)
            draw.text([xmin, ymin, xmax, ymax], self.config["labels"][label], font=font, fill="#00FF00")
            del draw
        # 3. 获取预测框
        predict_boxes, predict_feature_list = self.predict(tensord_image)
        if predict_boxes is None:
            print("predict_boxes is None")
            return image
        # 4. 绘制预测框（红）
        for predict_box in predict_boxes:
            (xmin, ymin, xmax, ymax, conf, label) = predict_box
            draw = PIL.ImageDraw.Draw(image)
            draw.rectangle([xmin, ymin, xmax, ymax], outline="#FF0000")
            # 绘制标签
            font = PIL.ImageFont.truetype("/Users/limengfan/PycharmProjects/210414_CfgYoloV3/assets/simhei.ttf", 32)
            draw.text([xmin, ymin, xmax, ymax], self.config["labels"][label], font=font, fill="#FF0000")
            del draw
        # 5. 计算损失
        loss = self.yolov3_loss(predict_feature_list, [tensord_boxes])
        print("predict_boxes:", predict_boxes)
        print("tensord_boxes:", tensord_boxes)
        print("loss:", loss)
        return image
