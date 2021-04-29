from typing import List

import numpy

import torch


def jaccard(_box_a, _box_b):
    # 计算真实框的左上角和右下角
    b1_x1, b1_x2 = _box_a[:, 0] - _box_a[:, 2] / 2, _box_a[:, 0] + _box_a[:, 2] / 2
    b1_y1, b1_y2 = _box_a[:, 1] - _box_a[:, 3] / 2, _box_a[:, 1] + _box_a[:, 3] / 2
    # 计算先验框的左上角和右下角
    b2_x1, b2_x2 = _box_b[:, 0] - _box_b[:, 2] / 2, _box_b[:, 0] + _box_b[:, 2] / 2
    b2_y1, b2_y2 = _box_b[:, 1] - _box_b[:, 3] / 2, _box_b[:, 1] + _box_b[:, 3] / 2
    box_a = torch.zeros_like(_box_a)
    box_b = torch.zeros_like(_box_b)
    box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3] = b1_x1, b1_y1, b1_x2, b1_y2
    box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3] = b2_x1, b2_y1, b2_x2, b2_y2
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)

    inter = inter[:, :, 0] * inter[:, :, 1]
    # 计算先验框和真实框各自的面积
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    # 求IOU
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]
    # inter[:, :, 0] is the width of intersection and inter[:, :, 1] is height


def jaccard_tensor(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [A,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [B,4]
    Return:
        jaccard overlap: (tensor) Shape: [A, B]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class YoloV3Loss(torch.nn.Module):
    """
    YoloV3 损失函数
    """

    def __init__(self, config: dict) -> None:
        super().__init__()

        self.lambda_xy = 0.05  # 预测框中心误差权重
        self.lambda_wh = 0.05  # 预测框大小误差权重
        self.lambda_class = 0.5  # 预测框类别误差权重
        self.lambda_conf = 1.0  # 预测框置信度误差权重
        self.lambda_obj = 1.0  # 预测框有物体掩码误差权重
        self.lambda_noobj = 1.0  # 预测框无物体掩码误差权重

        self.normd_anchors = numpy.asarray(config["anchors"]).astype(numpy.float32)
        self.normd_anchors[:, :, 0] /= config["image_width"]
        self.normd_anchors[:, :, 1] /= config["image_height"]
        self.normd_anchors = self.normd_anchors.reshape((9, 2))
        self.normd_anchors_box = torch.cat(
            (
                torch.zeros((self.normd_anchors.shape[0], 2)),
                torch.from_numpy(self.normd_anchors)
            ), 1)

        self.classes = config["classes"]
        self.bbox_attrs = 4 + 1 + self.classes

        self.ignore_threshold = 0.3  # iou 忽略的阈值

        self.cuda = config["cuda"]

    def decode_pyramid_boxes(self,
                             pyramid_boxes_list: List[torch.Tensor],
                             pyramid_features: int,
                             predict_feature: torch.Tensor,
                             ignore_on: bool
                             ) -> (
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
            torch.Tensor, torch.Tensor
    ):
        """
        将真值框分成三个特征层，并变换成 tensor 的格式
        """
        assert pyramid_features in [13, 26, 52]

        if pyramid_features == 13:
            pyramid_anch_index_list = [0, 1, 2]
            cur_anchors = self.normd_anchors[0:3]
        elif pyramid_features == 26:
            pyramid_anch_index_list = [3, 4, 5]
            cur_anchors = self.normd_anchors[3:6]
        elif pyramid_features == 52:
            pyramid_anch_index_list = [6, 7, 8]
            cur_anchors = self.normd_anchors[6:9]
        else:
            raise Exception("unexpected error")

        batch_size = len(pyramid_boxes_list)

        # 将真值框变换为如下的 tensor 格式
        boxes_x = torch.zeros(batch_size, 3, pyramid_features, pyramid_features)
        boxes_y = torch.zeros(batch_size, 3, pyramid_features, pyramid_features)
        boxes_w = torch.zeros(batch_size, 3, pyramid_features, pyramid_features)
        boxes_h = torch.zeros(batch_size, 3, pyramid_features, pyramid_features)

        boxes_loss_weight_xw = torch.zeros(batch_size, 3, pyramid_features, pyramid_features)
        boxes_loss_weight_yh = torch.zeros(batch_size, 3, pyramid_features, pyramid_features)

        boxes_obj_conf = torch.zeros(batch_size, 3, pyramid_features, pyramid_features)
        boxes_class_conf_list = torch.zeros(batch_size, 3, pyramid_features, pyramid_features,
                                            self.classes)

        boxes_obj_mask = torch.zeros(batch_size, 3, pyramid_features, pyramid_features)
        boxes_noobj_mask = torch.ones(batch_size, 3, pyramid_features, pyramid_features)

        # 遍历这一个批次所有的图片
        for bs_i, pyramid_boxes in enumerate(pyramid_boxes_list):
            if pyramid_boxes.shape[0] == 0:
                continue

            # 将真值框的 xy 变换为相对于网格的偏移量，顺便获取真值框在 tensor 表示中的网格索引（truth_grid_x，truth_grid_y——
            truth_feature_box = pyramid_boxes[:, 0:4] * pyramid_features

            truth_grid_x = torch.floor(truth_feature_box[:, 0]).int()
            truth_grid_y = torch.floor(truth_feature_box[:, 1]).int()

            truth_x = truth_feature_box[:, 0] - truth_grid_x
            truth_y = truth_feature_box[:, 1] - truth_grid_y

            # 和真值框 iou 最大的 anchor 索引，确定真值框所在的特征层
            truth_box = pyramid_boxes[:, :4].clone().detach()
            truth_box[:, 0] = 0
            truth_box[:, 1] = 0
            normd_anch_ious = jaccard_tensor(truth_box, self.normd_anchors_box)
            max_anch_ious_index = torch.argmax(normd_anch_ious, dim=-1)

            for box_i, anch_i in enumerate(max_anch_ious_index):
                if anch_i not in pyramid_anch_index_list:
                    continue
                pyramid_anch_i = anch_i % 3

                boxes_x[bs_i, pyramid_anch_i, truth_grid_y[box_i], truth_grid_x[box_i]] = truth_x[box_i]
                boxes_y[bs_i, pyramid_anch_i, truth_grid_y[box_i], truth_grid_x[box_i]] = truth_y[box_i]

                truth_w_box = torch.log(pyramid_boxes[box_i][2] / self.normd_anchors[anch_i][0])
                truth_h_box = torch.log(pyramid_boxes[box_i][3] / self.normd_anchors[anch_i][1])
                boxes_w[bs_i, pyramid_anch_i, truth_grid_y[box_i], truth_grid_x[box_i]] = truth_w_box
                boxes_h[bs_i, pyramid_anch_i, truth_grid_y[box_i], truth_grid_x[box_i]] = truth_h_box

                boxes_loss_weight_xw[bs_i, pyramid_anch_i, truth_grid_y[box_i], truth_grid_x[box_i]] = \
                    pyramid_boxes[box_i][2]
                boxes_loss_weight_yh[bs_i, pyramid_anch_i, truth_grid_y[box_i], truth_grid_x[box_i]] = \
                    pyramid_boxes[box_i][3]

                boxes_obj_conf[bs_i, pyramid_anch_i, truth_grid_y[box_i], truth_grid_x[box_i]] = 1
                boxes_class_conf_list[bs_i, pyramid_anch_i, truth_grid_y[box_i], truth_grid_x[box_i],
                                      pyramid_boxes[box_i][4].int()] = 1

                boxes_obj_mask[bs_i, pyramid_anch_i, truth_grid_y[box_i], truth_grid_x[box_i]] = 1
                boxes_noobj_mask[bs_i, pyramid_anch_i, truth_grid_y[box_i], truth_grid_x[box_i]] = 0  # 可以确定一定有物体

        # print("loss in cuda") if self.cuda else print("loss not in cuda")

        if ignore_on:
            # ----------------------------------------------------------------------------------------------------- #
            # 一些预测框和真值框 iou 较大的地方，有可能有物体
            cur_anchors_num = 3
            predict_feature_height = pyramid_features
            predict_feature_width = pyramid_features
            # 4. 将预测网络输出的特征层进行维度变换，将预测框个数与预测属性分开，并将预测属性转置为末位维度的属性，便于提取和解析
            predict_feature = predict_feature.contiguous().view(
                batch_size,
                cur_anchors_num,
                self.bbox_attrs,
                predict_feature_height,
                predict_feature_width,
            ).permute(0, 1, 3, 4, 2).contiguous()

            # 5. 分隔预测属性
            predict_x = predict_feature[..., 0]
            predict_y = predict_feature[..., 1]
            predict_w = predict_feature[..., 2]
            predict_h = predict_feature[..., 3]

            # 6. 解析 xy
            norm_predict_x = torch.sigmoid(predict_x)
            norm_predict_y = torch.sigmoid(predict_y)
            # 6.1 构造 grid tensor
            grid_x = torch.linspace(0, predict_feature_width - 1, predict_feature_width) \
                .repeat(predict_feature_height, 1) \
                .repeat(batch_size * cur_anchors_num, 1, 1) \
                .view(predict_x.shape)
            grid_y = torch.linspace(0, predict_feature_height - 1, predict_feature_height) \
                .repeat(predict_feature_width, 1) \
                .t() \
                .repeat(batch_size * cur_anchors_num, 1, 1) \
                .view(predict_y.shape)
            if self.cuda:
                grid_x = grid_x.cuda()
                grid_y = grid_y.cuda()
            # 6.2 叠加 grid tensor
            grid_predict_x = norm_predict_x + grid_x
            grid_predict_y = norm_predict_y + grid_y
            # 6.3 归一化 x，y
            normd_predict_x = grid_predict_x / predict_feature_width
            normd_predict_y = grid_predict_y / predict_feature_height

            # 7. 解析 wh
            # 7.1 构造 anchor tensor
            anchor_width = torch.Tensor(cur_anchors)[:, 0].unsqueeze(dim=1)
            anchor_height = torch.Tensor(cur_anchors)[:, 1].unsqueeze(dim=1)
            grid_anchor_width = anchor_width.repeat(batch_size, 1). \
                repeat(1, 1, predict_feature_height * predict_feature_width). \
                view(predict_w.shape)
            grid_anchor_height = anchor_height.repeat(batch_size, 1). \
                repeat(1, 1, predict_feature_height * predict_feature_width). \
                view(predict_h.shape)
            if self.cuda:
                grid_anchor_width = grid_anchor_width.cuda()
                grid_anchor_height = grid_anchor_height.cuda()
            # 7.2 乘以 anchor tensor
            anchord_predict_width = torch.exp(predict_w) * grid_anchor_width
            anchord_predict_height = torch.exp(predict_h) * grid_anchor_height
            # 6.3 归一化 w, h
            normd_predict_w = anchord_predict_width / predict_feature_width
            normd_predict_h = anchord_predict_height / predict_feature_height

            normd_predict_boxes = torch.cat(
                [
                    normd_predict_x.unsqueeze(dim=4),
                    normd_predict_y.unsqueeze(dim=4),
                    normd_predict_w.unsqueeze(dim=4),
                    normd_predict_h.unsqueeze(dim=4),
                ], dim=-1
            )

            for bs_i, pyramid_boxes in enumerate(pyramid_boxes_list):
                if self.cuda:
                    pyramid_boxes = pyramid_boxes.cuda()
                bs_normd_predict_boxes = normd_predict_boxes[bs_i].view(-1, 4)
                predict_truth_ious = jaccard(pyramid_boxes[..., :4], bs_normd_predict_boxes)
                predict_truth_ious_max, _ = torch.max(predict_truth_ious, dim=0)
                predict_truth_ious_max = predict_truth_ious_max.view(normd_predict_boxes[bs_i].size()[:3])
                # a = predict_truth_ious_max > self.ignore_threshold
                # aa = torch.unique(a)
                # print(aa.size())
                boxes_noobj_mask[bs_i][predict_truth_ious_max > self.ignore_threshold] = 0

        if self.cuda:
            return boxes_x.cuda(), \
                   boxes_y.cuda(), \
                   boxes_w.cuda(), \
                   boxes_h.cuda(), \
                   boxes_loss_weight_xw.cuda(), \
                   boxes_loss_weight_yh.cuda(), \
                   boxes_obj_conf.cuda(), \
                   boxes_class_conf_list.cuda(), \
                   boxes_obj_mask.cuda(), \
                   boxes_noobj_mask.cuda()

        return boxes_x, \
               boxes_y, \
               boxes_w, \
               boxes_h, \
               boxes_loss_weight_xw, \
               boxes_loss_weight_yh, \
               boxes_obj_conf, \
               boxes_class_conf_list, \
               boxes_obj_mask, \
               boxes_noobj_mask

    def compute_loss(self, predict_feature: torch.Tensor, decoded_boxes) -> (
            torch.Tensor, torch.Tensor):
        """
        逐个特征层计算损失
        """
        (boxes_x, boxes_y, boxes_w, boxes_h, boxes_loss_weight_xw, boxes_loss_weight_yh, boxes_obj_conf,
         boxes_class_conf_list, boxes_obj_mask, boxes_noobj_mask) = decoded_boxes

        predict_feature = predict_feature.view(
            predict_feature.shape[0],
            3,
            self.bbox_attrs,
            predict_feature.shape[2],
            predict_feature.shape[3],
        ).permute(0, 1, 3, 4, 2).contiguous()

        predict_x = torch.sigmoid(predict_feature[..., 0])
        predict_y = torch.sigmoid(predict_feature[..., 1])
        predict_w = predict_feature[..., 2]
        predict_h = predict_feature[..., 3]
        predict_obj_conf = torch.sigmoid(predict_feature[..., 4])
        predict_class_conf_list = torch.sigmoid(predict_feature[..., 5:])

        boxes_loss_scale = 2 - boxes_loss_weight_xw * boxes_loss_weight_yh

        loss_x = torch.sum(torch.nn.BCELoss()(predict_x, boxes_x) * boxes_loss_scale * boxes_obj_mask)
        loss_y = torch.sum(torch.nn.BCELoss()(predict_y, boxes_y) * boxes_loss_scale * boxes_obj_mask)

        loss_w = torch.sum(torch.nn.MSELoss()(predict_w, boxes_w) * 0.5 * boxes_loss_scale * boxes_obj_mask)
        loss_h = torch.sum(torch.nn.MSELoss()(predict_h, boxes_h) * 0.5 * boxes_loss_scale * boxes_obj_mask)

        loss_conf = self.lambda_obj * torch.sum(
            torch.nn.BCELoss()(predict_obj_conf, boxes_obj_mask) * boxes_obj_mask) + \
                    self.lambda_noobj * torch.sum(
            torch.nn.BCELoss()(predict_obj_conf, boxes_obj_mask) * boxes_noobj_mask)

        loss_class = torch.sum(torch.nn.BCELoss()(predict_class_conf_list[boxes_obj_mask == 1],
                                                  boxes_class_conf_list[boxes_obj_mask == 1]))

        # print("\n---------------------------------------")
        # print(loss_x, loss_y)
        # print(loss_w, loss_h)
        # print(loss_conf, loss_class)
        # print("---------------------------------------\n")

        loss = loss_x * self.lambda_xy + loss_y * self.lambda_xy + \
               loss_w * self.lambda_wh + loss_h * self.lambda_wh + \
               loss_conf * self.lambda_conf + loss_class * self.lambda_class

        return loss, torch.sum(boxes_obj_mask)

    def forward(self, predict_feature_list,
                tensord_boxes_list: List[torch.Tensor]) -> torch.Tensor:
        boxes_13 = self.decode_pyramid_boxes(tensord_boxes_list, 13, predict_feature_list[0], False)
        boxes_26 = self.decode_pyramid_boxes(tensord_boxes_list, 26, predict_feature_list[1], False)
        boxes_52 = self.decode_pyramid_boxes(tensord_boxes_list, 52, predict_feature_list[2], False)

        loss_13, loss_13_num = self.compute_loss(predict_feature_list[0], boxes_13)
        loss_26, loss_26_num = self.compute_loss(predict_feature_list[1], boxes_26)
        loss_52, loss_52_num = self.compute_loss(predict_feature_list[2], boxes_52)

        loss_list = []

        if not torch.isnan(loss_13):
            loss_list.append(loss_13)
        if not torch.isnan(loss_26):
            loss_list.append(loss_26)
        if not torch.isnan(loss_52):
            loss_list.append(loss_52)

        assert len(loss_list) != 0

        loss = sum(loss_list)

        loss_num = loss_13_num + loss_26_num + loss_52_num

        return loss / loss_num
