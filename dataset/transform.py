from typing import List

import numpy
import PIL.Image

import torch
import torchvision


def rand(min: float, max: float) -> float:
    return numpy.random.rand() * (max - min) + min


class Compose(object):
    """
    复合多种变换操作
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, boxes):
        for transform in self.transforms:
            image, boxes = transform(image, boxes)

        return image, boxes


def get_transforms(config: dict, train: bool) -> Compose:
    """
    :param train: 是否是训练集，训练集包含额外的数据增强变换，且训练集只返回标注框，验证集返回标注字典
    :return:
    """
    transforms = []
    if train:
        transforms.append(ReformAndExtractBoxes())
        # transforms.append(ScaleImageAndBoxes(config=config))
        transforms.append(RandomScaleImageAndBoxes(config=config))
        transforms.append(RandomTransformImage())
        transforms.append(RandomFlipImageAndBoxes(config=config))
        transforms.append(NormImageAndBoxes(config=config))
    else:
        transforms.append(ScaleImage(config=config))
        transforms.append(NormImage(config=config))

    return Compose(transforms)


class ReformAndExtractBoxes(object):
    """
    从标注数据中提取包围盒，并变换包围盒的格式
    boxes (xmin, ymin, xmax, ymax, label) -> (x, y, w, h, label)
    """

    def __call__(self, raw_image: PIL.Image.Image, truth_annotation: dict) -> (PIL.Image.Image, numpy.ndarray):
        raw_boxes = []
        for box in truth_annotation["boxes"]:
            xmin, ymin, xmax, ymax, label = box
            raw_x = (xmax + xmin) / 2
            raw_y = (ymax + ymin) / 2
            raw_w = xmax - xmin
            raw_h = ymax - ymin
            raw_boxes.append([raw_x, raw_y, raw_w, raw_h, label])
        return raw_image, numpy.asarray(raw_boxes).astype(numpy.float32)


class ScaleImageAndBoxes(object):
    """
    boxes 和 image 的 等比例放缩
    """

    def __init__(self, config: dict) -> None:
        super().__init__()

        self.config = config

    def __call__(self, raw_image: PIL.Image.Image, raw_boxes: numpy.ndarray) -> (PIL.Image.Image, numpy.ndarray):
        # 1. 图像原始大小，图像放缩后大小
        raw_width, raw_height = raw_image.size
        scaled_width = self.config["image_width"]
        scaled_height = self.config["image_height"]

        # 2. 计算图像放缩倍数，取最小的那个放缩值
        scale = min(scaled_width / raw_width, scaled_height / raw_height)

        # 3. 等比例放缩后的图像大小
        nw = int(raw_width * scale)
        nh = int(raw_height * scale)

        # 4. 图像等比例放缩
        scaled_image = raw_image.resize((nw, nh), PIL.Image.BICUBIC)

        # 5. 填补图像边缘
        new_image = PIL.Image.new("RGB", (scaled_width, scaled_height), (128, 128, 128))  # 创建一张灰色底板作为返回的图像
        new_image.paste(scaled_image, ((scaled_width - nw) // 2, (scaled_height - nh) // 2))  # 等比例放缩后的图像粘贴到底板中央

        # 6. 变换 boxes
        scaled_boxes = raw_boxes.copy()
        scaled_boxes[:, 0:4] = raw_boxes[:, 0:4] * scale
        scaled_boxes[:, 0] += (scaled_width - nw) // 2
        scaled_boxes[:, 1] += (scaled_height - nh) // 2

        return new_image, scaled_boxes


class RandomScaleImageAndBoxes(object):
    """
    boxes 和 image 的 等随机比例放缩
    """

    def __init__(self, config: dict) -> None:
        super().__init__()

        self.config = config

    def __call__(self, raw_image: PIL.Image.Image, raw_boxes: numpy.ndarray) -> (PIL.Image.Image, numpy.ndarray):
        # 1. 图像原始大小，图像放缩后大小
        raw_width, raw_height = raw_image.size
        scaled_width = self.config["image_width"]
        scaled_height = self.config["image_height"]

        # 2. 计算图像放缩倍数，取最小的那个放缩值
        scale = min(scaled_width / raw_width, scaled_height / raw_height)
        scale = rand(0.1, 1.0) * scale  # 0.1 ~ 1.0 scale

        # 3. 等比例放缩后的图像大小
        nw = int(raw_width * scale)
        nh = int(raw_height * scale)

        # 4. 图像等比例放缩
        scaled_image = raw_image.resize((nw, nh), PIL.Image.BICUBIC)

        # 4.5 随机平移
        dx = int(rand(0.0, scaled_width - nw))
        dy = int(rand(0.0, scaled_height - nh))

        # 5. 填补图像边缘
        new_image = PIL.Image.new("RGB", (scaled_width, scaled_height), (128, 128, 128))  # 创建一张灰色底板作为返回的图像
        new_image.paste(scaled_image, (dx, dy))  # 等比例放缩后的图像粘贴到底板中央

        # 6. 变换 boxes
        scaled_boxes = raw_boxes.copy()
        scaled_boxes[:, 0:4] = raw_boxes[:, 0:4] * scale
        scaled_boxes[:, 0] += dx
        scaled_boxes[:, 1] += dy

        return new_image, scaled_boxes


class RandomTransformImage(object):
    """
    随机变换图片
    """

    def __call__(self, scaled_image: PIL.Image.Image, scaled_boxes: numpy.ndarray) -> (PIL.Image.Image, numpy.ndarray):
        new_image = scaled_image
        if rand(0.0, 1.0) < 0.5:
            new_image = torchvision.transforms.ColorJitter(
                brightness=(1.0, 10.0),  # 亮度的偏移幅度
                # contrast=(1.0, 10.0),  # 对比度偏移幅度
                # saturation=(1.0, 10.0),  # 饱和度偏移幅度
                # hue=(0.2, 0.4),  # 色相偏移幅度
            )(scaled_image)
        if rand(0.0, 1.0) < 0.5:
            new_image = torchvision.transforms.ColorJitter(
                # brightness=(1.0, 10.0),  # 亮度的偏移幅度
                contrast=(1.0, 10.0),  # 对比度偏移幅度
                # saturation=(1.0, 10.0),  # 饱和度偏移幅度
                # hue=(0.2, 0.4),  # 色相偏移幅度
            )(scaled_image)
        if rand(0.0, 1.0) < 0.5:
            new_image = torchvision.transforms.ColorJitter(
                # brightness=(1.0, 10.0),  # 亮度的偏移幅度
                # contrast=(1.0, 10.0),  # 对比度偏移幅度
                saturation=(1.0, 10.0),  # 饱和度偏移幅度
                # hue=(0.2, 0.4),  # 色相偏移幅度
            )(scaled_image)
        if rand(0.0, 1.0) < 0.01:
            new_image = torchvision.transforms.Grayscale(num_output_channels=3)(new_image)

        return new_image, scaled_boxes


class RandomFlipImageAndBoxes(object):
    """
    随机翻转图片
    """

    def __init__(self, config: dict) -> None:
        super().__init__()

        self.config = config

    def __call__(self, scaled_image: PIL.Image.Image, scaled_boxes: numpy.ndarray) -> (PIL.Image.Image, numpy.ndarray):
        new_image = scaled_image
        if rand(0.0, 1.0) < 0.5:
            new_image = torchvision.transforms.RandomHorizontalFlip(p=2)(new_image)
            scaled_boxes[:, 0] = self.config["image_width"] - scaled_boxes[:, 0]

        if rand(0.0, 1.0) < 0.5:
            new_image = torchvision.transforms.RandomVerticalFlip(p=2)(new_image)
            scaled_boxes[:, 1] = self.config["image_height"] - scaled_boxes[:, 1]

        return new_image, scaled_boxes


class ScaleImage(object):
    """
    boxes 的 等比例放缩
    """

    def __init__(self, config: dict) -> None:
        super().__init__()

        self.config = config

    def __call__(self, raw_image: PIL.Image.Image, truth_annotation: dict) -> (PIL.Image.Image, dict):
        # 1. 图像原始大小，图像放缩后大小
        raw_width, raw_height = raw_image.size
        scaled_width = self.config["image_width"]
        scaled_height = self.config["image_height"]

        # 2. 计算图像放缩倍数，取最小的那个放缩值
        scale = min(scaled_width / raw_width, scaled_height / raw_height)

        # 3. 等比例放缩后的图像大小
        nw = int(raw_width * scale)
        nh = int(raw_height * scale)

        # 4. 图像等比例放缩
        scaled_image = raw_image.resize((nw, nh), PIL.Image.BICUBIC)

        # 5. 填补图像边缘
        new_image = PIL.Image.new("RGB", (scaled_width, scaled_height), (128, 128, 128))  # 创建一张灰色底板作为返回的图像
        new_image.paste(scaled_image, ((scaled_width - nw) // 2, (scaled_height - nh) // 2))  # 等比例放缩后的图像粘贴到底板中央

        return new_image, truth_annotation


class RescaleBoxes(object):
    """
    boxes 等比例放缩（反向）
    """

    def __init__(self, config: dict) -> None:
        super().__init__()

        self.config = config

    def __call__(self, raw_image: PIL.Image.Image, scaled_boxes: numpy.ndarray) -> numpy.ndarray:
        # 1. 图像原始大小，图像放缩后大小
        raw_width, raw_height = raw_image.size
        scaled_width = self.config["image_width"]
        scaled_height = self.config["image_height"]

        # 2. 计算图像放缩倍数，取最小的那个放缩值
        scale = min(scaled_width / raw_width, scaled_height / raw_height)

        # 3. 等比例放缩后的图像大小
        nw = int(raw_width * scale)
        nh = int(raw_height * scale)

        # 4. 变换 boxes
        rescaled_boxes = scaled_boxes.copy()
        rescaled_boxes[:, 0] -= (scaled_width - nw) // 2
        rescaled_boxes[:, 1] -= (scaled_height - nh) // 2
        rescaled_boxes[:, 2] -= (scaled_width - nw) // 2
        rescaled_boxes[:, 3] -= (scaled_height - nh) // 2
        rescaled_boxes[:, 0:4] = rescaled_boxes[:, 0:4] / scale
        rescaled_boxes = numpy.around(rescaled_boxes).astype(numpy.int)

        return rescaled_boxes


class NormImageAndBoxes(object):
    """
    boxes 和 image 的 归一化
    """

    def __init__(self, config: dict) -> None:
        super().__init__()

        self.config = config

    def __call__(self, scaled_image: PIL.Image.Image, scaled_boxes: numpy.ndarray) -> (
            numpy.ndarray, numpy.ndarray):
        # 1. 归一化 PIL.Image.Image，width * height * RGB -> channels(RGB) * height * width
        norm_image = numpy.asarray(torchvision.transforms.ToTensor()(scaled_image))
        # 2. 归一化 boxes
        norm_boxes = scaled_boxes.copy()
        norm_boxes[:, 0] /= self.config["image_width"]
        norm_boxes[:, 1] /= self.config["image_height"]
        norm_boxes[:, 2] /= self.config["image_width"]
        norm_boxes[:, 3] /= self.config["image_height"]

        return norm_image, norm_boxes


class NormImage(object):
    """
    image 的 归一化
    """

    def __init__(self, config: dict) -> None:
        super().__init__()

        self.config = config

    def __call__(self, scaled_image: PIL.Image.Image, truth_annotation: dict) -> (numpy.ndarray, dict):
        # 1. 归一化 PIL.Image.Image，width * height * RGB -> channels(RGB) * height * width
        norm_image = numpy.asarray(torchvision.transforms.ToTensor()(scaled_image))

        return norm_image, truth_annotation


class RenormAndReformBoxes(object):
    """
    从训练集的训练数据中恢复包围盒，并变换包围盒的格式
    box_num * (norm_x, norm_y, norm_w, norm_h, label) -> box_num * (xmin, ymin, xmax, ymax, label)
    """

    def __init__(self, config: dict) -> None:
        super().__init__()

        self.config = config

    def __call__(self, tensord_boxes: torch.Tensor) -> numpy.ndarray:
        numpy_boxes = tensord_boxes.numpy().copy()
        numpy_boxes[:, 0] *= self.config["image_width"]
        numpy_boxes[:, 1] *= self.config["image_height"]
        numpy_boxes[:, 2] *= self.config["image_width"]
        numpy_boxes[:, 3] *= self.config["image_height"]

        scaled_boxes = numpy_boxes.copy()
        scaled_boxes[:, 0] = numpy_boxes[:, 0] - numpy_boxes[:, 2] / 2
        scaled_boxes[:, 1] = numpy_boxes[:, 1] - numpy_boxes[:, 3] / 2
        scaled_boxes[:, 2] = numpy_boxes[:, 0] + numpy_boxes[:, 2] / 2
        scaled_boxes[:, 3] = numpy_boxes[:, 1] + numpy_boxes[:, 3] / 2

        return numpy.around(scaled_boxes).astype(numpy.int)


def train_collate_fn(batch: List[tuple]) -> (torch.Tensor, torch.Tensor):
    """
    数据集工具函数，对一个批次的数据进行解包后打包
    :param batch:
    :return:
    """
    # print("1:", type(batch), batch)                                 # batch 是一个返回值的数组：[(image, boxes), ……]
    # print("2:", *batch)                                             # *batch 将数组解包为：(image, boxes), ……
    # print("3:", type(zip(*batch)), list(zip(*batch)))               # zip 再次打包为：(image, ……) and (boxes, ……)
    norm_images, norm_boxess = zip(*batch)

    tensord_images = torch.as_tensor(norm_images)
    tensord_boxes_list = [torch.as_tensor(norm_boxes) for norm_boxes in norm_boxess]

    return tensord_images, tensord_boxes_list


def eval_collate_fn(batch: List[tuple]) -> (torch.Tensor, List[dict]):
    """
    数据集工具函数，对一个批次的数据进行解包后打包
    :param batch:
    :return:
    """
    # print("1:", type(batch), batch)                                 # batch 是一个返回值的数组：[(image, boxes), ……]
    # print("2:", *batch)                                             # *batch 将数组解包为：(image, boxes), ……
    # print("3:", type(zip(*batch)), list(zip(*batch)))               # zip 再次打包为：(image, ……) and (boxes, ……)
    norm_images, truth_annotations = zip(*batch)

    tensord_images = torch.as_tensor(norm_images)
    truth_annotation_list = list(truth_annotations)

    return tensord_images, truth_annotation_list
