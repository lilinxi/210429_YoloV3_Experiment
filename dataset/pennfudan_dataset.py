import os

import numpy
import PIL.Image

import torch.utils.data

import conf.config
import dataset.transform


class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, config: dict, train: bool) -> None:
        super().__init__()

        # 1. 初始化数据集根目录和数据变换
        self.config = config
        self.root = self.config["dataset_root"]
        self.transforms: dataset.transform.Compose = dataset.transform.get_transforms(self.config, train)

        # 2. 对所有的文件路径进行排序，确保图像和蒙版对齐
        self.images_dir = os.path.join(self.root, "PNGImages")
        self.masks_dir = os.path.join(self.root, "PedMasks")
        self.images_name = list(sorted(os.listdir(self.images_dir)))
        self.masks_name = list(sorted(os.listdir(self.masks_dir)))

    def __getitem__(self, idx: int) -> (PIL.Image.Image, dict):  # 训练集 -> (PIL.Image.Image, numpy.ndarry):
        # 1. 拼接文件路径
        image_path = os.path.join(self.images_dir, self.images_name[idx])
        mask_path = os.path.join(self.masks_dir, self.masks_name[idx])

        # 2. 读取图像文件和蒙版文件
        raw_image = PIL.Image.open(image_path).convert("RGB")  # 读取图像，转化为 RGB
        mask = PIL.Image.open(mask_path)  # 读取蒙版，为灰度图：0 为背景，非 0 为实例的掩码

        # 3. 解析蒙版文件，对每个蒙版，获取一个二值掩码，得到列表
        mask = numpy.array(mask)  # 蒙版图像转化为 numpy 数组，(w, h) -> (h, w)
        obj_ids = numpy.unique(mask)  # 获取所有的实例，每组相同的像素表示一个实例，(objects + 1, )
        obj_ids = obj_ids[1:]  # 去除 0，0 表示背景，并不表示实例，(objects, )
        # (h, w) == (objects, 1, 1) -> (objects, h, w)
        masks = mask == obj_ids[:, None, None]  # 将 mask 转化为二值掩码的数组，每个实例一个二值掩码

        truth_annotation = {}

        raw_boxes = []

        # 4. 解析二值掩码，对于每个二值掩码，获取其包围盒，得到包围盒列表
        num_objs = len(obj_ids)  # 实例的数量
        for i in range(num_objs):
            pos = numpy.where(masks[i])  # 获取所有的掩码像素的坐标，pos: (h, w)
            # 获取掩码的包围盒的坐标
            xmin = numpy.min(pos[1])
            xmax = numpy.max(pos[1])
            ymin = numpy.min(pos[0])
            ymax = numpy.max(pos[0])
            raw_boxes.append([xmin, ymin, xmax, ymax, 0])  # 这里并没有对实例进行分类，只有一类，所有的实例都为分类 1，其 index 为 0

        truth_annotation["boxes"] = numpy.asarray(raw_boxes)
        truth_annotation["raw_image"] = raw_image
        truth_annotation["filename"] = self.images_name[idx]

        # 5. 执行数据变换
        scaled_image, truth_annotation = self.transforms(raw_image, truth_annotation)

        # 返回索引图像及其标签结果
        return scaled_image, truth_annotation

    def __len__(self) -> int:  # 获取数据集的长度
        return len(self.images_name)

    @staticmethod
    def Dataloader(
            config: dict,
            batch_size: int = 1,
            train: bool = False,
            shuffle: bool = False,
            num_workers: int = 0,
            drop_last: bool = True,
            sampler: torch.utils.data.Sampler = None,
    ) -> torch.utils.data.DataLoader:
        pennfudan_dataset = PennFudanDataset(
            config=config,
            train=train
        )

        pennfudan_dataloader = torch.utils.data.DataLoader(
            pennfudan_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=dataset.transform.train_collate_fn if train else dataset.transform.eval_collate_fn,
            drop_last=drop_last,
            sampler=sampler,
        )

        return pennfudan_dataloader

    @staticmethod
    def TrainDataloader(  # 训练集
            config: dict,
            batch_size: int = 1,
            shuffle: bool = False,
            num_workers: int = 0,
    ) -> torch.utils.data.DataLoader:
        return PennFudanDataset.Dataloader(
            config,
            batch_size=batch_size,
            train=True,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=True,
        )

    @staticmethod
    def EvalDataloader(  # 训练集
            config: dict,
            batch_size: int = 1,
            shuffle: bool = False,
            num_workers: int = 0,
    ) -> torch.utils.data.DataLoader:
        return PennFudanDataset.Dataloader(
            config,
            batch_size=batch_size,
            train=False,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=False,
        )


# -----------------------------------------------------------------------------------------------------------#
# Test
# -----------------------------------------------------------------------------------------------------------#

if __name__ == "__main__":
    EPOCH = 2

    pennfudan_train_dataloader = PennFudanDataset.TrainDataloader(
        config=conf.config.VocConfig,
    )

    pennfudan_eval_dataloader = PennFudanDataset.EvalDataloader(
        config=conf.config.VocConfig,
    )

    print(len(pennfudan_train_dataloader))
    print(len(pennfudan_eval_dataloader))

    for epoch in range(EPOCH):
        print("Epoch:", epoch)
        for step, (tensord_images, tensord_target_list) in enumerate(pennfudan_train_dataloader):
            print("step:", step)
            print(tensord_images)
            print(tensord_target_list)
            break
        break

    for epoch in range(EPOCH):
        print("Epoch:", epoch)
        for step, (tensord_images, truth_annotation_list) in enumerate(pennfudan_eval_dataloader):
            print("step:", step)
            print(tensord_images)
            print(truth_annotation_list)
            break
        break
