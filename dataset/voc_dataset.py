import numpy
import PIL.Image

import torch.utils.data
import torchvision

import conf.config
import dataset.transform


class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, config: dict, image_set: str, train: bool, enhancement: bool) -> None:
        super().__init__()

        self.config = config
        self.voc2012_dataset = torchvision.datasets.VOCDetection(
            root=self.config["dataset_root"],
            image_set=image_set,
        )
        self.transforms: dataset.transform.Compose = dataset.transform.get_transforms(self.config, train, enhancement)

    def __getitem__(self, index: int) -> (PIL.Image.Image, dict):  # 训练集 -> (PIL.Image.Image, numpy.ndarry):
        (raw_image, raw_annotation) = self.voc2012_dataset[index]

        truth_annotation = {}

        raw_boxes = []
        for object in raw_annotation["annotation"]["object"]:
            xmin, ymin, xmax, ymax = \
                int(object["bndbox"]["xmin"]), \
                int(object["bndbox"]["ymin"]), \
                int(object["bndbox"]["xmax"]), \
                int(object["bndbox"]["ymax"])

            raw_boxes.append(
                [
                    xmin, ymin, xmax, ymax,
                    self.config["labels"].index(object["name"]) if object["name"] in self.config["labels"] else -1
                ]
            )

        truth_annotation["boxes"] = numpy.asarray(raw_boxes)
        truth_annotation["raw_image"] = raw_image
        truth_annotation["filename"] = raw_annotation["annotation"]["filename"]

        # 5. 执行数据变换
        scaled_image, truth_annotation = self.transforms(raw_image, truth_annotation)

        # 返回索引图像及其标签结果
        return scaled_image, truth_annotation

    def __len__(self) -> int:  # 获取数据集的长度
        return len(self.voc2012_dataset)

    @staticmethod
    def Dataloader(
            config: dict,
            image_set: str,
            batch_size: int = 1,
            train: bool = False,
            shuffle: bool = False,
            num_workers: int = 0,
            drop_last: bool = True,
            sampler: torch.utils.data.Sampler = None,
            enhancement: bool = True,
    ) -> torch.utils.data.DataLoader:
        voc_dataset = VOCDataset(
            config=config,
            image_set=image_set,
            train=train,
            enhancement=enhancement,
        )

        voc_dataloader = torch.utils.data.DataLoader(
            voc_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=dataset.transform.train_collate_fn if train else dataset.transform.eval_collate_fn,
            drop_last=drop_last,
            sampler=sampler,
        )

        return voc_dataloader

    @staticmethod
    def TrainDataloader(  # 训练集
            config: dict,
            batch_size: int = 1,
            shuffle: bool = False,
            num_workers: int = 0,
    ) -> torch.utils.data.DataLoader:
        return VOCDataset.Dataloader(
            config,
            image_set="train",
            batch_size=batch_size,
            train=True,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=True,
        )

    @staticmethod
    def EvalAsTrainDataloader(  # 以训练集的格式访问验证集
            config: dict,
            batch_size: int = 1,
            shuffle: bool = False,
            num_workers: int = 0,
    ) -> torch.utils.data.DataLoader:
        return VOCDataset.Dataloader(
            config,
            image_set="val",
            batch_size=batch_size,
            train=True,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=True,
        )

    @staticmethod
    def EvalDataloader(  # 验证集
            config: dict,
            batch_size: int = 1,
            shuffle: bool = False,
            num_workers: int = 0,
    ) -> torch.utils.data.DataLoader:
        return VOCDataset.Dataloader(
            config,
            image_set="val",
            batch_size=batch_size,
            train=False,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=False,
        )

    @staticmethod
    def TrainAsEvalDataloader(  # 以验证集的格式访问训练集
            config: dict,
            batch_size: int = 1,
            shuffle: bool = False,
            num_workers: int = 0,
    ) -> torch.utils.data.DataLoader:
        return VOCDataset.Dataloader(
            config,
            image_set="train",
            batch_size=batch_size,
            train=False,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=False,
        )

    @staticmethod
    def TrainvalAsTrainDataloader(  # 以训练集的格式访问训练集-验证集合集
            config: dict,
            batch_size: int = 1,
            shuffle: bool = False,
            num_workers: int = 0,
    ) -> torch.utils.data.DataLoader:
        return VOCDataset.Dataloader(
            config,
            image_set="trainval",
            batch_size=batch_size,
            train=True,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=False,
        )

    @staticmethod
    def TrainvalAsEvalDataloader(  # 以验证集的格式访问训练集-验证集合集
            config: dict,
            batch_size: int = 1,
            shuffle: bool = False,
            num_workers: int = 0,
    ) -> torch.utils.data.DataLoader:
        return VOCDataset.Dataloader(
            config,
            image_set="trainval",
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

    voc_train_dataloader = VOCDataset.TrainDataloader(
        config=conf.config.VocConfig,
    )

    voc_eval_dataloader = VOCDataset.EvalDataloader(
        config=conf.config.VocConfig,
    )

    print(len(voc_train_dataloader))
    print(len(voc_eval_dataloader))

    for epoch in range(EPOCH):
        print("Epoch:", epoch)
        for step, (tensord_images, tensord_target_list) in enumerate(voc_train_dataloader):
            print("step:", step)
            print(tensord_images)
            print(tensord_target_list)
            break
        break

    for epoch in range(EPOCH):
        print("Epoch:", epoch)
        for step, (tensord_images, truth_annotation_list) in enumerate(voc_eval_dataloader):
            print("step:", step)
            print(tensord_images)
            print(truth_annotation_list)
            break
        break
