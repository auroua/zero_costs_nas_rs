from detectron2.data import DatasetMapper, build_detection_train_loader


def build_dataloader(cfg, build_aug):
    mapper = DatasetMapper(cfg, is_train=True, augmentations=build_aug(cfg))
    return build_detection_train_loader(cfg, mapper=mapper)