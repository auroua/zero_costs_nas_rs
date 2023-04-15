import os
import shutil


# base_path = '/home/albert_wei/fdisk_a/datasets_seg/mass_roads/'
base_path = '/home/albert_wei/fdisk_a/datasets_seg/WHU_Building/'


if __name__ == '__main__':
    file_list_path = 'ImageSets/Segmentation/'
    train_files_path = os.path.join(base_path, file_list_path, 'train.txt')
    val_files_path = os.path.join(base_path, file_list_path, 'val.txt')
    test_files_path = os.path.join(base_path, file_list_path, 'test.txt')

    train_img_target_path = os.path.join(base_path, "SegImages/train/")
    val_img_target_path = os.path.join(base_path, "SegImages/val/")
    test_img_target_path = os.path.join(base_path, "SegImages/test/")

    train_mask_target_path = os.path.join(base_path, "gtMasks/train/")
    val_mask_target_path = os.path.join(base_path, "gtMasks/val/")
    test_mask_target_path = os.path.join(base_path, "gtMasks/test/")

    train_file_list = []
    with open(train_files_path) as ft:
        for l in ft:
            train_file_list.append(l.strip())

    val_file_list = []
    with open(val_files_path) as fv:
        for l in fv:
            val_file_list.append(l.strip())

    test_file_list = []
    with open(test_files_path) as ft:
        for l in ft:
            test_file_list.append(l.strip())

    print(len(train_file_list), len(val_file_list), len(test_file_list))

    train_image_full_list = [os.path.join(base_path, 'Images', f) for f in train_file_list]
    train_masks_full_list = [os.path.join(base_path, 'SegmentationClass', f) for f in train_file_list]

    val_image_full_list = [os.path.join(base_path, 'Images', f) for f in val_file_list]
    val_masks_full_list = [os.path.join(base_path, 'SegmentationClass', f) for f in val_file_list]

    test_image_full_list = [os.path.join(base_path, 'Images', f) for f in test_file_list]
    test_masks_full_list = [os.path.join(base_path, 'SegmentationClass', f) for f in test_file_list]

    print(len(train_image_full_list), len(train_masks_full_list),
          len(val_image_full_list), len(val_masks_full_list),
          len(test_image_full_list), len(test_masks_full_list)
          )

    print("======================= Begin Images Conversion =======================")
    images_list = [train_image_full_list, val_image_full_list, test_image_full_list]
    image_target_path_list = [train_img_target_path, val_img_target_path, test_img_target_path]
    for file_list, target_path in zip(images_list, image_target_path_list):
        for f in file_list:
            file_name = f.split('/')[-1].strip() + '.tif'
            target_file_name = os.path.join(target_path, file_name)
            shutil.copy(f.strip() + '.tif', target_file_name)

    print("======================= Begin Masks Conversion =======================")
    mask_list = [train_masks_full_list, val_masks_full_list, test_masks_full_list]
    mask_target_path_list = [train_mask_target_path, val_mask_target_path, test_mask_target_path]
    for file_list, target_path in zip(mask_list, mask_target_path_list):
        for f in file_list:
            file_name = f.split('/')[-1].strip() + '.tif'
            target_file_name = os.path.join(target_path, file_name)
            shutil.copy(f.strip() + '.tif', target_file_name)