import os
import logging
from detectron2.utils.file_io import PathManager


logger = logging.getLogger(__name__)


def load_mass_roads_semantic(img_dir, gt_dir):
    ret = []
    for image_file, gt_file in _get_mass_road_files(img_dir, gt_dir):
        ret.append(
            {
                "file_name": image_file,
                "sem_seg_file_name": gt_file,
                "height": 512,
                "width": 512,
            }
        )
    assert len(ret), f"No images found in {img_dir}!"
    assert PathManager.isfile(ret[0]["sem_seg_file_name"]), "Please generate gt masks."
    return ret


def _get_mass_road_files(image_dir, gt_dir):
    files = []
    files_list = PathManager.ls(image_dir)
    logger.info(f"{len(files_list)} images found in '{image_dir}'.")
    for f in files_list:
        image_file = os.path.join(image_dir, f)
        gt_file = os.path.join(gt_dir, f)
        files.append((image_file, gt_file))
    assert len(files), "No images found in {}".format(image_dir)
    for f in files[0]:
        assert PathManager.isfile(f), f
    return files