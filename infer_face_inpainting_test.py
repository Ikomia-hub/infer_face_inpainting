import logging
import cv2
from ikomia.core import task
from ikomia.utils.tests import run_for_test


logger = logging.getLogger(__name__)


def test(t, data_dict):
    logger.info("===== Test::infer Hugging Face semantic segmentation=====")
    params = task.get_parameters(t)
    params["model_name"] = "nvidia/segformer-b5-finetuned-ade-640-640"
    img = cv2.imread(data_dict["images"]["detection"]["coco"])[::-1]
    input_img_0 = t.get_input(0)
    input_img_0.set_image(img)
    task.set_parameters(t, params)
    yield run_for_test(t)