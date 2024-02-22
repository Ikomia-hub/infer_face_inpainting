# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from ikomia import core, dataprocess
import copy
# Your imports below
from transformers import AutoFeatureExtractor, AutoModelForSemanticSegmentation
from ikomia.utils import strtobool
import numpy as np
import torch
import os
from torch import nn
import cv2
from diffusers import AutoPipelineForInpainting, DPMSolverMultistepScheduler
import random
from PIL import Image


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferFaceInpaintingParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.cuda = torch.cuda.is_available()
        self.model_name_seg = "matei-dorian/segformer-b5-finetuned-human-parsing"
        self.dilatation_percent_face = 0.0
        self.dilatation_percent_hair = 0.03
        self.crop_percent_bottom_face = 0.05
        self.mask_only = False
        self.model_name_diff = "SG161222/RealVisXL_V4.0"
        self.prompt = "high quality, portrait photo, detailed face, skin pores, no makeup"
        self.negative_prompt = '(face asymmetry, eyes asymmetry, deformed eyes, open mouth)'
        self.guidance_scale = 7.5
        self.num_inference_steps = 50
        self.strength = 0.75
        self.seed = -1
        self.update = False

    def set_values(self, params):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.cuda = strtobool(params["cuda"])
        self.model_name_seg = str(params["model_name_seg"])
        self.dilatation_percent_face = float(params["dilatation_percent_face"])
        self.dilatation_percent_hair = float(params["dilatation_percent_hair"])
        self.crop_percent_bottom_face = float(params["crop_percent_bottom_face"])
        self.mask_only = strtobool(params["mask_only"])
        self.model_name_diff = str(params["model_name_diff"])
        self.prompt = str(params["prompt"])
        self.negative_prompt = str(params["negative_prompt"])
        self.guidance_scale = float(params["guidance_scale"])
        self.num_inference_steps = int(params["num_inference_steps"])
        self.strength = float(params["strength"])
        self.seed = int(params["seed"])
        self.update = True


    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        params = {}
        params["cuda"] = str(self.cuda)
        params["model_name_seg"] = str(self.model_name_seg)
        params["dilatation_percent_face"] = str(self.dilatation_percent_face)
        params["dilatation_percent_hair"] = str(self.dilatation_percent_hair)
        params["crop_percent_bottom_face"] = str(self.crop_percent_bottom_face)
        params["mask_only"] = str(self.mask_only)
        params["model_name_diff"] = str(self.model_name_diff)
        params["prompt"] = str(self.prompt)
        params["negative_prompt"] = str(self.negative_prompt)
        params["guidance_scale"] = str(self.guidance_scale)
        params["num_inference_steps"] = str(self.num_inference_steps)
        params["seed"] = str(self.seed)
        params["strength"] = str(self.strength)

        return params


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferFaceInpainting(dataprocess.CSemanticSegmentationTask):

    def __init__(self, name, param):
        dataprocess.CSemanticSegmentationTask.__init__(self, name)
        self.add_output(dataprocess.CImageIO())
        
        if param is None:
            self.set_param_object(InferFaceInpaintingParam())
        else:
            self.set_param_object(copy.deepcopy(param))

        # Detect if we have a GPU available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_seg = None
        self.pipe = None
        self.feature_extractor = None
        self.classes = None
        self.update = False
        self.model_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "weights")
        self.generator = None

    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def infer_seg(self, image):
        param = self.get_param_object()

        # Image pre-pocessing (image transformation and conversion to PyTorch tensor)
        pixel_val = self.feature_extractor(image, return_tensors="pt", resample=0).pixel_values
        if torch.cuda.is_available():
            if param.cuda is True:
                pixel_val = pixel_val.cuda()

        # Prediction
        outputs = self.model_seg(pixel_val)
        logits = outputs.logits

        upsampled_logits = nn.functional.interpolate(
            logits,
            size = image.shape[:2],
            mode="bilinear",
            align_corners=False,
        )

        pred_seg = upsampled_logits.argmax(dim=1)[0]

        # dstImage
        dst_image = pred_seg.cpu().detach().numpy().astype(dtype=np.uint8)

        # Get mask
        self.set_names(self.classes)
        self.set_mask(dst_image)

        mask = dst_image

        # Create binary masks for classes 2 (Hair) and 11 (Face)
        class_2_mask = np.isin(mask, [2]).astype(int)
        class_11_mask = np.isin(mask, [11]).astype(int)

        # Convert to uint8
        class_2_mask_uint8 = (class_2_mask * 255).astype(np.uint8)
        class_11_mask_uint8 = (class_11_mask * 255).astype(np.uint8)

        # Find the vertical extent of the face mask
        rows_where_face_exists = np.any(class_11_mask, axis=1)
        face_start, face_end = np.where(rows_where_face_exists)[0][[0, -1]]

        # Calculate the height to crop based on the face mask size
        face_height = face_end - face_start
        crop_height = int(face_height * param.crop_percent_bottom_face)
        if crop_height > 0:
            class_11_mask_uint8[face_end - crop_height + 1:face_end + 1] = 0  # Crop from the bottom of the face mask

        # Dilate both masks (dilation is minimal due to dilatation_percent being 0)
        kernel_size_11 = max(1, int(min(class_11_mask_uint8.shape[:2]) * param.dilatation_percent_face))
        kernel_11 = np.ones((kernel_size_11, kernel_size_11), np.uint8)
        dilated_class_11_mask = cv2.dilate(class_11_mask_uint8, kernel_11, iterations=1)

        kernel_size_2 = max(1, int(min(class_2_mask_uint8.shape[:2]) * param.dilatation_percent_hair))
        kernel_2 = np.ones((kernel_size_2, kernel_size_2), np.uint8)
        dilated_class_2_mask = cv2.dilate(class_2_mask_uint8, kernel_2, iterations=1)

        # Merge masks with priority to face mask in overlaps
        overlap = (dilated_class_2_mask > 0) & (dilated_class_11_mask > 0)
        merged_mask = dilated_class_2_mask.copy()
        merged_mask[overlap] = dilated_class_11_mask[overlap]
        merged_mask[dilated_class_11_mask > 0] = 255

        # Apply merged mask to the image
        boolean_mask = merged_mask == 255
        image_mask_overlay = image.copy()
        for c in range(3):  # Apply to all three channels
            image_mask_overlay[boolean_mask, c] = 255  # Set to black for demonstration

        return image_mask_overlay, merged_mask
    

    def infer_diff(self, image, image_inpaint):
        param = self.get_param_object()

        if param.update or self.pipe is None:
            self.pipe = AutoPipelineForInpainting.from_pretrained(
                                            "SG161222/RealVisXL_V4.0",
                                            torch_dtype=torch.float16,
                                            variant="fp16",
                                            cache_dir= self.model_folder).to(self.device)
            self.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config, use_karras_sigmas=True)

        # Convert to PIL
        image_input = Image.fromarray(image)
        mask_image = Image.fromarray(image_inpaint)

        # Edit size to be a multiple of 8
        width, height = image_input.size
        width = width // 8 * 8
        height = height // 8 * 8

        # Set Seed
        if param.seed == -1:
            self.seed = random.randint(0, 191965535)
        else:
            self.seed = param.seed
        self.generator = torch.Generator(self.device).manual_seed(self.seed)

        # Inpainting
        with torch.no_grad():
            inpaint_img = self.pipe(
                prompt=param.prompt,
                image=image_input,
                negative_prompt=param.negative_prompt,
                mask_image=mask_image,
                guidance_scale=param.guidance_scale,
                num_inference_steps=param.num_inference_steps,
                strength=param.strength,  # make sure to use `strength` below 1.0
                height=height,
                width=width,
                generator=self.generator,
            ).images[0]

        return inpaint_img


    def run(self):
        # Core function of your process
        # Call begin_task_run for initialization
        self.begin_task_run()

        image_in = self.get_input(0)

        # Get image from input/output (numpy array):
        image = image_in.get_image()

        param = self.get_param_object()

        if param.update or self.model_seg is None:
            param = self.get_param_object()
            # Feature extractor selection
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                param.model_name_seg,
                cache_dir=self.model_folder,
            )
            self.model_seg = AutoModelForSemanticSegmentation.from_pretrained(
                param.model_name_seg,
                cache_dir=self.model_folder,
                )

            self.device = torch.device("cuda") if param.cuda and torch.cuda.is_available() else torch.device("cpu")
            self.model_seg.to(self.device)

            print("Will run on {}".format(self.device.type))

            # Get label name
            self.classes = list(self.model_seg.config.id2label.values())

            param.update = False

        # Inference
        image_mask_face_hair, dst_image = self.infer_seg(image)

        output_bin_img = self.get_output(2)
        output_bin_img.set_image(image_mask_face_hair)

        if not param.mask_only:
            self.add_output(dataprocess.CImageIO())
            output_img_inpaint = self.get_output(3)
            image_inpaint = self.infer_diff(image, dst_image)
            image_inpaint_np = np.array(image_inpaint)
            output_img_inpaint.set_image(image_inpaint_np)


        # Step progress bar:
        self.emit_step_progress()

        # Call end_task_run to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferFaceInpaintingFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_face_inpainting"
        self.info.short_description = "Face inpainting"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Inpaint"
        self.info.version = "1.1.1"
        self.info.icon_path = "icons/icon.png"
        self.info.authors = ""
        self.info.article = ""
        self.info.journal = ""
        self.info.license = "Apache License Version 2.0"
        # URL of documentation
        self.info.documentation_link = ""
        # Code source repository
        self.info.repository = ""
        self.info.original_repository = ""
        # Keywords used for search
        self.info.keywords = "semantic, segmentation, inference, transformer,"\
                            "Hugging Face, Diffusion,"
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "INPAINTING"

    def create(self, param=None):
        # Create process object
        return InferFaceInpainting(self.info.name, param)
