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
from ikomia.utils import pyqtutils, qtconversion
from infer_face_inpainting.infer_face_inpainting_process import InferFaceInpaintingParam
# PyQt GUI framework
from PyQt5.QtWidgets import *
from infer_face_inpainting.utils import Autocomplete
import os

# --------------------
# - Class which implements widget associated with the process
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class InferFaceInpaintingWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferFaceInpaintingParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.grid_layout = QGridLayout()

        # Loading model from list
        model_list_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                        "model_list.txt")
        model_list_file = open(model_list_path, "r")

        model_list = model_list_file.read()
        model_list = model_list.split("\n")
        self.combo_model = Autocomplete(model_list, parent=None, i=True, allow_duplicates=False)
        self.label_model = QLabel("Segmentation model")
        self.grid_layout.addWidget(self.combo_model, 1, 0)
        self.grid_layout.addWidget(self.label_model, 0, 0)
        self.combo_model.setCurrentText(self.parameters.model_name_seg)
        model_list_file.close()

        # Cuda
        self.check_mask_only = pyqtutils.append_check(self.grid_layout,
                                                 "Only generate mask",
                                                 self.parameters.mask_only)


        self.spin_dilatation_percent_face = pyqtutils.append_double_spin(
                                                        self.grid_layout,
                                                        "Face mask dilatation (%)",
                                                        self.parameters.dilatation_percent_face,
                                                        min=0, step=0.01, decimals=3
                                                    )

        self.spin_crop_percent_bottom_face = pyqtutils.append_double_spin(
                                                        self.grid_layout,
                                                        "Face mask bottom crop (%)",
                                                        self.parameters.crop_percent_bottom_face,
                                                        min=0, step=0.01, decimals=3
                                                    )


        self.spin_dilatation_percent_hair = pyqtutils.append_double_spin(
                                                        self.grid_layout,
                                                        "Hair mask dilatation (%)",
                                                        self.parameters.dilatation_percent_hair,
                                                        min=0, step=0.01, decimals=3
                                                    )

       # Model name diffusion
        self.combo_model_diff = pyqtutils.append_combo(
            self.grid_layout, "Diffusion model")
        self.combo_model_diff.addItem("SG161222/RealVisXL_V4.0")
        self.combo_model_diff.addItem("stabilityai/stable-diffusion-xl-base-1.0")

        self.combo_model_diff.setCurrentText(self.parameters.model_name_diff)

        # Prompt
        self.edit_prompt = pyqtutils.append_edit(self.grid_layout, "Prompt", self.parameters.prompt)

        # Negative prompt
        self.edit_negative_prompt = pyqtutils.append_edit(
                                                    self.grid_layout,
                                                    "Negative prompt",
                                                    self.parameters.negative_prompt
                                                    )


        # Number of inference steps
        self.spin_number_of_steps = pyqtutils.append_spin(
                                                    self.grid_layout,
                                                    "Number of steps",
                                                    self.parameters.num_inference_steps,
                                                    min=1, step=1
                                                    )
        # Guidance scale
        self.spin_guidance_scale = pyqtutils.append_double_spin(
                                                        self.grid_layout,
                                                        "Guidance scale",
                                                        self.parameters.guidance_scale,
                                                        min=0, step=0.1, decimals=1
                                                    )

        # Stength
        self.spin_strength= pyqtutils.append_double_spin(
                                            self.grid_layout,
                                            "Strength",
                                            self.parameters.strength,
                                            min=0, max=1, step=0.1, decimals=2
                                            )

        # Seed
        self.spin_seed = pyqtutils.append_spin(
                                            self.grid_layout,
                                            "Seed",
                                            self.parameters.seed,
                                            min=-1, step=1
                                            )


        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)

        # Set widget layout
        self.set_layout(layout_ptr)


    def on_apply(self):
        # Apply button clicked slot
        self.parameters.update = True
        self.parameters.mask_only = self.check_mask_only.isChecked()
        self.parameters.model_name_seg = self.combo_model.currentText()
        self.parameters.dilatation_percent_face = self.spin_dilatation_percent_face.value()
        self.parameters.crop_percent_bottom_face = self.spin_crop_percent_bottom_face.value()
        self.parameters.dilatation_percent_hair = self.spin_dilatation_percent_hair.value()
        self.parameters.model_name_diff = self.combo_model_diff.currentText()
        self.parameters.prompt = self.edit_prompt.text()
        self.parameters.num_inference_steps = self.spin_number_of_steps.value()
        self.parameters.guidance_scale = self.spin_guidance_scale.value()
        self.parameters.negative_prompt = self.edit_negative_prompt.text()
        self.parameters.seed = self.spin_seed.value()
        self.parameters.strength = self.spin_strength.value()

        # Send signal to launch the process
        self.emit_apply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class InferFaceInpaintingWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "infer_face_inpainting"

    def create(self, param):
        # Create widget object
        return InferFaceInpaintingWidget(param, None)
