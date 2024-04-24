<div align="center">
  <img src="https://raw.githubusercontent.com/Ikomia-hub/infer_face_inpainting/main/icons/icon.png" alt="Algorithm icon">
  <h1 align="center">infer_face_inpainting</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/infer_face_inpainting">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/infer_face_inpainting">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/infer_face_inpainting/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/infer_face_inpainting.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>


Replace faces using diffusion inpainting. This algorithm uses Segformer for segmentation and RealVisXL V4.0 for inpainting.

This algorithm requires at least 13GB GPU VRAM to run. 

## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).
![illustration](https://raw.githubusercontent.com/Ikomia-hub/infer_face_inpainting/main/icons/inference_steps.jpg)

```sh
pip install ikomia
```

#### 2. Create your workflow

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_face_inpainting", auto_connect=True)

# Run on your image  
wf.run_on(url="https://images.pexels.com/photos/774909/pexels-photo-774909.jpeg?cs=srgb&dl=pexels-andrea-piacquadio-774909.jpg&fm=jpg&w=640&h=960")

# Inpect your result
display(algo.get_image_with_mask())
display(algo.get_output(2).get_image())
display(algo.get_output(3).get_image())
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.
- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).
- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters

- **model_name_seg** (str) - default 'matei-dorian/segformer-b5-finetuned-human-parsing': Name of the segmentation model. Other model available:
    - mattmdjaga/segformer_b2_clothes
- **dilatation_percent_face** (float) - default '0.001': Dilation percentage of the face mask.
- **dilatation_percent_hair** (float) - default '0.03': Dilation percentage of the hair mask.
- **crop_percent_bottom_face** (float) - default '0.05': The mask is generated accross the hair, face and neck. In case you don't want the neck to be segmented you crop this part by increasing the percentage.
- **nask_only** (bool) - default 'False': If True, only the segmentation step will be done. This allows for quick segmentation mask adjustement before doing the inpainting. 
- **model_name_diff** (str) - default 'SG161222/RealVisXL_V4.0': Name of the stable diffusion model.

*Note: for faster inference with less than 10 steps, use 'SG161222/RealVisXL_V4.0_Lightning' with guidance_scale=1-2*

- **prompt** (str): Text prompt to guide the image generation.
- **negative_prompt** (str, *optional*): The prompt not to guide the image generation. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
- **num_inference_steps** (int) - default '50': Number of denoising steps (minimum: 1; maximum: 500). For 'RealVisXL_V4.0_Lightning' we recommend using between 5 and 10 steps.
- **guidance_scale** (float) - default '7.5': Scale for classifier-free guidance (minimum: 1; maximum: 20). For 'RealVisXL_V4.0_Lightning' we recommend using 0 and 2.
- **strength** (int) - default '0.75':  Conceptually, indicates how much to transform the reference image. Must be between 0 and 1. image will be used as a starting point, adding more noise to it the larger the strength. The number of denoising steps depends on the amount of noise initially added. When strength is 1, added noise will be maximum and the denoising process will run for the full number of iterations specified in num_inference_steps. A value of 1, therefore, essentially ignores image.
- **seed** (int) - default '-1': Seed value. '-1' generates a random number between 0 and 191965535.


**Parameters** should be in **strings format**  when added to the dictionary.

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_face_inpainting", auto_connect=True)

algo.set_parameters({
        "model_name_seg":"matei-dorian/segformer-b5-finetuned-human-parsing",
        "dilatation_percent_face":"0.0",
        "dilatation_percent_hair":"0.03",
        "crop_percent_bottom_face":"0.22",
        "mask_only":"False",
        "model_name_diff": "SG161222/RealVisXL_V4.0",
        "prompt":"high quality, portrait photo, blond hair, detailed face, skin pores, no makeup",
        "negative_prompt":"(face asymmetry, eyes asymmetry, deformed eyes, open mouth)",
        "guidance_scale":"7.5",
        "num_inference_steps":"50",
        "strength":"0.75",
        "seed":"-1",
})


# Run on your image  
wf.run_on(url="https://images.pexels.com/photos/774909/pexels-photo-774909.jpeg?cs=srgb&dl=pexels-andrea-piacquadio-774909.jpg&fm=jpg&w=640&h=960")

# Inpect your result
display(algo.get_image_with_mask())
display(algo.get_output(2).get_image())
display(algo.get_output(3).get_image())
```

## :mag: Explore algorithm outputs

Every algorithm produces specific outputs, yet they can be explored them the same way using the Ikomia API. For a more in-depth understanding of managing algorithm outputs, please refer to the [documentation](https://ikomia-dev.github.io/python-api-documentation/advanced_guide/IO_management.html).

```python
import ikomia
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_face_inpainting", auto_connect=True)

# Run on your image  
wf.run_on(url="https://images.pexels.com/photos/774909/pexels-photo-774909.jpeg?cs=srgb&dl=pexels-andrea-piacquadio-774909.jpg&fm=jpg&w=640&h=960")

# Iterate over outputs
for output in algo.get_outputs():
    # Print information
    print(output)
    # Export it to JSON
    output.to_json()
```


## :page_with_curl: Citation

- Segformer
    - [Documentation](https://arxiv.org/abs/2105.15203)
    - [Code source](https://github.com/NVlabs/SegFormer)   

```bibtex
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```

- RealVisXL v4.0
    - [Documentation](https://civitai.com/models/139562/realvisxl-v40)
