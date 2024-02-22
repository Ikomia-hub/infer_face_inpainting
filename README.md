<div align="center">
  <img src="https://raw.githubusercontent.com/Ikomia-hub/infer_hf_semantic_seg/main/icons/icon.png" alt="Algorithm icon">
  <h1 align="center">infer_hf_semantic_seg</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/infer_hf_semantic_seg">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/infer_hf_semantic_seg">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/infer_hf_semantic_seg/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/infer_hf_semantic_seg.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>



## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

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
        "prompt":"high quality, portrait photo, detailed face, skin pores, no makeup",
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
