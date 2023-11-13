
# Colorful illustration
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/jjuun/Colorful-illustration)
<p align="center">
	<img src = "https://github.com/jjuun0/Colorful-illustration/assets/66052461/7a4f483c-075d-4414-b01e-ebd4267f544d", height="300px", width="500px">
</p>

- This is a project that allows you to create colorful illustration images using text prompts..
  - ex: "a colorful baby panda"
- Finetuning was done using lora based on SDXL 1.0.

## How to use
- Environment: testing
  ```
  diffusers==0.21.4
  transformers==4.35.0
  opencv-python
  torch==1.12.0+cu113
  accelerate==0.24.1
  ```
- RUN
	- `python app.py`
   
 	 	![app.py](https://github.com/jjuun0/Colorful-illustration/assets/66052461/18fd66b4-7fd0-49d7-8367-8e84a0e65207)

- Prompt 
	- Please include **"a colorful"** to write the simple prompt you want to create.  
- (option) Advanced option  
	- Number of steps: how many steps are used in inference. 
	- Guidance scale: how much weight the prompt gives to the model.
	- Seed: adjust the seed value you want. 
	- Randomize seed: check when you want the seed of a random value.
	- Additional prompt: write additionally in Prompt. (ex: high quality)
	- Negative prompt: write it so that objects you do not want are not created.


### certificate
![image](https://github.com/jjuun0/Colorful-illustration/assets/66052461/7c19cabb-5fcd-492a-9d55-dd4787876512)
