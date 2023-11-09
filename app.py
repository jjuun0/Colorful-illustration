import gradio as gr
import numpy as np
import random
import torch

from diffusers import StableDiffusionXLPipeline, AutoencoderKL, EulerAncestralDiscreteScheduler
from utils import randomize_seed_fn

MAX_SEED = np.iinfo(np.int32).max

def model_load():
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", vae=vae, torch_dtype=torch.float16
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    
    # load lora weight
    pipe.load_lora_weights("jjuun/vivid_color_style")

    return pipe.to('cuda')


def sdxl_process(seed, prompt, additional_prompt, negative_prompt, num_steps, guidance_scale):
    pipe = model_load()
    generator = torch.Generator("cuda")
    generator.manual_seed(int(seed))
    
    special_prompt = 'jjj, scratch art style'
    prompt = f'{special_prompt}, {prompt}, with a black background'
    output = pipe(prompt, additional_prompt, negative_prompt=negative_prompt, num_inference_steps=num_steps, guidance_scale=guidance_scale,
                  generator=generator).images[0]
    
    return output


title = "🌈 Colorful illustration" 
description_en = "🚀 How to use: please make sure to include 'a colorful' in prompt and click Run button!"


def create_demo():

    with gr.Blocks() as demo:
        gr.Markdown(f"<h1 style='text-align: center;'>{title}</h1>")
        gr.Markdown(f"<h3 style='text-align: center'>{description_en}</h3>")
        gr.Markdown(f"<a href='https://github.com/jjuun0'><img src='https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=GitHub&logoColor=white'/></a>")

        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt")
                run_button = gr.Button("Run")
                with gr.Accordion("Advanced options", open=False):
                    
                    num_steps = gr.Slider(label="Number of steps", minimum=1, maximum=100, value=20, step=1)
                    guidance_scale = gr.Slider(label="Guidance scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                    seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                    a_prompt = gr.Textbox(label="Additional prompt", value="")
                    n_prompt = gr.Textbox(
                        label="Negative prompt",
                        value="",
                    )
            with gr.Column():
                result = gr.Image(label="Output")
                result_seed = gr.Textbox(label="Used seed")
        
        gr.Examples(
            
            examples= [["a colorful fox", "20", "9", "0", "", "", "examples/fox.png"], 
                       ["a colorful messi", "20", "9", "191251724", "", "", "examples/messi.png"],
                       ["a colorful pyramid", "20", "9", "0", "", "", "examples/pyramid.png"],
                       ["a colorful octopus playing violin", "20", "9", "0", "", "", "examples/octopus.png"]],
            inputs = [prompt, num_steps, guidance_scale, seed, a_prompt, n_prompt, result]
        )

        inputs = [
            seed,
            prompt,
            a_prompt,
            n_prompt,
            num_steps,
            guidance_scale,
        ]

        run_button.click(
            fn=randomize_seed_fn,
            inputs=[seed, randomize_seed],
            outputs=result_seed,
            queue=False,
            api_name=False,
        ).then(
            fn=sdxl_process,
            inputs=inputs,
            outputs=result,
            api_name=False,
        )
    

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.queue().launch()
