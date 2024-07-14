from diffusers import DiffusionPipeline, StableDiffusion3Pipeline, \
    StableDiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionXLPipeline, \
        KDPM2AncestralDiscreteScheduler, AutoencoderKL
import torch
from PIL import Image





class StableDiffusionXL():
    def __init__(self, cache_dir):
        self.pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                                      torch_dtype=torch.float16,
                                                      use_safetensors=True,
                                                      variant="fp16",
                                                      cache_dir=cache_dir)
        self.pipe.to("cuda")
    
    def generate_images(self, prompt, negative_prompt=None):
        images = self.pipe(prompt=prompt,
                           negative_prompt=negative_prompt).images[0]
        resized_images = images.resize((256, 256))
        return resized_images


class StableDiffusion3():
    def __init__(self, cache_dir):
        self.pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers",
                                                      torch_dtype=torch.float16,
                                                      cache_dir=cache_dir)
        self.pipe.to("cuda")
    
    def generate_images(self, prompt, negative_prompt=None):
        images = self.pipe(prompt=prompt,
                           negative_prompt=negative_prompt).images[0]
        resized_images = images.resize((256, 256))
        return resized_images

class StableDiffusion2():
    def __init__(self, cache_dir):
        model_id = "stabilityai/stable-diffusion-2-1-base"

        # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id,
                                                            torch_dtype=torch.float16,
                                                            cache_dir=cache_dir)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to("cuda")
    
    def generate_images(self, prompt, negative_prompt=None):
        images = self.pipe(prompt=prompt,
                           negative_prompt=negative_prompt).images[0]
        resized_images = images.resize((256, 256))
        return resized_images


class StableDiffusion():
    def __init__(self, cache_dir):
        self.pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                      torch_dtype=torch.float16,
                                                      cache_dir=cache_dir)
        self.pipe.to("cuda")
    
    def generate_images(self, prompt, negative_prompt=None):
        images = self.pipe(prompt=prompt,
                           negative_prompt=negative_prompt).images[0]
        resized_images = images.resize((256, 256))
        return resized_images


class BandWManga():
    def __init__(self, cache_dir):
        self.pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                                      cache_dir=cache_dir)
        self.pipe.load_lora_weights("alvdansen/BandW-Manga")
        self.pipe.to("cuda")
    
    def generate_images(self, prompt, negative_prompt=None):
        images = self.pipe(prompt=prompt,
                           negative_prompt=negative_prompt).images[0]
        resized_images = images.resize((256, 256))
        return resized_images



class Mobius():
    def __init__(self, cache_dir):
        self.vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", 
            torch_dtype=torch.float16,
            cache_dir=cache_dir
        )

        # Configure the pipeline
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "Corcelio/mobius", 
            vae=self.vae,
            torch_dtype=torch.float16,
            cache_dir=cache_dir
        )
        self.pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to("cuda")
    def generate_images(self, prompt, negative_prompt=None):
        image = self.pipe(
                    prompt, 
                    width=256,
                    height=256,
                    guidance_scale=7,
                    num_inference_steps=50,
                    clip_skip=3,
                    negative_prompt=negative_prompt
                ).images[0]
        return image


class Fluently():
    def __init__(self, cache_dir):

        self.pipe = DiffusionPipeline.from_pretrained("fluently/Fluently-XL-Final",
                                                    torch_dtype=torch.float16,
                                                    cache_dir=cache_dir)
        self.pipe.to("cuda")
    
    def generate_images(self, prompt, negative_prompt=None): 
        images = self.pipe(prompt=prompt,
                           negative_prompt=negative_prompt).images[0    ]
        resized_images = images.resize((256, 256))
        return resized_images


class Visionix():
    def __init__(self, cache_dir):

        self.pipe = DiffusionPipeline.from_pretrained("ehristoforu/Visionix-alpha",
                                                  torch_dtype=torch.float16,
                                                  cache_dir=cache_dir)  
        self.pipe.to("cuda")
    
    def generate_images(self, prompt, negative_prompt=None):
        images = self.pipe(prompt=prompt,
                           negative_prompt=negative_prompt).images[0]
        resized_images = images.resize((256, 256))
        return resized_images


class DeepFloyd():
    def __init__(self, cache_dir):
        self.pipe = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0",
                                                  torch_dtype=torch.float16,
                                                  cache_dir=cache_dir)
        self.pipe.to("cuda")
    
    def generate_images(self, prompt, negative_prompt=None):
        images = self.pipe(prompt=prompt,
                           negative_prompt=negative_prompt).images[0]
        resized_images = images.resize((256, 256))
        return resized_images

class MVDream():
    def __init__(self, cache_dir):
        self.pipe = DiffusionPipeline.from_pretrained("dylanebert/mvdream",
                                                      custom_pipeline="dylanebert/multi-view-diffusion",
                                                      trust_remote_code=True,
                                                      cache_dir=cache_dir)

        self.pipe.to("cuda")
    
    def generate_images(self, prompt, negative_prompt=None, combined=False):
        images = self.pipe(prompt=prompt,
                           guidance_scale=5,
                           num_inference_steps=30,
                           elevation=0,
                           negative_prompt=negative_prompt)
        if combined:
            images = self.create_image_grid(images)

        else:
            images = [Image.fromarray((img * 255).astype("uint8")) for img in images]
        
        return images


    def create_image_grid(self, images):
        images = [Image.fromarray((img * 255).astype("uint8")) for img in images]

        width, height = images[0].size
        grid_img = Image.new("RGB", (2 * width, 2 * height))

        grid_img.paste(images[0], (0, 0))
        grid_img.paste(images[1], (width, 0))
        grid_img.paste(images[2], (0, height))
        grid_img.paste(images[3], (width, height))

        return grid_img


MODEL_DICT = {
    "stable_diffusion": StableDiffusion,
    "stable_diffusion_2": StableDiffusion2,
    "stable_diffusion_3": StableDiffusion3,
    "stable_diffusion_xl": StableDiffusionXL,
    "bandw_manga": BandWManga,
    "mobius": Mobius,
    "fluently": Fluently,
    "visionix": Visionix,
    "deep-floyd-guidance": DeepFloyd,
    "mvdream-multiview-diffusion-guidance": MVDream
}

class DiffusionModel():
    def __init__(self, model_name, cache_dir):
        self.model = MODEL_DICT[model_name](cache_dir)
        self.model_name = model_name
        self.model.pipe.enable_model_cpu_offload()
    
    def generate_images(self, prompt, negative_prompt=None, combined=False):
        if self.model_name == "mvdream":
            return self.model.generate_images(prompt, negative_prompt, combined)
        return self.model.generate_images(prompt, negative_prompt)



if __name__ == "__main__":
    prompt = "a wing of a dragon"
    
    cache_dir = "/homes/55/runjia/scratch/diffusion_model_weights"    
    model = MVDream(cache_dir=cache_dir)
    images = model.generate_images(prompt)
    images.save("sunset.png")