import os
from dataclasses import dataclass, field

import threestudio
import torch
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.misc import cleanup, get_device
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *
import numpy as np
from PIL import Image
import torch.nn.functional as F
from threestudio.utils.base import update_end_if_possible
import cv2
from .mllm_optimizer import run_model_optimization
from .cross_attention import get_attn_maps_sd3, DenseCRF, crf_refine, attn_map_postprocess, set_forward_sd3, register_cross_attention_hook, set_forward_mvdream, animal_part_extractor, prompt2tokens
from diffusers import DiffusionPipeline
from collections import defaultdict
from transformers import AutoTokenizer
import pickle




torch.set_float32_matmul_precision('medium')
torch.random.manual_seed(0)


def smooth_loss(inputs):
    if inputs.shape[0] > 3:
        inputs = inputs.permute(2, 0, 1)
    L1 = torch.mean(torch.abs((inputs[:,:,:-1] - inputs[:,:,1:])))
    L2 = torch.mean(torch.abs((inputs[:,:-1,:] - inputs[:,1:,:])))
    L3 = torch.mean(torch.abs((inputs[:,:-1,:-1] - inputs[:,1:,1:])))
    L4 = torch.mean(torch.abs((inputs[:,1:,:-1] - inputs[:,:-1,1:])))
    return (L1 + L2 + L3 + L4) / 4


@threestudio.register("partdream-system")
class PartDreamSystem(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        prompt_save_path: str = ""
        cache_dir: str = None
        save_dir: str = ""
        api_key: str = ""


        # global guidance model names
        use_global_attn: bool = False
        global_model_name: str = "stable-diffusion-3-medium-diffusers"
        attention_guidance_start_step: int = 1000
        attention_guidance_timestep_start:int = 850
        attention_guidance_timestep_end:int = 400
        attention_guidance_free_style_timestep_start:int = 500
        record_attention_interval: int = 10

        use_crf: bool = False

        cross_attention_scale: float = 1.0
        self_attention_scale: float = 1.0

        visualize: bool = False
        visualize_save_dir: str = ""

        attention_system: defaultdict = field(default_factory=defaultdict)


    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()

        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        set_forward_mvdream(self.guidance.model)

        self.attention_guidance_prompt = self.cfg.prompt_processor.prompt
        self.attention_guidance_negative_prompt = self.cfg.prompt_processor.negative_prompt

        self.part_prompts = animal_part_extractor(self.cfg.prompt_processor.prompt, api_key=self.cfg.api_key)
        self.index_by_part = {}

        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.prompt_utils = self.prompt_processor()
        self.guidance_tokenizer = AutoTokenizer.from_pretrained(self.cfg.prompt_processor.pretrained_model_name_or_path,
                                                                subfolder="tokenizer")
        self.part_token_index_list = []
        for part_prompt in self.part_prompts:
            token_index_list = self.get_token_index(self.guidance_tokenizer, self.cfg.prompt_processor.prompt, part_prompt)
            self.part_token_index_list.append(token_index_list)



        file_name = self.attention_guidance_prompt.replace(" ", "_")+".pth"
        save_path = os.path.join("custom/threestudio-mvdream/system/cross_attention/cache", file_name)
        # initialize the global guidance model
        if not os.path.exists(save_path):
            self.global_model = DiffusionPipeline.from_pretrained(self.cfg.global_model_name,
                                                            use_safetensors=True,
                                                            torch_dtype=torch.float16,
                                                            cache_dir=self.cfg.cache_dir)
            set_forward_sd3(self.global_model.transformer)
            register_cross_attention_hook(self.global_model.transformer)
            self.global_model = self.global_model.to("cuda")
            self.global_model.enable_model_cpu_offload()
        else:
            self.global_model = None
        self.postprocessor = DenseCRF(
            iter_max=10,
            pos_xy_std=1,
            pos_w=3,
            bi_xy_std=67,
            bi_rgb_std=3,
            bi_w=4,
        )
        self.attn_map_info = defaultdict(list)


        self.attn_geometry = threestudio.find(self.cfg.attention_system.geometry_type)(self.cfg.attention_system.geometry)

        self.attn_material = threestudio.find(self.cfg.attention_system.material_type)(self.cfg.attention_system.material)
        self.attn_background = threestudio.find(self.cfg.attention_system.background_type)(
            self.cfg.attention_system.background
        )
        self.attn_renderer = threestudio.find(self.cfg.attention_systemrenderer_type)(
            self.cfg.attention_system.renderer,
            geometry=self.attn_geometry,
            material=self.attn_material,
            background=self.attn_background,
        )


    def get_token_index(self,
                        tokenizer,
                        prompt:str,
                        sub_prompt:str):
        """
        Get the token index of the sub_prompt in the prompt
        args:
        tokenizer: the tokenizer
        prompt: str, the prompt
        sub_prompt: str, the sub_prompt

        return:
        token_index_list: List[int], the list of token index
        """
        tokens = prompt2tokens(tokenizer, prompt)
        sub_tokens = prompt2tokens(tokenizer, sub_prompt)
        bos_token = tokenizer.bos_token
        eos_token = tokenizer.eos_token
        pad_token = tokenizer.pad_token
        token_index_list = []
        for idx, token in enumerate(tokens):
            if token == eos_token:
                break
            if token in [bos_token, pad_token]:
                continue
            if token in sub_tokens:
                token_index_list.append(idx)
        return token_index_list
    
                        

    def get_attn_maps_sd3(self,
                          images=None,
                          use_crf:bool = False,
                          device="cuda"):
        """
        Get the attention maps from the global guidance model
        args:

        images: torch.Tensor, shape (B, H, W, 3)
        use_crf: bool, whether to use crf to refine the attention maps

        return:
        attn_map_by_tokens: Dict[str, torch.Tensor], shape (B, num_tokens, H, W)
        """
        
        with torch.no_grad():
         # if the global model is half precision, convert the image to half precision
            view_suffix = ["back view", "side view", "front view", "side view"] * int(images.shape[0] / 4)
            attn_map_by_tokens = defaultdict(list)
            file_name = self.attention_guidance_prompt.replace(" ", "_")+".pth"
            save_path = os.path.join("custom/threestudio-mvdream/system/cross_attention/cache", file_name)
            if os.path.exists(save_path):
                attn_map_by_tokens = torch.load(save_path)
                return attn_map_by_tokens
            if images is not None:
                images = images.to(self.global_model.dtype)
            else:
                images = [None] * 4
            for idx, image in enumerate(images):
                # convert image to PIL image
                if image is not None:
                    image = Image.fromarray((image.cpu().detach().numpy()*255).astype(np.uint8))
                    if self.cfg.visualize:
                        image.save(os.path.join(self.cfg.visualize_save_dir, f"image_{idx}.png"))

                suffix = view_suffix[idx]
                prompt = self.attention_guidance_prompt + ", " + suffix
                output = get_attn_maps_sd3(model=self.global_model,
                                prompt=prompt,
                                negative_prompt=None,
                                only_animal_names=True,
                                animal_names=self.part_prompts,
                                image=image,
                                timestep_start=self.cfg.attention_guidance_timestep_start,
                                timestep_end=self.cfg.attention_guidance_timestep_end,
                                free_style_timestep_start=self.cfg.attention_guidance_free_style_timestep_start,
                                save_by_timestep=True,
                                save_dir=self.cfg.visualize_save_dir,
                                api_key=self.cfg.api_key,
                                normalize=True,)
            
                attn_map_by_token = output['attn_map_by_token']
                if use_crf:
                    diffused_image = output['diffused_image']

                    probmaps, _ = crf_refine(diffused_image,
                                            attn_map_by_token,
                                            self.postprocessor,
                                            save_dir=self.cfg.visualize_save_dir)


                    attn_map_by_token = attn_map_postprocess(probmaps,
                                            attn_map_by_token,
                                            amplification_factor=2,
                                            save_dir=self.cfg.visualize_save_dir,)
                
                for key, value in attn_map_by_token.items():
                    value = torch.tensor(value)
                    scale = torch.sum(torch.ones_like(value)) / torch.sum(value)
                    value = value * scale
                    self.attn_map_info[key].append(value)

            del attn_map_by_token



    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return self.renderer(**batch)

    def training_step(self, batch, batch_idx):
        out = self(batch)
        guidance_out = self.guidance(out["comp_rgb"],
                                     self.prompt_utils,
                                     token_index=self.part_token_index_list,
                                     cross_attention_scale=self.cfg.cross_attention_scale,
                                     self_attention_scale=self.cfg.self_attention_scale,
                                     **batch)
        
        # get the global attention map
        if self.cfg.use_global_attn and batch_idx > self.cfg.attention_guidance_start_step and \
            batch_idx < (self.cfg.attention_guidance_start_step + self.cfg.record_attention_interval):


            self.get_attn_maps_sd3(images=out["comp_rgb"],
                                    use_crf=self.cfg.use_crf)
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    if isinstance(value, torch.Tensor):
                        for cam2world in value:
                            self.attn_map_info[key].append(cam2world.detach().cpu())
                   
        
        if batch_idx == (self.cfg.attention_guidance_start_step + self.cfg.record_attention_interval) and self.cfg.use_global_attn:
            for key, value in self.attn_map_info.items():
                self.attn_map_info[key] = torch.stack(value)
            
            attn_file_name = self.attention_guidance_prompt.replace(" ", "_")+"attn.pth"
            attn_save_path = os.path.join("custom/threestudio-mvdream/system/cross_attention/cache", attn_file_name)
            torch.save(self.attn_map_info, attn_save_path)



            stop = 1



        loss = 0.0
        

        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])


        if self.C(self.cfg.loss.lambda_orient) > 0:
            if "normal" not in out:
                raise ValueError(
                    "Normal is required for orientation loss, no normal is found in the output."
                )
            loss_orient = (
                out["weights"].detach()
                * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
            ).sum() / (out["opacity"] > 0).sum()
            self.log("train/loss_orient", loss_orient)
            loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

        if self.C(self.cfg.loss.lambda_sparsity) > 0:
            loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
            self.log("train/loss_sparsity", loss_sparsity)
            loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        if self.C(self.cfg.loss.lambda_opaque) > 0:
            opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
            loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
            self.log("train/loss_opaque", loss_opaque)
            loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

        # z variance loss proposed in HiFA: http://arxiv.org/abs/2305.18766
        # helps reduce floaters and produce solid geometry
        if self.C(self.cfg.loss.lambda_z_variance) > 0:
            loss_z_variance = out["z_variance"][out["opacity"] > 0.5].mean()
            self.log("train/loss_z_variance", loss_z_variance)
            loss += loss_z_variance * self.C(self.cfg.loss.lambda_z_variance)

        if (
            hasattr(self.cfg.loss, "lambda_eikonal")
            and self.C(self.cfg.loss.lambda_eikonal) > 0
        ):
            loss_eikonal = (
                (torch.linalg.norm(out["sdf_grad"], ord=2, dim=-1) - 1.0) ** 2
            ).mean()
            self.log("train/loss_eikonal", loss_eikonal)
            loss += loss_eikonal * self.C(self.cfg.loss.lambda_eikonal)

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="validation_step",
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="test_step",
            step=self.true_global_step,
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )



@threestudio.register("partdream-gaussian-system")
class PartDreamGaussianSystem(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        visualize_samples: bool = False
        prompt: str = ""
        prompt_save_path: str = ""
        iteration_num: int = 0
        cache_dir: str = None
        save_dir: str = ""
        api_key: str = ""


        # global guidance model names
        use_global_attn: bool = False
        global_model_name: str = "stable-diffusion-3-medium-diffusers"
        attention_guidance_start_step: int = 4000
        attention_guidance_interval: int = 100
        attention_guidance_timestep_start:int = 850
        attention_guidance_timestep_end:int = 400
        attention_guidance_free_style_timestep_start:int = 500
        use_crf: bool = False

        cross_attention_scale: float = 1.0
        self_attention_scale: float = 1.0


        mllm_optimize_prompt: bool = False


        visualize: bool = False
        visualize_save_dir: str = ""


    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()

        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        set_forward_mvdream(self.guidance.model)
        part_model_names = self.cfg.guidance_type
        global_model_names = self.cfg.guidance_type + "," + self.cfg.global_model_name
        # use mllm to optimize the prompt
        if self.cfg.mllm_optimize_prompt:
            print("Optimizing the prompt using MLLM")
            (optimzied_global_prompts,
            optimized_negative_global_prompts,
            optimized_part_prompts,
            optimized_negative_part_prompts) = run_model_optimization(original_prompt=self.cfg.prompt,
                                                    original_negative_prompt="",
                                                    part_model_names=part_model_names,
                                                    global_model_names=global_model_names,
                                                    api_key=self.cfg.api_key,
                                                    iteration_num=self.cfg.iteration_num,
                                                    cache_dir=self.cfg.cache_dir,
                                                    save_dir=self.cfg.prompt_save_path,)
        
            # update the prompt the optimized prompt
            self.cfg.prompt_processor.prompt = optimzied_global_prompts[self.cfg.guidance_type]['global']
            self.cfg.prompt_processor.negative_prompt = optimized_negative_global_prompts[self.cfg.guidance_type]['global']
            self.attention_guidance_prompt = optimzied_global_prompts[self.cfg.global_model_name]['global']
            self.attention_guidance_negative_prompt = optimized_negative_global_prompts[self.cfg.global_model_name]['global']
            # TODO: update the part expert prompt

        else:
            self.cfg.prompt_processor.prompt = self.cfg.prompt
            self.attention_guidance_prompt = self.cfg.prompt
            self.attention_guidance_negative_prompt = self.cfg.prompt_processor.negative_prompt

        self.part_prompts = animal_part_extractor(self.cfg.prompt, api_key=self.cfg.api_key)
        self.index_by_part = {}

        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.prompt_utils = self.prompt_processor()
        self.guidance_tokenizer = AutoTokenizer.from_pretrained(self.cfg.prompt_processor.pretrained_model_name_or_path, subfolder="tokenizer")
        self.part_token_index_list = []
        for part_prompt in self.part_prompts:
            token_index_list = self.get_token_index(self.guidance_tokenizer, self.cfg.prompt_processor.prompt, part_prompt)
            self.part_token_index_list.append(token_index_list)

        self.part_prompt_utils_list = []
        for prompt in self.part_prompts:
            self.cfg.prompt_processor.prompt = prompt + ", 3D asset" # follow the mvdream preprocessing
            prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
                self.cfg.prompt_processor
            )
            prompt_processor = prompt_processor()
            self.part_prompt_utils_list.append(prompt_processor)



        file_name = self.attention_guidance_prompt.replace(" ", "_")+".pth"
        save_path = os.path.join("custom/threestudio-mvdream/system/cross_attention/cache", file_name)
        # initialize the global guidance model
        if not os.path.exists(save_path):
            self.global_model = DiffusionPipeline.from_pretrained(self.cfg.global_model_name,
                                                            use_safetensors=True,
                                                            torch_dtype=torch.float16,
                                                            cache_dir=self.cfg.cache_dir)
            set_forward_sd3(self.global_model.transformer)
            register_cross_attention_hook(self.global_model.transformer)
            self.global_model = self.global_model.to("cuda")
            self.global_model.enable_model_cpu_offload()
        else:
            self.global_model = None
        self.postprocessor = DenseCRF(
            iter_max=10,
            pos_xy_std=1,
            pos_w=3,
            bi_xy_std=67,
            bi_rgb_std=3,
            bi_w=4,
        )
        self.attn_map_by_token = defaultdict(list)
        self.attn_map_batches = []


        

    
    def get_mean_variance(self,
                          heatmap: torch.Tensor):
        """
        Get the mean and variance of the heatmap
        args:
        heatmap: torch.Tensor, shape (H, W)
        
        return:
        mean: torch.Tensor, shape (2,)
        variance: torch.Tensor, shape (2,)
        """
        heatmap_flat = heatmap.flatten()
        resolution = heatmap.shape[-1]
        # Normalize the heatmap so the sum of all elements equals 1
        heatmap_normalized = heatmap_flat / torch.sum(heatmap_flat)

        # Create coordinate grids
        H, W = heatmap.shape
        x_coords, y_coords = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        x_coords = x_coords.flatten().float().to(heatmap.device)
        y_coords = y_coords.flatten().float().to(heatmap.device)

        # Calculate the first moment (mean) for x and y
        mean_x = torch.sum(x_coords * heatmap_normalized)
        mean_y = torch.sum(y_coords * heatmap_normalized)
        mean_x = (mean_x - resolution/2) / (resolution/2)
        mean_y = (mean_y - resolution/2) / (resolution/2)


        # Stack means into a single tensor for convenience
        mean = torch.tensor([mean_x, mean_y])

        # Calculate the second moment (variance)
        variance_x = torch.sum(((x_coords - mean_x) ** 2) * heatmap_normalized)
        variance_y = torch.sum(((y_coords - mean_y) ** 2) * heatmap_normalized)
        variance_x = variance_x / ((resolution/2)**2)
        variance_y = variance_y / ((resolution/2)**2)

        # Stack variances into a single tensor for convenience
        variance = torch.tensor([variance_x, variance_y])
        

        return mean, variance


    def get_token_index(self,
                        tokenizer,
                        prompt:str,
                        sub_prompt:str):
        """
        Get the token index of the sub_prompt in the prompt
        args:
        tokenizer: the tokenizer
        prompt: str, the prompt
        sub_prompt: str, the sub_prompt

        return:
        token_index_list: List[int], the list of token index
        """
        tokens = prompt2tokens(tokenizer, prompt)
        sub_tokens = prompt2tokens(tokenizer, sub_prompt)
        bos_token = tokenizer.bos_token
        eos_token = tokenizer.eos_token
        pad_token = tokenizer.pad_token
        token_index_list = []
        for idx, token in enumerate(tokens):
            if token == eos_token:
                break
            if token in [bos_token, pad_token]:
                continue
            if token in sub_tokens:
                token_index_list.append(idx)
        return token_index_list
    
                        

    def get_attn_maps_sd3(self,
                          images=None,
                          use_crf:bool = False,
                          device="cuda"):
        """
        Get the attention maps from the global guidance model
        args:

        images: torch.Tensor, shape (B, H, W, 3)
        use_crf: bool, whether to use crf to refine the attention maps

        return:
        attn_map_by_tokens: Dict[str, torch.Tensor], shape (B, num_tokens, H, W)
        """
        
        with torch.no_grad():
         # if the global model is half precision, convert the image to half precision
            view_suffix = ["back view", "side view", "front view", "side view"]
            attn_map_by_tokens = defaultdict(list)
            file_name = self.attention_guidance_prompt.replace(" ", "_")+".pth"
            save_path = os.path.join("custom/threestudio-mvdream/system/cross_attention/cache", file_name)
            if os.path.exists(save_path):
                attn_map_by_tokens = torch.load(save_path)
                return attn_map_by_tokens
            if images is not None:
                images = images.to(self.global_model.dtype)
            else:
                images = [None] * 4
            for idx, image in enumerate(images):
                # convert image to PIL image
                if image is not None:
                    image = Image.fromarray((image.cpu().detach().numpy()*255).astype(np.uint8))
                    if self.cfg.visualize:
                        image.save(os.path.join(self.cfg.visualize_save_dir, f"image_{idx}.png"))

                suffix = view_suffix[idx]
                prompt = self.attention_guidance_prompt + ", " + suffix
                output = get_attn_maps_sd3(model=self.global_model,
                                prompt=prompt,
                                negative_prompt=None,
                                only_animal_names=True,
                                animal_names=self.part_prompts,
                                image=image,
                                timestep_start=self.cfg.attention_guidance_timestep_start,
                                timestep_end=self.cfg.attention_guidance_timestep_end,
                                free_style_timestep_start=self.cfg.attention_guidance_free_style_timestep_start,
                                save_by_timestep=True,
                                save_dir=self.cfg.visualize_save_dir,
                                api_key=self.cfg.api_key,
                                normalize=True,)
            
                attn_map_by_token = output['attn_map_by_token']
                if use_crf:
                    diffused_image = output['diffused_image']

                    probmaps, _ = crf_refine(diffused_image,
                                            attn_map_by_token,
                                            self.postprocessor,
                                            save_dir=self.cfg.visualize_save_dir)


                    attn_map_by_token = attn_map_postprocess(probmaps,
                                            attn_map_by_token,
                                            amplification_factor=2,
                                            save_dir=self.cfg.visualize_save_dir,)
                
                for key, value in attn_map_by_token.items():
                    value = torch.tensor(value)
                    scale = torch.sum(torch.ones_like(value)) / torch.sum(value)
                    value = value * scale
                    attn_map_by_tokens[key].append(value)
            for key, value in attn_map_by_tokens.items():
                attn_map_by_tokens[key] = torch.stack(value).to(device)
            # process the attention maps
            torch.save(attn_map_by_tokens, save_path)

            return attn_map_by_tokens

    def get_mean_variance_from_attn_map(self,
                                        attn_map_by_token: Dict[str, torch.Tensor]):
        """
        Get the mean and variance from the attention map
        args:
        attn_map_by_token: Dict[str, torch.Tensor], the attention map by token
        """
        mean_by_token = defaultdict(list)
        variance_by_token = defaultdict(list)
        for key, value in attn_map_by_token.items():
            for view_idx in range(value.shape[0]):
                attn_map = value[view_idx]
                mean, variance = self.get_mean_variance(attn_map)
                mean_by_token[key].append(mean)
                variance_by_token[key].append(variance)
        
        return mean_by_token, variance_by_token
                

    def create_3D_gaussian(self,
                            mean_by_token: Dict[str, torch.Tensor],
                            variance_by_token: Dict[str, torch.Tensor]):
        """
        create the 3D gaussian for the part, return the mean and covariance matrix
        args:
        mean_by_token: Dict[str, torch.Tensor], the mean by token, shape (B, 4, 2)
        variance_by_token: Dict[str, torch.Tensor], the variance by token shape (B, 4, 2)

        return:
        mean_3d_by_token: Dict[str, torch.Tensor], the mean in 3D space, shape (B, 3)
        covariance_3d_by_token: Dict[str, torch.Tensor], the variance in 3D space, shape (B, 3)
        """
        mean_3d_by_token = {}
        covariance_3d_by_token = {}
        for token, means in mean_by_token.items():
            x_mean, y_mean, z_mean = [], [], []
            x_var, y_var, z_var = [], [], []
            variances = variance_by_token[token]
            for view_idx, (mean, variance) in enumerate(zip(means, variances)):
                if view_idx == 0:
                    y_mean.append(mean[1])
                    x_var.append(variance[1])
                elif view_idx == 1:
                    x_mean.append(mean[1])
                    y_var.append(variance[1])
                elif view_idx == 2:
                    # looking from the back
                    y_mean.append(- mean[1])
                    x_var.append(variance[1])
                elif view_idx == 3:
                    x_mean.append(- mean[1])
                    y_var.append(variance[1])
                z_mean.append(mean[0])
                z_var.append(variance[0])
            x_mean = torch.mean(torch.stack(x_mean))
            y_mean = torch.mean(torch.stack(y_mean))
            z_mean = torch.mean(torch.stack(z_mean))

            x_var = torch.mean(torch.stack(x_var))
            y_var = torch.mean(torch.stack(y_var))
            z_var = torch.mean(torch.stack(z_var))

            mean_3d = torch.tensor([x_mean, y_mean, -z_mean])
            covariance_3d = torch.tensor([x_var, y_var, z_var])
            mean_3d_by_token[token] = mean_3d
            covariance_3d_by_token[token] = covariance_3d
        return mean_3d_by_token, covariance_3d_by_token

    def create_3D_gaussian_from_attn_map(self,
                                        attn_map_by_token: Dict[str, torch.Tensor]):
        """
        Create the 3D gaussian from the attention map
        args:
        attn_map_by_token: Dict[str, torch.Tensor], the attention map by token
        
        return:
        mean: torch.Tensor, shape (N, 3)
        covariance: torch.Tensor, shape (N, 3, 3)
        """


        mean_by_token, variance_by_token = self.get_mean_variance_from_attn_map(attn_map_by_token)
        mean_3d_by_token, covariance_3d_by_token = self.create_3D_gaussian(mean_by_token, variance_by_token)
        return mean_3d_by_token, covariance_3d_by_token

    def render_gaussian(self,
                    batch: Dict[str, Any],
                    means: torch.Tensor, # N, 3 
                    vars: torch.Tensor, # N, 3
                    ) -> Dict[str, Any]:

        c2w = batch["c2w"] # B, 4, 4
        fovy = batch["fovy"] # B
        means = means.unsqueeze(0).to(c2w.device)
        vars = vars.unsqueeze(0).to(c2w.device)
        H, W = batch["height"], batch["width"] # B
        horizontal_angles = batch["azimuth"] # B
        #convert degree to radian
        horizontal_angles = horizontal_angles * torch.pi / 180

        focal_length: Float[Tensor, "B"] = 0.5 * H / torch.tan(0.5 * fovy)

        w2c: Float[Tensor, "B 4 4"] = torch.zeros(c2w.shape[0], 4, 4).to(c2w)
        w2c[:, :3, :3] = c2w[:, :3, :3].permute(0, 2, 1)
        w2c[:, :3, 3:] = -c2w[:, :3, :3].permute(0, 2, 1) @ c2w[:, :3, 3:]
        w2c[:, 3, 3] = 1.0
        # get the mean position in the camera coordinate
     
        means = torch.cat([means, torch.tensor([1.0]).to(means.device).unsqueeze(0).repeat(means.shape[0], 1)], dim=-1) # N, 4
   
        camera_means = []
        for mean in means:
            mean = mean.unsqueeze(0).repeat(w2c.shape[0], 1) # B, 4
            camera_mean = torch.bmm(w2c, mean.unsqueeze(-1)).squeeze(-1)
            camera_means.append(camera_mean)
        camera_means = torch.stack(camera_means) # N, B, 4

        # render mean position on the image
        camera_means = camera_means[:, :, :3] # N, B, 3

        projected_means = camera_means[:, :, :2] / camera_means[:, :, 2].unsqueeze(-1) # N, B, 2
        projected_means[:, :, 0] = projected_means[:, :, 0] * -1
        projected_means = focal_length.unsqueeze(-1) * projected_means + torch.tensor([W//2, H//2]).to(projected_means.device).unsqueeze(0)
        projected_means = torch.round(projected_means) # N, B, 2

        projected_vars = vars[:, None, :] / torch.abs(camera_means[:, :, 2])[:, :, None] # N, 1, 3 / N, B, 1 => N, B, 3
        projected_vars = projected_vars * focal_length[None, :, None] # N, B, 3

        #adjust the variance based the angle
        projected_vars[:, :, 0] = projected_vars[:, :, 0] * torch.abs(torch.sin(horizontal_angles)).unsqueeze(0)
        projected_vars[:, :, 1] = projected_vars[:, :, 1] * torch.abs(torch.cos(horizontal_angles)).unsqueeze(0)

        adjusted_vars = []
        adjusted_horizontal_var = torch.max(projected_vars[:, :, :2], dim=-1)[0]
        adjusted_vars.append(adjusted_horizontal_var)
        adjusted_vars.append(projected_vars[:, :, 2])
        adjusted_vars = torch.stack(adjusted_vars, dim=-1) # N, B, 2


        return projected_means.squeeze(), adjusted_vars.squeeze()
    

    def create_projected_gaussian(self,
                                batch,
                                attn_map_by_token: Dict[str, torch.Tensor]):
        """
        Create the projected gaussian
        args:
        batch: Dict[str, Any], the batch
        attn_map_by_token: Dict[str, torch.Tensor], the attention map by token

        return:
        projected_means: Dict[str, torch.Tensor], shape (N, B, 2)
        projected_vars: Dict[str, torch.Tensor], shape (N, B, 2)
        """

        mean_by_token, variance_by_token = self.create_3D_gaussian_from_attn_map(attn_map_by_token)
        projected_means = {}
        projected_vars = {}
        projected_gaussian_map = {}
        H, W = batch["height"], batch["width"]

        for token, means in mean_by_token.items():
            projected_mean, projected_var = self.render_gaussian(batch, means, variance_by_token[token])
            projected_means[token] = projected_mean
            projected_vars[token] = projected_var * 8 # Hardcoded the variance scale 
            projected_gaussian_map[token] = self.gaussian_filter(H, W, projected_mean, projected_var * 8) # Hardcoded the variance scale
            if self.cfg.visualize:
                file_path = token.replace(" ", "_")
                for idx in range(projected_gaussian_map[token].shape[0]):
                    gaussian_map = projected_gaussian_map[token][idx].cpu().detach().numpy()
                    gaussian_map = gaussian_map / np.max(gaussian_map)
                    gaussian_map = (gaussian_map * 255).astype(np.uint8)
                    gaussian_map = cv2.applyColorMap(gaussian_map, cv2.COLORMAP_JET)
         
                    cv2.imwrite(os.path.join(self.cfg.visualize_save_dir, f"{file_path}_{idx}.png"), gaussian_map)

        
        return projected_means, projected_vars, projected_gaussian_map


    def gaussian_filter(self,
                        H: int,
                        W: int,
                        mean: torch.Tensor,
                        var: torch.Tensor):

        """
        Create the gaussian filter
        args:
        H: int, the height of the image
        W: int, the width of the image
        mean: torch.Tensor, shape (B, 2)
        var: torch.Tensor, shape (B, 2)

        return:
        gaussian_map: torch.Tensor, shape (B, H, W)
        """

        B = mean.size(0)  # Batch size

        # Create coordinate grids
        y = torch.linspace(0, H, H, device=mean.device).view(1, H, 1).expand(B, H, W)
        x = torch.linspace(0, W, W, device=mean.device).view(1, 1, W).expand(B, H, W)
        
        # Extract mean and variance components
        mean_y = mean[:, 1].view(B, 1, 1)
        mean_x = mean[:, 0].view(B, 1, 1)
        var_y = var[:, 1].view(B, 1, 1)
        var_x = var[:, 0].view(B, 1, 1)
        
        # Compute Gaussian function
        gauss_y = torch.exp(-((y - mean_y) ** 2) / (2 * var_y))
        gauss_x = torch.exp(-((x - mean_x) ** 2) / (2 * var_x))
        
        gaussian_map = gauss_y * gauss_x
        
        # Normalize the gaussian map
        gaussian_map = gaussian_map/torch.max(gaussian_map)
        
        return gaussian_map
    
    def recenter_images(self,
                        rendered_images: List[torch.Tensor],
                        part_gaussian_map: torch.Tensor,
                        part_mean: List[torch.Tensor],
                        part_var: List[torch.Tensor],
                        batch: Dict[str, Any],
                        extend_scale: float = 2):
        """
        Recenter images based on the mean and variance of parts.
        
        Parameters:
        rendered_images (list of torch.Tensor): List of images rendered per view.
        part_gaussian_map (torch.Tensor): Gaussian map of the part.
        part_mean (list of torch.Tensor): List of part means per view.
        part_var (list of torch.Tensor): List of part variances per view.
        extend_scale (float): Scale to extend the bounding box.
        batch (dict): Dictionary containing 'width' and 'height' of the images.
        
        Returns:
        list of torch.Tensor: List of recentered images.
        """
        recentered_images = []
        recentered_part_gaussian_maps = []
        for view_idx, rendered_image_per_view in enumerate(rendered_images):
            part_mean_per_view = part_mean[view_idx]
            part_var_per_view = part_var[view_idx]

            lower_bound = (part_mean_per_view - torch.sqrt(part_var_per_view) * extend_scale).int()
            upper_bound = (part_mean_per_view + torch.sqrt(part_var_per_view) * extend_scale).int()

            lower_bound = torch.max(lower_bound, torch.tensor([0, 0]).to(lower_bound.device))
            upper_bound = torch.min(upper_bound, torch.tensor([batch["width"], batch["height"]]).to(upper_bound.device))

            recentered_image = rendered_image_per_view[lower_bound[1]:upper_bound[1], lower_bound[0]:upper_bound[0]]
            h, w, _ = recentered_image.shape
            larger_dim = max(h, w)


            # Create padding tensor
            padding = torch.zeros((larger_dim, larger_dim, 3)).to(recentered_image.device)
            
            # Compute the placement of the cropped image within the padded image
            pad_h = (larger_dim - h) // 2
            pad_w = (larger_dim - w) // 2
            
            # Place the cropped image into the center of the padded image
            padding[pad_h:pad_h + h, pad_w:pad_w + w, :] = recentered_image
            
            # Resize the padded image back to the original dimensions
            recentered_image = F.interpolate(padding.permute(2, 0, 1).unsqueeze(0), size=(batch["height"], batch["width"]), mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0)
            
            a = torch.max(part_gaussian_map)
            recentered_part_gaussian_map = part_gaussian_map[view_idx][lower_bound[1]:upper_bound[1], lower_bound[0]:upper_bound[0]]
            padding = torch.zeros((larger_dim, larger_dim)).to(recentered_image.device)
            padding[pad_h:pad_h + h, pad_w:pad_w + w] = recentered_part_gaussian_map
            recentered_part_gaussian_map = F.interpolate(padding.unsqueeze(0).unsqueeze(0), size=(batch["height"], batch["width"]), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)

            if self.cfg.visualize:
                cv2.imwrite(os.path.join(self.cfg.visualize_save_dir,
                                         f"recentered_image_{view_idx}.png"),
                                         (recentered_image.cpu().detach().numpy()*255).astype(np.uint8))
                cv2.imwrite(os.path.join(self.cfg.visualize_save_dir,
                                            f"recentered_part_gaussian_map_{view_idx}.png"),
                                            (recentered_part_gaussian_map.cpu().detach().numpy()*255).astype(np.uint8))
            recentered_images.append(recentered_image)
            recentered_part_gaussian_maps.append(recentered_part_gaussian_map)

        recentered_images = torch.stack(recentered_images)
        recentered_part_gaussian_map = torch.stack(recentered_part_gaussian_maps)
        
        return recentered_images, recentered_part_gaussian_map

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return self.renderer(**batch)

    def training_step(self, batch, batch_idx):
        out = self(batch)
    
        if self.attn_map_by_token is not None:
            (projected_means,
             projected_vars, 
             projected_gaussian_map) = \
                self.create_projected_gaussian(batch, self.attn_map_by_token)
        else:
            projected_gaussian_map = None
        

        guidance_out = self.guidance(out["comp_rgb"],
                                     self.prompt_utils,
                                     mask = projected_gaussian_map,
                                     token_index=self.part_token_index_list,
                                     cross_attention_scale=self.cfg.cross_attention_scale,
                                     self_attention_scale=self.cfg.self_attention_scale,
                                     **batch)
        # get the global attention map
        if self.cfg.use_global_attn and \
            (batch_idx % self.cfg.attention_guidance_interval == 0 or batch_idx == self.cfg.attention_guidance_start_step)\
            and self.true_global_step >= self.cfg.attention_guidance_start_step:
            attn_map_iter_idx = 0
            while attn_map_iter_idx < 100:
                attn_map_by_token = self.get_attn_maps_sd3(images=out["comp_rgb"],
                                                        use_crf=self.cfg.use_crf)
                # make sure the keys are the same
                if list(attn_map_by_token.keys()) == self.part_prompts:
                    break

            assert len(list(attn_map_by_token.keys())) == len(self.part_prompts), \
                "The number of attention maps should be the same as the number of part prompts."
            self.attn_map_by_token = attn_map_by_token

        loss = 0.0
        # if self.cfg.use_global_attn and batch_idx > self.cfg.attention_guidance_start_step:
        #     for part_idx, part_prompt_utils in enumerate(self.part_prompt_utils_list):
        #         part_gaussian_map = projected_gaussian_map[self.part_prompts[part_idx]]
        #         part_mean = projected_means[self.part_prompts[part_idx]]
        #         part_var = projected_vars[self.part_prompts[part_idx]]
        #         rendered_images = out["comp_rgb"] # 4, H, W, 3
        #         if self.cfg.use_2d_recentering:
        #             rendered_images, part_gaussian_map = self.recenter_images(rendered_images,
        #                                                                     part_gaussian_map,
        #                                                                     part_mean,
        #                                                                     part_var,
        #                                                                     batch)



        #         part_guidance_out = self.guidance(rendered_images,
        #                                     part_prompt_utils,
        #                                     mask = part_gaussian_map,
        #                                     **batch)
        #         for name, value in part_guidance_out.items():
        #             if name.startswith("loss_"):
        #                 loss += value*self.C(self.cfg.part_loss_scale)


        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])


        if self.C(self.cfg.loss.lambda_orient) > 0:
            if "normal" not in out:
                raise ValueError(
                    "Normal is required for orientation loss, no normal is found in the output."
                )
            loss_orient = (
                out["weights"].detach()
                * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
            ).sum() / (out["opacity"] > 0).sum()
            self.log("train/loss_orient", loss_orient)
            loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

        if self.C(self.cfg.loss.lambda_sparsity) > 0:
            loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
            self.log("train/loss_sparsity", loss_sparsity)
            loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        if self.C(self.cfg.loss.lambda_opaque) > 0:
            opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
            loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
            self.log("train/loss_opaque", loss_opaque)
            loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

        # z variance loss proposed in HiFA: http://arxiv.org/abs/2305.18766
        # helps reduce floaters and produce solid geometry
        if self.C(self.cfg.loss.lambda_z_variance) > 0:
            loss_z_variance = out["z_variance"][out["opacity"] > 0.5].mean()
            self.log("train/loss_z_variance", loss_z_variance)
            loss += loss_z_variance * self.C(self.cfg.loss.lambda_z_variance)

        if (
            hasattr(self.cfg.loss, "lambda_eikonal")
            and self.C(self.cfg.loss.lambda_eikonal) > 0
        ):
            loss_eikonal = (
                (torch.linalg.norm(out["sdf_grad"], ord=2, dim=-1) - 1.0) ** 2
            ).mean()
            self.log("train/loss_eikonal", loss_eikonal)
            loss += loss_eikonal * self.C(self.cfg.loss.lambda_eikonal)

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="validation_step",
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="test_step",
            step=self.true_global_step,
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )







@threestudio.register("partdream-system-legacy")
class PartDreamSystemLegacy(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        visualize_samples: bool = False
        bound_path: str = None

        # bound_intersection_path: str = None
        use_2d_recentering: bool = False

        # process prompt
        prompt: str = ""
        prompt_save_path: str = ""
        iteration_num: int = 0
        cache_dir: str = None
        save_dir: str = ""
        api_key: str = ""

        # guidance model names


        use_part_expert: bool = False

        part_expert_guidance_type: str = ""
        part_expert_guidance: dict = field(default_factory=dict)

        part_expert_prompt_processor_type: str = ""
        part_expert_prompt_processor: dict = field(default_factory=dict)


        use_learnable_layout: bool = False
        update_layout: bool = False
        layout_lr: float = 1e-5

        layout_learn_start_step: int = 2000
        layout_learn_only_step: int = 2000
        layout_learn_stop_step: int = 2000
        layout_regularization: List[bool] = None
        layout_regularization_lambda: List[float] = None
        alpha: float = 1e-5

        use_mutual_negative_prompt: bool = False

        visualize: bool = False
        gaussian_var_rescales: List[float] = None


    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()
        assert self.cfg.bound_path is not None, "bound_path should be provided."
        self.bound = np.load(self.cfg.bound_path)
        assert len(self.bound.shape) == 4, "bound should be a 4D array."
        self.bound = torch.tensor(self.bound).to(self.device) # shape (num_volumes, height, width, depth)
        # set up the weighting field initialization (assume gaussian in the first place)
        self.weight_field_mean = []
        self.weight_field_var = []
        num_of_bound = self.bound.shape[0]
        self.global_mean_shift = 0
        self.is_layout_step = False
 
        # set up the gaussian bounded renderer
        for idx in range(num_of_bound):
   
            
            bound = self.bound[idx]
            # get the indices of the bounding box
            i_indices, j_indices, k_indices = torch.nonzero(bound.float(), as_tuple=True)
            resolution = bound.shape[0]
            # get the mean of the bounding box
            mean = torch.tensor([i_indices.float().mean(), j_indices.float().mean(), k_indices.float().mean()])
            mean = (mean-resolution//2)/resolution
            if idx == num_of_bound - 1:
                self.global_mean_shift = mean
                break
            self.weight_field_mean.append(mean)
            height, width, depth = i_indices.max()-i_indices.min(), j_indices.max()-j_indices.min(), k_indices.max()-k_indices.min()
            var = torch.tensor([height, width, depth])/resolution
            self.weight_field_var.append(var)

            # convert mean and variance to leanernable parameter
        self.weight_field_mean = torch.stack(self.weight_field_mean)
        self.weight_field_var = torch.stack(self.weight_field_var)
        self.weight_field_mean = torch.nn.Parameter(self.weight_field_mean, requires_grad=True)
        self.weight_field_var = torch.nn.Parameter(self.weight_field_var, requires_grad=False)

        # create the optimizer for the mean and variance
        self.layout_optimizer = torch.optim.Adam([self.weight_field_mean],
                                                 lr=self.cfg.layout_lr)
        self.last_weight_field_mean = self.weight_field_mean.clone().detach().to(self.weight_field_mean.device)
        self.init_weight_field_mean = self.weight_field_mean.clone().detach().to(self.weight_field_mean.device)

        # initialize the gradient saver
        self.part_grad_list = []

        
        # set up sds guidance models
        if self.cfg.use_geometry_sds:
            self.guidance = threestudio.find(self.cfg.geometry_guidance_type)(self.cfg.guidance)
        else:
            self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

        # initialize mvdream prompt_processor
        optimzied_global_prompts, optimized_negative_global_prompts, optimized_part_prompts, \
            optimized_negative_part_prompts = run_model_optimization(self.cfg.prompt,
                                                                    self.cfg.part_model_names,
                                                                    self.cfg.global_model_names,
                                                                    self.cfg.api_key,
                                                                    self.cfg.iteration_num,
                                                                    self.cfg.cache_dir,
                                                                    self.cfg.prompt_save_path,)
                                                                                                               
        self.prompt_processor_list = []

        for prompt_idx, prompt in enumerate(self.cfg.prompt_list):
            self.cfg.prompt_processor.prompt=prompt
            self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
                self.cfg.prompt_processor
            )
            self.prompt_processor_list.append(self.prompt_processor)
        self.prompt_utils_list = []
        for prompt_processor in self.prompt_processor_list:
            prompt_processor = prompt_processor()
            self.prompt_utils_list.append(prompt_processor)

        if self.cfg.use_part_expert:
            self.part_expert_guidance = threestudio.find(self.cfg.part_expert_guidance_type)(self.cfg.part_expert_guidance)

            # initialize the single view prompt_processor
            self.part_expert_prompt_processor_list = []
            for prompt in self.cfg.prompt_list:
                self.cfg.part_expert_prompt_processor.prompt=prompt
                self.part_expert_prompt_processor = threestudio.find(self.cfg.part_expert_prompt_processor_type)(
                    self.cfg.part_expert_prompt_processor
                )
                self.part_expert_prompt_processor_list.append(self.part_expert_prompt_processor)
            self.part_expert_prompt_utils_list = []
            for part_expert_prompt_processor in self.part_expert_prompt_processor_list:
                part_expert_prompt_processor = part_expert_prompt_processor()
                self.part_expert_prompt_utils_list.append(part_expert_prompt_processor)
            
        # initialize the loss weight
        self.loss_weight = self.cfg.loss_weight
    

    def render_gaussian(self,
                        batch: Dict[str, Any],
                        means: torch.Tensor, # N, 3 
                        vars: torch.Tensor, # N, 3
                        ) -> Dict[str, Any]:
        means = means*2 # hard coded should fix the create boundary function
        c2w = batch["c2w"] # B, 4, 4
        fovy = batch["fovy"] # B
        H, W = batch["height"], batch["width"] # B
        horizontal_angles = batch["azimuth"] # B
        #convert degree to radian
        horizontal_angles = horizontal_angles * torch.pi / 180

        focal_length: Float[Tensor, "B"] = 0.5 * H / torch.tan(0.5 * fovy)

        w2c: Float[Tensor, "B 4 4"] = torch.zeros(c2w.shape[0], 4, 4).to(c2w)
        w2c[:, :3, :3] = c2w[:, :3, :3].permute(0, 2, 1)
        w2c[:, :3, 3:] = -c2w[:, :3, :3].permute(0, 2, 1) @ c2w[:, :3, 3:]
        w2c[:, 3, 3] = 1.0
        # get the mean position in the camera coordinate
        means = means - self.global_mean_shift.to(means.device).unsqueeze(0)
        means = torch.cat([means, torch.tensor([1.0]).to(means.device).unsqueeze(0).repeat(means.shape[0], 1)], dim=-1) # N, 4
   
        camera_means = []
        for mean in means:
            mean = mean.unsqueeze(0).repeat(w2c.shape[0], 1) # B, 4
            camera_mean = torch.bmm(w2c, mean.unsqueeze(-1)).squeeze(-1)
            camera_means.append(camera_mean)
        camera_means = torch.stack(camera_means) # N, B, 4

        # render mean position on the image
        camera_means = camera_means[:, :, :3] # N, B, 3

        projected_means = camera_means[:, :, :2] / camera_means[:, :, 2].unsqueeze(-1) # N, B, 2
        projected_means[:, :, 0] = projected_means[:, :, 0] * -1
        projected_means = focal_length.unsqueeze(-1) * projected_means + torch.tensor([W//2, H//2]).to(projected_means.device).unsqueeze(0)
        projected_means = torch.round(projected_means) # N, B, 2

        projected_vars = vars[:, None, :] / torch.abs(camera_means[:, :, 2])[:, :, None] # N, 1, 3 / N, B, 1 => N, B, 3
        projected_vars = projected_vars * focal_length[None, :, None] # N, B, 3

        #adjust the variance based the angle
        projected_vars[:, :, 0] = projected_vars[:, :, 0] * torch.abs(torch.sin(horizontal_angles)).unsqueeze(0)
        projected_vars[:, :, 1] = projected_vars[:, :, 1] * torch.abs(torch.cos(horizontal_angles)).unsqueeze(0)

        adjusted_vars = []
        adjusted_horizontal_var = torch.max(projected_vars[:, :, :2], dim=-1)[0]
        adjusted_vars.append(adjusted_horizontal_var)
        adjusted_vars.append(projected_vars[:, :, 2])
        adjusted_vars = torch.stack(adjusted_vars, dim=-1) # N, B, 2


        return projected_means, adjusted_vars
        

    def forward(self, batch: Dict[str, Any],
                bg_color: torch.Tensor = None,
                bound: torch.Tensor = None) -> Dict[str, Any]:
        
        if bg_color is None:
            return self.renderer(**batch,
                                bound=bound)
        return self.renderer(**batch,
                            bound=bound,
                            bg_color=bg_color)


    def training_step(self, batch, batch_idx):
        iter_idx = batch_idx%len(self.bound)
        is_global_step = batch_idx%len(self.bound) == len(self.bound) - 1
        bound_list = [self.bound[iter_idx]]
        loss_weight_list = [self.loss_weight[iter_idx]]
        prompt_utils_list = [self.prompt_utils_list[iter_idx]]
        if self.cfg.use_part_expert:
            part_expert_prompt_utils_list = [self.part_expert_prompt_utils_list[iter_idx]]

        # only learn the layout
        # if batch_idx < self.cfg.layout_learn_only_step + self.cfg.layout_learn_start_step and \
        #     batch_idx >= self.cfg.layout_learn_start_step and self.cfg.use_learnable_layout:
        #     is_global_step = True
        #     self.is_layout_step = True
        #     bound_list = [self.bound[-1]]
        #     loss_weight_list = [self.loss_weight[-1]]
        #     prompt_utils_list = [self.prompt_utils_list[-1]]
        #     if self.cfg.use_part_expert:
        #         part_expert_prompt_utils_list = [self.part_expert_prompt_utils_list[-1]]

    
        projected_means, projected_vars = self.render_gaussian(batch,
                                                                self.weight_field_mean,
                                                                self.weight_field_var) # N, B, 2
        if is_global_step:
            all_part_grads = torch.stack(self.part_grad_list) # 2, 4, H, W, 3
            def gaussian_filter(H, W, mean, var):
                var = var * 8  #Hardcoded
                x = torch.arange(0, W).to(mean)
                y = torch.arange(0, H).to(mean)
                y, x = torch.meshgrid(x, y)
                x = x - mean[0]
                y = y - mean[1]
                weight = torch.exp(-((x**2)/(2*var[0]**2+1e-9) + (y**2)/(2*var[1]**2+1e-9)))
                return weight
            weight_filters = []
            for idx, (projected_mean, projected_var) in enumerate(zip(projected_means, projected_vars)):
                weight_filters_for_one_bound = []
                for jdx, (mean, var) in enumerate(zip(projected_mean, projected_var)):
                    weight = gaussian_filter(batch['height'], batch['width'], mean, var/10)
                    weight_filters_for_one_bound.append(weight)
                weight_filters_for_one_bound = torch.stack(weight_filters_for_one_bound) # B, H, W
                weight_filters.append(weight_filters_for_one_bound)
            weight_filters = torch.stack(weight_filters) # N, B, H, W
            global_weight_filters = torch.sum(weight_filters, dim=0)# B, H, W

        # conduct rendering for each bounding box ========================================
        out_list = [self(batch, bound=bound) for bound in bound_list]

        # visualize the gaussian filter and the rendered images ========================================
        if self.cfg.visualize:
            if is_global_step:
                for camera_idx, weight_filter in enumerate(global_weight_filters):
                    cv2.imwrite(f"global_weight_filter_{camera_idx}.png", weight_filter.cpu().detach().numpy()*255)
                for bound_idx, weight_filters_for_one_bound in enumerate(weight_filters):
                    for camera_idx, weight_filter in enumerate(weight_filters_for_one_bound):
                        cv2.imwrite(f"weight_filter_{bound_idx}_{camera_idx}.png", weight_filter.cpu().detach().numpy()*255)
            for idx, out in enumerate(out_list):
                rendered_images = out["comp_rgb"]
                rendered_images_to_save = [Image.fromarray((rendered_image_to_save * 255).astype(np.uint8))\
                                            for rendered_image_to_save in rendered_images.cpu().detach().numpy()]
                # save
                for jdx, rendered_image_to_save in enumerate(rendered_images_to_save):
                    if is_global_step:
                        rendered_image = rendered_images[jdx].cpu().detach().numpy()*255
                   
                        image = rendered_image.copy()
                        for bound_idx in range(projected_means.shape[0]):
                            mean = projected_means[bound_idx, jdx].cpu().detach().numpy().astype(np.int32)
                            var = projected_vars[bound_idx, jdx].cpu().detach().numpy().astype(np.int32)
                            image = cv2.circle(image, (mean[0], mean[1]), 5, (255, 0, 0), -1)
                            image = cv2.ellipse(image, (mean[0], mean[1]), (var[0], var[1]), 0, 0, 360, (0, 255, 0), 2)
                        cv2.imwrite(f"combined_rendered_image_{jdx}.png", image)
                        del image
                        


                    rendered_image_to_save.save(f"rendered_images_{idx}_{jdx}.png")


        # conduct the 2d recentering ========================================
        if self.cfg.use_2d_recentering:

            for idx, out in enumerate(out_list):
                rendered_images = out["comp_rgb"]
                opacity = out["opacity"]
                opacity_mask = opacity > 0.2
                # rendered_images_to_save = [Image.fromarray((rendered_image_to_save * 255).astype(np.uint8))\
                #                             for rendered_image_to_save in rendered_images.cpu().detach().numpy()]
                # export_to_gif(rendered_images_to_save, f"rendered_images_{idx}.gif")
                bound = self.bound[idx]
                i_indices, j_indices, k_indices = torch.nonzero(bound.float(), as_tuple=True)
                bound_h, bound_w, bound_z= bound.shape[0], bound.shape[1], bound.shape[2]
                image_h, image_w = rendered_images.shape[1], rendered_images.shape[2]
                x_max, x_min = int(i_indices.max()/bound_h*image_w), int(i_indices.min()/bound_h*image_w) # lateral
                y_max, y_min = int(j_indices.max()/bound_w*image_w), int(j_indices.min()/bound_w*image_w) # frontal
                z_max, z_min = int(k_indices.max()/bound_z*image_h), int(k_indices.min()/bound_z*image_h) # sagittal
                z_min = max(z_min - int(0.1*image_h), 0)
                z_max = min(z_max + int(0.1*image_h), image_h)
                x_min = min(x_min, y_min) - int(0.1*image_h)
                x_max = max(x_max, y_max) + int(0.1*image_h)
                x_min = max(x_min, 0)
                x_max = min(x_max, image_w)

                cropped_images = []
                cropped_weight_filters = []
                crop_anchor_points = []
                cropped_part_grads_list = []
                for camera_idx, rendered_image in enumerate(rendered_images):
    
                    part_expert_opacity_mask = opacity_mask[camera_idx, :, :, 0]
                    opacity_i, opacity_j = torch.nonzero(part_expert_opacity_mask, as_tuple=True)
                    if len(opacity_i) != 0:
                        opacity_j_min, opacity_j_max = opacity_j.min(), opacity_j.max()
                        # part_expert_min_x = max(image_h - x_max, opacity_j_min-int(0.1*image_h))
                        # part_expert_max_x = min(image_h - x_min, opacity_j_max+int(0.1*image_h))
                        part_expert_min_x = opacity_j_min-int(0.1*image_h)
                        part_expert_max_x =  opacity_j_max+int(0.1*image_h)
                        part_expert_min_y = opacity_i.min()-int(0.1*image_h)
                        part_expert_max_y = opacity_i.max()+int(0.1*image_h)
                    else:
                        part_expert_min_x, part_expert_max_x = image_h-x_max, image_h-x_min
                        part_expert_min_y, part_expert_max_y = image_h-z_max, image_h-z_min

                    part_expert_max_x = min(part_expert_max_x, image_h)
                    part_expert_max_y = min(part_expert_max_y, image_h)
                    part_expert_min_x = max(part_expert_min_x, 0)
                    part_expert_min_y = max(part_expert_min_y, 0)

                    if part_expert_min_x == part_expert_max_x or part_expert_min_y == part_expert_max_y:
                        part_expert_min_x, part_expert_max_x = image_h-x_max, image_h-x_min
                        part_expert_min_y, part_expert_max_y = image_h-z_max, image_h-z_min
                    
                    if not is_global_step:
                        crop_anchor_points.append([part_expert_min_x, part_expert_max_x, part_expert_min_y, part_expert_max_y])
                

                    cropped_rendered_image = \
                        rendered_image[part_expert_min_y:part_expert_max_y, part_expert_min_x:part_expert_max_x]
        
                    if is_global_step:
                        cropped_weight_filter = weight_filters[:, camera_idx, part_expert_min_y:part_expert_max_y, part_expert_min_x:part_expert_max_x]
                        cropped_part_grad = all_part_grads[:, camera_idx, part_expert_min_y:part_expert_max_y, part_expert_min_x:part_expert_max_x]

                    rendered_image = F.interpolate(cropped_rendered_image.permute(2, 0, 1).unsqueeze(0),
                                                        (image_h, image_w),
                                                        mode='bilinear',
                                                        align_corners=True)

                    cropped_images.append(rendered_image.squeeze(0).permute(1, 2, 0))
                    if is_global_step:
                        cropped_weight_filter = F.interpolate(cropped_weight_filter.unsqueeze(0),
                                    (image_h, image_w),
                                    mode='bilinear',
                                    align_corners=True)
                        cropped_weight_filters.append(cropped_weight_filter.squeeze(0).squeeze(0))
                        cropped_part_grad = F.interpolate(cropped_part_grad.permute(0, 3, 1, 2),
                                    (image_h, image_w),
                                    mode='bilinear',
                                    align_corners=True)
                        cropped_part_grads_list.append(cropped_part_grad)
 

                if is_global_step:
                    cropped_weight_filters = torch.stack(cropped_weight_filters).permute(1, 0, 2, 3)
                    cropped_part_grads = torch.stack(cropped_part_grads_list).permute(1, 0, 3, 4, 2)# 4, 2, 3, H, W --> 2, 4, H, W, 3
                
                cropped_images = torch.stack(cropped_images)
                out_list[idx]["comp_rgb"] = cropped_images
          

        # iterate the guidance of each compositional part ========================================
        if is_global_step:
            if self.is_layout_step:
                cropped_part_grads = cropped_part_grads/torch.max(cropped_part_grads)
                if self.cfg.visualize:
                    for layout_idx, part_grad in enumerate(cropped_part_grads):
                        for camera_idx, part_grad_to_save in enumerate(part_grad):
                            part_grad_to_save = Image.fromarray((part_grad_to_save.cpu().detach().numpy() * 255).astype(np.uint8))
                            part_grad_to_save.save(f"part_grad_{layout_idx}_{camera_idx}.png")
                weighted_part_grads = cropped_part_grads * cropped_weight_filters[:, :, :, :, None] # 2, 4, H, W, 3 
                weighted_part_grads = torch.sum(weighted_part_grads, dim=0) # 4, H, W, 3
                weighted_part_grads = weighted_part_grads/torch.max(weighted_part_grads)
                if self.cfg.visualize:
                    for camera_idx, weighted_part_grad in enumerate(weighted_part_grads):
                        weighted_part_grad = Image.fromarray((weighted_part_grad.cpu().detach().numpy() * 255).astype(np.uint8))
                        weighted_part_grad.save(f"weighted_part_grad_{camera_idx}.png")
                guidance_out_list = []
                for idx, out in enumerate(out_list):
                    original_image = out["comp_rgb"]
  
                    updated_image = original_image - self.cfg.alpha * weighted_part_grads
                    if self.cfg.visualize:
                        for camera_idx, updated_image_to_save in enumerate(updated_image):
                            updated_image_to_save = Image.fromarray((updated_image_to_save.cpu().detach().numpy() * 255).astype(np.uint8))
                            updated_image_to_save.save(f"updated_image_{camera_idx}.png")
                    handle = original_image.register_hook(lambda grad: grad*0)
                    guidance_out = self.guidance(updated_image, prompt_utils_list[idx],
                                                **batch,
                                                is_global_step=is_global_step)
                    guidance_out['handle'] = handle
                    guidance_out_list.append(guidance_out)
                # guidance_out_list = [self.guidance(out["comp_rgb"],prompt_utils_list[idx],
                #                                    **batch,
                #                                    is_global_step=is_global_step) \
                #                     for idx, out in enumerate(out_list)]
              
            else:
                guidance_out_list = [self.guidance(out["comp_rgb"],prompt_utils_list[idx],
                                                   **batch,
                                                   is_global_step=is_global_step) \
                                    for idx, out in enumerate(out_list)]
        else:
            guidance_out_list = [self.guidance(out["comp_rgb"],prompt_utils_list[idx],
                                               **batch,
                                               gaussian_var_div=self.cfg.gaussian_var_rescales[iter_idx]) \
                                for idx, out in enumerate(out_list)]
            
        # save the part_grads for the next step if it
        if not is_global_step:
            single_part_means = projected_means[iter_idx] # 4, 2
            part_image_grads = guidance_out_list[0]["image_grad"] # 4, H, W, 3
            original_part_image_grads = torch.zeros_like(part_image_grads)
            for camera_idx, single_part_mean in enumerate(single_part_means):
                part_image_grad = part_image_grads[camera_idx] # H, W, 3
                original_part_image_grad = torch.zeros_like(part_image_grad)
                original_height = int(crop_anchor_points[camera_idx][3]) - int(crop_anchor_points[camera_idx][2])
                original_width = int(crop_anchor_points[camera_idx][1]) - int(crop_anchor_points[camera_idx][0])
                min_x = int(single_part_mean[0]) - int(original_width/2)
                max_x = int(single_part_mean[0]) + int(original_width/2)
                min_y = int(single_part_mean[1]) - int(original_height/2)
                max_y = int(single_part_mean[1]) + int(original_height/2)
                max_x = min(max_x, part_image_grad.shape[1])
                max_y = min(max_y, part_image_grad.shape[0])
                min_x = max(min_x, 0)
                min_y = max(min_y, 0)
                resized_part_image_grad = F.interpolate(part_image_grad.permute(2, 0, 1).unsqueeze(0),
                                                        (max_y-min_y, max_x-min_x),
                                                        mode='bilinear',
                                                        align_corners=True).squeeze(0).permute(1, 2, 0)
       
                original_part_image_grad[min_y:max_y, min_x:max_x] = resized_part_image_grad
                original_part_image_grads[camera_idx] = original_part_image_grad
            self.part_grad_list.append(original_part_image_grads)
            stop = 1


                
            
        if self.cfg.use_part_expert and not is_global_step: # we only apply it to part
            part_expert_guidance_out_list = [self.part_expert_guidance(out["comp_rgb"],part_expert_prompt_utils_list[idx], **batch) \
                                            for idx, out in enumerate(out_list)]
        else:
            part_expert_guidance_out_list = [{} for _ in range(len(out_list))]

        # iterate the loss of each compositional part ========================================
        loss = 0.0
        subpart_idx = 0
        handle = None
        for out, guidance_out, part_expert_guidance_out in zip(out_list, guidance_out_list, part_expert_guidance_out_list):
            sub_loss = 0.0
            for name, value in guidance_out.items():
                if name == "handle":
                    handle = value
                elif name != "image_grad":
                    self.log(f"train/{name}", value)
                    if name.startswith("loss_"):
                        sub_loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])
        
            
            if self.cfg.use_part_expert:
                for name, value in part_expert_guidance_out.items():
                    self.log(f"train/{name}", value)
                    if name.startswith("loss_"):
                        sub_loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])*5
            if is_global_step and self.is_layout_step:
                loss += sub_loss * loss_weight_list[subpart_idx]
                subpart_idx += 1
                self.part_grad_list = []
                self.crop_anchor_points = []
                continue
            if self.C(self.cfg.loss.lambda_orient) > 0:
                if is_global_step:
                    if "normal" not in out:
                        raise ValueError(
                            "Normal is required for orientation loss, no normal is found in the output."
                        )
                    loss_orient = (
                        out["weights"].detach()
                        * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
                    ).sum() / (out["opacity"] > 0).sum()
                    self.log("train/loss_orient", loss_orient)
                    sub_loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

            if self.C(self.cfg.loss.lambda_sparsity) > 0:
                loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
                self.log("train/loss_sparsity", loss_sparsity)
                sub_loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

            if self.C(self.cfg.loss.lambda_opaque) > 0:
                opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
                loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
                self.log("train/loss_opaque", loss_opaque)
                sub_loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

            # z variance loss proposed in HiFA: http://arxiv.org/abs/2305.18766
            # helps reduce floaters and produce solid geometry
            if self.C(self.cfg.loss.lambda_z_variance) > 0:
                loss_z_variance = out["z_variance"][out["opacity"] > 0.5].mean()
                self.log("train/loss_z_variance", loss_z_variance)
                sub_loss += loss_z_variance * self.C(self.cfg.loss.lambda_z_variance)

            if (
                hasattr(self.cfg.loss, "lambda_eikonal")
                and self.C(self.cfg.loss.lambda_eikonal) > 0
            ):
                loss_eikonal = (
                    (torch.linalg.norm(out["sdf_grad"], ord=2, dim=-1) - 1.0) ** 2
                ).mean()
                self.log("train/loss_eikonal", loss_eikonal)
                sub_loss += loss_eikonal * self.C(self.cfg.loss.lambda_eikonal)

            for name, value in self.cfg.loss.items():
                self.log(f"train_params/{name}", self.C(value))
            loss += sub_loss * loss_weight_list[subpart_idx]
            subpart_idx += 1

            if is_global_step:
                rgbs = out["comp_rgb_fg"]
                rgb_smooth_loss = 0.0
                depth_maps = out["depth"]
                depth_smooth_loss = 0.0
                for idx, depth_map, rgb in zip(range(len(depth_maps)), depth_maps, rgbs):
                    depth_smooth_loss += smooth_loss(depth_map)*2
                    rgb_smooth_loss += smooth_loss(rgb)
                smooth_loss_value = depth_smooth_loss + rgb_smooth_loss
                loss += smooth_loss_value * 5000

                # clean the list at the end of the training step
    
                self.part_grad_list = []

            self.crop_anchor_points = []

        return {"loss": loss,
                "handle": handle}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        is_global_step = batch_idx%len(self.bound) == len(self.bound) - 1
        self.dataset = self.trainer.train_dataloader.dataset
        update_end_if_possible(
            self.dataset, self.true_current_epoch, self.true_global_step
        )
        self.do_update_step_end(self.true_current_epoch, self.true_global_step)
        if outputs["handle"] is not None:
            outputs['handle'].remove()
        if batch_idx > self.cfg.layout_learn_stop_step:
            self.is_layout_step = False
        if batch_idx > self.cfg.layout_learn_start_step and is_global_step and self.cfg.use_learnable_layout\
            and batch_idx < self.cfg.layout_learn_stop_step:
            self.is_layout_step = not self.is_layout_step
    
            layout_regularization_loss = torch.tensor(0.0).to(self.weight_field_mean.device)
            for idx, (regularization, lambda_) in enumerate(zip(self.cfg.layout_regularization, self.cfg.layout_regularization_lambda)):
                if regularization:
                    self.init_weight_field_mean = self.init_weight_field_mean.to(self.weight_field_mean.device)
                    layout_regularization_loss += \
                        torch.mean((self.weight_field_mean[:, idx] - self.init_weight_field_mean[:, idx])**2)*torch.tensor(lambda_).to(self.weight_field_mean.device)

            layout_regularization_loss.backward()
            print("weight_field_mean grad:", self.weight_field_mean.grad)
            if self.weight_field_mean.grad is not None and \
                torch.isnan(self.weight_field_mean.grad).sum() == 0 and \
                    torch.isinf(self.weight_field_mean.grad).sum() == 0:
                resolution = self.bound.shape[1]

                # modify the gradient so that there is no net change in the mean
                common_mode_grad = torch.mean(self.weight_field_mean.grad, dim=0)
                self.weight_field_mean.grad = self.weight_field_mean.grad - common_mode_grad


                self.layout_optimizer.step()
           
                print("current mean:", self.weight_field_mean)
                print("last mean:", self.last_weight_field_mean)
                if self.cfg.update_layout:
                    mean_shifts = self.weight_field_mean - self.last_weight_field_mean.to(self.weight_field_mean.device)
                    print("mean shift:", mean_shifts)
                    update_last_mean_idx =torch.zeros_like(self.last_weight_field_mean)
                    for layout_idx, mean_shift in enumerate(mean_shifts):
                        for mean_idx, shift in enumerate(mean_shift):
                            if int(shift.item()*resolution) != 0:
                                update_last_mean_idx[layout_idx, mean_idx] = 1
                    
                    update_last_mean_idx = update_last_mean_idx.bool().to(self.weight_field_mean.device)
                    self.last_weight_field_mean = self.last_weight_field_mean.to(self.weight_field_mean.device)
                    if torch.sum(update_last_mean_idx) > 0:
                        self.last_weight_field_mean[update_last_mean_idx] = \
                            self.weight_field_mean[update_last_mean_idx]
                        for idx, mean_shift in enumerate(mean_shifts):
                            new_bound = torch.roll(self.bound[idx],
                                                        shifts=(int(mean_shift[0].item()*resolution),
                                                                int(mean_shift[1].item()*resolution),
                                                                int(mean_shift[2].item()*resolution)),
                                                        dims=(0, 1, 2))
                            # new_bound = new_bound + self.bound[idx]
                            # new_bound = torch.clamp(new_bound, 0, 1)
                            self.bound[idx] = new_bound
                            stop = 1
                        global_bound = torch.sum(self.bound[:-1], dim=0)
                        self.bound[-1] = torch.clamp(global_bound, 0, 1)
                self.layout_optimizer.zero_grad()
            # with torch.no_grad():
            #     for idx, mean in enumerate(self.weight_field_mean):
            #         self.weight_field_mean[idx] = torch.clamp(mean, -1, 1)
            stop = 1

    def validation_step(self, batch, batch_idx):
        bg_color = torch.tensor([1.0, 1.0, 1.0]).to(self.device)
        out = self(batch, bound=self.bound[-1], bg_color=bg_color)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="validation_step",
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        bg_color = torch.tensor([1.0, 1.0, 1.0]).to(self.device)
        out = self(batch, bound=self.bound[-1], bg_color=bg_color)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="test_step",
            step=self.true_global_step,
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )


@threestudio.register("mvdream-system-multi-bounded")
class MVDreamMultiBoundedSystem(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        visualize_samples: bool = False
        bound_path: str = None
        # bound_intersection_path: str = None
        prompt_list: List[str] = None
        loss_weight: List[float] = None
        use_iterative: bool = False
        use_2d_recentering: bool = False

        use_part_expert: bool = False

        part_expert_guidance_type: str = ""
        part_expert_guidance: dict = field(default_factory=dict)

        part_expert_prompt_processor_type: str = ""
        part_expert_prompt_processor: dict = field(default_factory=dict)

        use_geometry_sds: bool = False
        geometry_guidance_type: str = ""

        use_learnable_layout: bool = False
        update_layout: bool = False
        layout_lr: float = 1e-5

        layout_learn_start_step: int = 2000
        # layout_learn_only_step: int = 2000
        layout_learn_stop_step: int = 2000
        layout_regularization: List[bool] = None
        layout_regularization_lambda: List[float] = None

        use_mutual_negative_prompt: bool = False

        visualize: bool = False
        gaussian_var_rescales: List[float] = None

        guidance_weight: List[float] = None



    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()
        assert self.cfg.bound_path is not None, "bound_path should be provided."
        self.bound = np.load(self.cfg.bound_path)
        assert len(self.bound.shape) == 4, "bound should be a 4D array."
        self.bound = torch.tensor(self.bound).to(self.device) # shape (num_volumes, height, width, depth)
        self.use_iterative = self.cfg.use_iterative
        # set up the weighting field initialization (assume gaussian in the first place)
        self.weight_field_mean = []
        self.weight_field_var = []
        num_of_bound = self.bound.shape[0]
        self.global_mean_shift = 0
        self.is_layout_step = False
 
        # set up the gaussian bounded renderer
        for idx in range(num_of_bound):
   
            
            bound = self.bound[idx]
            # get the indices of the bounding box
            i_indices, j_indices, k_indices = torch.nonzero(bound.float(), as_tuple=True)
            resolution = bound.shape[0]
            # get the mean of the bounding box
            mean = torch.tensor([i_indices.float().mean(), j_indices.float().mean(), k_indices.float().mean()])
            mean = (mean-resolution//2)/resolution
            if idx == num_of_bound - 1:
                self.global_mean_shift = mean
                break
            self.weight_field_mean.append(mean)
            height, width, depth = i_indices.max()-i_indices.min(), j_indices.max()-j_indices.min(), k_indices.max()-k_indices.min()
            var = torch.tensor([height, width, depth])/resolution
            self.weight_field_var.append(var)

            # convert mean and variance to leanernable parameter
        self.weight_field_mean = torch.stack(self.weight_field_mean)
        self.weight_field_var = torch.stack(self.weight_field_var)
        self.weight_field_mean = torch.nn.Parameter(self.weight_field_mean, requires_grad=True)
        self.weight_field_var = torch.nn.Parameter(self.weight_field_var, requires_grad=False)

        # create the optimizer for the mean and variance
        self.layout_optimizer = torch.optim.Adam([self.weight_field_mean],
                                                 lr=self.cfg.layout_lr)
        self.last_weight_field_mean = self.weight_field_mean.clone().detach().to(self.weight_field_mean.device)
        self.init_weight_field_mean = self.weight_field_mean.clone().detach().to(self.weight_field_mean.device)


        
        # set up sds guidance models
        if self.cfg.use_geometry_sds:
            self.guidance = threestudio.find(self.cfg.geometry_guidance_type)(self.cfg.guidance)
        else:
            self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

        # initialize mvdream prompt_processor
        self.prompt_processor_list = []
        original_negative_prompt = self.cfg.prompt_processor.negative_prompt
        for prompt_idx, prompt in enumerate(self.cfg.prompt_list):
            self.cfg.prompt_processor.prompt=prompt
            if self.cfg.use_mutual_negative_prompt and prompt_idx != len(self.cfg.prompt_list) - 1:
                all_negative_prompt = original_negative_prompt
                for negative_prompt_idx, negative_prompt in enumerate(self.cfg.prompt_list[:-1]):
                    if negative_prompt_idx == prompt_idx:
                        continue
                    all_negative_prompt += f", {negative_prompt}"
                self.cfg.prompt_processor.negative_prompt = all_negative_prompt
            self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
                self.cfg.prompt_processor
            )
            self.prompt_processor_list.append(self.prompt_processor)
        self.prompt_utils_list = []
        for prompt_processor in self.prompt_processor_list:
            prompt_processor = prompt_processor()
            self.prompt_utils_list.append(prompt_processor)

        if self.cfg.use_part_expert:
            self.part_expert_guidance = threestudio.find(self.cfg.part_expert_guidance_type)(self.cfg.part_expert_guidance)

            # initialize the single view prompt_processor
            self.part_expert_prompt_processor_list = []
            for prompt in self.cfg.prompt_list:
                self.cfg.part_expert_prompt_processor.prompt=prompt
                self.part_expert_prompt_processor = threestudio.find(self.cfg.part_expert_prompt_processor_type)(
                    self.cfg.part_expert_prompt_processor
                )
                self.part_expert_prompt_processor_list.append(self.part_expert_prompt_processor)
            self.part_expert_prompt_utils_list = []
            for part_expert_prompt_processor in self.part_expert_prompt_processor_list:
                part_expert_prompt_processor = part_expert_prompt_processor()
                self.part_expert_prompt_utils_list.append(part_expert_prompt_processor)
            
        # initialize the loss weight
        self.loss_weight = self.cfg.loss_weight


    def render_gaussian(self,
                        batch: Dict[str, Any],
                        means: torch.Tensor, # N, 3 
                        vars: torch.Tensor, # N, 3
                        ) -> Dict[str, Any]:
        means = means*2 # hard coded should fix the create boundary function
        c2w = batch["c2w"] # B, 4, 4
        fovy = batch["fovy"] # B
        H, W = batch["height"], batch["width"] # B
        horizontal_angles = batch["azimuth"] # B
        #convert degree to radian
        horizontal_angles = horizontal_angles * torch.pi / 180

        focal_length: Float[Tensor, "B"] = 0.5 * H / torch.tan(0.5 * fovy)

        w2c: Float[Tensor, "B 4 4"] = torch.zeros(c2w.shape[0], 4, 4).to(c2w)
        w2c[:, :3, :3] = c2w[:, :3, :3].permute(0, 2, 1)
        w2c[:, :3, 3:] = -c2w[:, :3, :3].permute(0, 2, 1) @ c2w[:, :3, 3:]
        w2c[:, 3, 3] = 1.0
        # get the mean position in the camera coordinate
        means = means - self.global_mean_shift.to(means.device).unsqueeze(0)
        means = torch.cat([means, torch.tensor([1.0]).to(means.device).unsqueeze(0).repeat(means.shape[0], 1)], dim=-1) # N, 4
   
        camera_means = []
        for mean in means:
            mean = mean.unsqueeze(0).repeat(w2c.shape[0], 1) # B, 4
            camera_mean = torch.bmm(w2c, mean.unsqueeze(-1)).squeeze(-1)
            camera_means.append(camera_mean)
        camera_means = torch.stack(camera_means) # N, B, 4

        # render mean position on the image
        camera_means = camera_means[:, :, :3] # N, B, 3

        projected_means = camera_means[:, :, :2] / camera_means[:, :, 2].unsqueeze(-1) # N, B, 2
        projected_means[:, :, 0] = projected_means[:, :, 0] * -1
        projected_means = focal_length.unsqueeze(-1) * projected_means + torch.tensor([W//2, H//2]).to(projected_means.device).unsqueeze(0)
        projected_means = torch.round(projected_means) # N, B, 2

        projected_vars = vars[:, None, :] / torch.abs(camera_means[:, :, 2])[:, :, None] # N, 1, 3 / N, B, 1 => N, B, 3
        projected_vars = projected_vars * focal_length[None, :, None] # N, B, 3

        #adjust the variance based the angle
        projected_vars[:, :, 0] = projected_vars[:, :, 0] * torch.abs(torch.sin(horizontal_angles)).unsqueeze(0)
        projected_vars[:, :, 1] = projected_vars[:, :, 1] * torch.abs(torch.cos(horizontal_angles)).unsqueeze(0)

        adjusted_vars = []
        adjusted_horizontal_var = torch.max(projected_vars[:, :, :2], dim=-1)[0]
        adjusted_vars.append(adjusted_horizontal_var)
        adjusted_vars.append(projected_vars[:, :, 2])
        adjusted_vars = torch.stack(adjusted_vars, dim=-1) # N, B, 2


        return projected_means, adjusted_vars
        

    def forward(self, batch: Dict[str, Any],
                bg_color: torch.Tensor = None,
                bound: torch.Tensor = None) -> Dict[str, Any]:
        
        if bg_color is None:
            return self.renderer(**batch,
                                bound=bound)
        return self.renderer(**batch,
                            bound=bound,
                            bg_color=bg_color)

    def training_step(self, batch, batch_idx):
        if self.use_iterative:
            iter_idx = batch_idx%len(self.bound)
            is_global_step = batch_idx%len(self.bound) == len(self.bound) - 1
            bound_list = [self.bound[iter_idx]]
            loss_weight_list = [self.loss_weight[iter_idx]]
            prompt_utils_list = [self.prompt_utils_list[iter_idx]]
            if self.cfg.use_part_expert:
                part_expert_prompt_utils_list = [self.part_expert_prompt_utils_list[iter_idx]]

            # # only learn the layout
            # if batch_idx < self.cfg.layout_learn_only_step + self.cfg.layout_learn_start_step and \
            #     batch_idx >= self.cfg.layout_learn_start_step and self.cfg.use_learnable_layout:
            #     is_global_step = True
            #     self.is_layout_step = True
            #     bound_list = [self.bound[-1]]
            #     loss_weight_list = [self.loss_weight[-1]]
            #     prompt_utils_list = [self.prompt_utils_list[-1]]
            #     if self.cfg.use_part_expert:
            #         part_expert_prompt_utils_list = [self.part_expert_prompt_utils_list[-1]]

        

        if is_global_step:
            projected_means, projected_vars = self.render_gaussian(batch,
                                                                    self.weight_field_mean,
                                                                    self.weight_field_var) # N, B, 2
            def gaussian_filter(H, W, mean, var):
                var = var * 8  #Hardcoded
                x = torch.arange(0, W).to(mean)
                y = torch.arange(0, H).to(mean)
                y, x = torch.meshgrid(x, y)
                x = x - mean[0]
                y = y - mean[1]
                weight = torch.exp(-((x**2)/(2*var[0]**2) + (y**2)/(2*var[1]**2)))
                return weight
            weight_filters = []
            for idx, (projected_mean, projected_var) in enumerate(zip(projected_means, projected_vars)):
                weight_filters_for_one_bound = []
                for jdx, (mean, var) in enumerate(zip(projected_mean, projected_var)):
                    weight = gaussian_filter(batch['height'], batch['width'], mean, var/10)
                    weight_filters_for_one_bound.append(weight)
                weight_filters_for_one_bound = torch.stack(weight_filters_for_one_bound) # B, H, W
                weight_filters.append(weight_filters_for_one_bound)
            weight_filters = torch.stack(weight_filters) # N, B, H, W
            global_weight_filters = torch.sum(weight_filters, dim=0)# B, H, W

        # conduct rendering for each bounding box ========================================
        out_list = [self(batch, bound=bound) for bound in bound_list]

        # visualize the gaussian filter and the rendered images ========================================
        if self.cfg.visualize:
            if is_global_step:
                for camera_idx, weight_filter in enumerate(global_weight_filters):
                    cv2.imwrite(f"global_weight_filter_{camera_idx}.png", weight_filter.cpu().detach().numpy()*255)
                for bound_idx, weight_filters_for_one_bound in enumerate(weight_filters):
                    for camera_idx, weight_filter in enumerate(weight_filters_for_one_bound):
                        cv2.imwrite(f"weight_filter_{bound_idx}_{camera_idx}.png", weight_filter.cpu().detach().numpy()*255)
            for idx, out in enumerate(out_list):
                rendered_images = out["comp_rgb"]
                rendered_images_to_save = [Image.fromarray((rendered_image_to_save * 255).astype(np.uint8))\
                                            for rendered_image_to_save in rendered_images.cpu().detach().numpy()]
                # save
                for jdx, rendered_image_to_save in enumerate(rendered_images_to_save):
                    if is_global_step:
                        rendered_image = rendered_images[jdx].cpu().detach().numpy()*255
                   
                        image = rendered_image.copy()
                        for bound_idx in range(projected_means.shape[0]):
                            mean = projected_means[bound_idx, jdx].cpu().detach().numpy().astype(np.int32)
                            var = projected_vars[bound_idx, jdx].cpu().detach().numpy().astype(np.int32)
                            image = cv2.circle(image, (mean[0], mean[1]), 5, (255, 0, 0), -1)
                            image = cv2.ellipse(image, (mean[0], mean[1]), (var[0], var[1]), 0, 0, 360, (0, 255, 0), 2)
                        cv2.imwrite(f"combined_rendered_image_{jdx}.png", image)
                        del image
                        


                    rendered_image_to_save.save(f"rendered_images_{idx}_{jdx}.png")


        # conduct the 2d recentering ========================================
        if self.cfg.use_2d_recentering:
            for idx, out in enumerate(out_list):
                rendered_images = out["comp_rgb"]
                opacity = out["opacity"]
                opacity_mask = opacity > 0.2
                # rendered_images_to_save = [Image.fromarray((rendered_image_to_save * 255).astype(np.uint8))\
                #                             for rendered_image_to_save in rendered_images.cpu().detach().numpy()]
                # export_to_gif(rendered_images_to_save, f"rendered_images_{idx}.gif")
                bound = self.bound[idx]
                i_indices, j_indices, k_indices = torch.nonzero(bound.float(), as_tuple=True)
                bound_h, bound_w, bound_z= bound.shape[0], bound.shape[1], bound.shape[2]
                image_h, image_w = rendered_images.shape[1], rendered_images.shape[2]
                x_max, x_min = int(i_indices.max()/bound_h*image_w), int(i_indices.min()/bound_h*image_w) # lateral
                y_max, y_min = int(j_indices.max()/bound_w*image_w), int(j_indices.min()/bound_w*image_w) # frontal
                z_max, z_min = int(k_indices.max()/bound_z*image_h), int(k_indices.min()/bound_z*image_h) # sagittal
                z_min = max(z_min - int(0.1*image_h), 0)
                z_max = min(z_max + int(0.1*image_h), image_h)
                x_min = min(x_min, y_min) - int(0.1*image_h)
                x_max = max(x_max, y_max) + int(0.1*image_h)
                x_min = max(x_min, 0)
                x_max = min(x_max, image_w)

                cropped_images = []
                if is_global_step:
                    cropped_weight_filters = []
                for jdx, rendered_image in enumerate(rendered_images):
                    if is_global_step:
                        global_weight_filter = global_weight_filters[jdx]
                    part_expert_opacity_mask = opacity_mask[jdx, :, :, 0]
                    opacity_i, opacity_j = torch.nonzero(part_expert_opacity_mask, as_tuple=True)
                    if len(opacity_i) != 0:
                        opacity_j_min, opacity_j_max = opacity_j.min(), opacity_j.max()
                        # part_expert_min_x = max(image_h - x_max, opacity_j_min-int(0.1*image_h))
                        # part_expert_max_x = min(image_h - x_min, opacity_j_max+int(0.1*image_h))
                        part_expert_min_x = opacity_j_min-int(0.1*image_h)
                        part_expert_max_x =  opacity_j_max+int(0.1*image_h)
                        part_expert_min_y = opacity_i.min()-int(0.1*image_h)
                        part_expert_max_y = opacity_i.max()+int(0.1*image_h)
                    else:
                        part_expert_min_x, part_expert_max_x = image_h-x_max, image_h-x_min
                        part_expert_min_y, part_expert_max_y = image_h-z_max, image_h-z_min

                    part_expert_max_x = min(part_expert_max_x, image_h)
                    part_expert_max_y = min(part_expert_max_y, image_h)
                    part_expert_min_x = max(part_expert_min_x, 0)
                    part_expert_min_y = max(part_expert_min_y, 0)

                    cropped_rendered_image = \
                        rendered_image[part_expert_min_y:part_expert_max_y, part_expert_min_x:part_expert_max_x]
          
                    h, w = cropped_rendered_image.shape[0], cropped_rendered_image.shape[1]
                    if is_global_step:
                        global_weight_filter = global_weight_filter[part_expert_min_y:part_expert_max_y, part_expert_min_x:part_expert_max_x]
                    if h==0 or w==0:
                        cropped_rendered_image = rendered_image[image_h-z_max:image_h-z_min, image_h-x_max:image_h-x_min]
                        if is_global_step:
                            global_weight_filter = global_weight_filter[image_h-z_max:image_h-z_min, image_h-x_max:image_h-x_min]
          
                    rendered_image = F.interpolate(cropped_rendered_image.permute(2, 0, 1).unsqueeze(0),
                                                        (image_h, image_w),
                                                        mode='bilinear',
                                                        align_corners=True)

                    cropped_images.append(rendered_image.squeeze(0).permute(1, 2, 0))
                    if is_global_step:
                        global_weight_filter = F.interpolate(global_weight_filter.unsqueeze(0).unsqueeze(0),
                                    (image_h, image_w),
                                    mode='bilinear',
                                    align_corners=True)
                        cropped_weight_filters.append(global_weight_filter.squeeze(0).squeeze(0))

                if is_global_step:
                    cropped_weight_filters = torch.stack(cropped_weight_filters)[:, None, :, :] # B, 1, H, W
                
                cropped_images = torch.stack(cropped_images)
                out_list[idx]["comp_rgb"] = cropped_images
        

        # iterate the guidance of each compositional part ========================================
        if is_global_step:
            if self.is_layout_step:
                guidance_out_list = [self.guidance(out["comp_rgb"],prompt_utils_list[idx],
                                                   **batch,
                                                   weight_filters=cropped_weight_filters,
                                                   is_global_step=is_global_step) \
                                    for idx, out in enumerate(out_list)]
            else:
                guidance_out_list = [self.guidance(out["comp_rgb"],prompt_utils_list[idx],
                                                   **batch,
                                                   is_global_step=is_global_step) \
                                    for idx, out in enumerate(out_list)]
        else:
            guidance_out_list = [self.guidance(out["comp_rgb"],prompt_utils_list[idx],
                                               **batch,
                                               gaussian_var_div=self.cfg.gaussian_var_rescales[iter_idx]) \
                                for idx, out in enumerate(out_list)]
            
        if self.cfg.use_part_expert:
            part_expert_guidance_out_list = [self.part_expert_guidance(out["comp_rgb"],part_expert_prompt_utils_list[idx], **batch) \
                                            for idx, out in enumerate(out_list)]
        else:
            part_expert_guidance_out_list = [{} for _ in range(len(out_list))]

        # iterate the loss of each compositional part ========================================
        loss = 0.0
        subpart_idx = 0
        handle = None
        for out, guidance_out, part_expert_guidance_out in zip(out_list, guidance_out_list, part_expert_guidance_out_list):
            sub_loss = 0.0
            for name, value in guidance_out.items():
                if name == "handle":
                    handle = value
                else:
                    self.log(f"train/{name}", value)
                    if name.startswith("loss_"):
                        sub_loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")]) * self.C(self.cfg.guidance_weight[0])
        
            
            if self.cfg.use_part_expert:
                for name, value in part_expert_guidance_out.items():
                    self.log(f"train/{name}", value)
                    if name.startswith("loss_"):
                        sub_loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])* self.C(self.cfg.guidance_weight[1])
            if is_global_step and self.is_layout_step:
                loss += sub_loss * loss_weight_list[subpart_idx]
                subpart_idx += 1
                continue
            if self.C(self.cfg.loss.lambda_orient) > 0:
                if (self.use_iterative and is_global_step) or not self.use_iterative:
                    if "normal" not in out:
                        raise ValueError(
                            "Normal is required for orientation loss, no normal is found in the output."
                        )
                    loss_orient = (
                        out["weights"].detach()
                        * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
                    ).sum() / (out["opacity"] > 0).sum()
                    self.log("train/loss_orient", loss_orient)
                    sub_loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

            if self.C(self.cfg.loss.lambda_sparsity) > 0:
                loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
                self.log("train/loss_sparsity", loss_sparsity)
                sub_loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

            if self.C(self.cfg.loss.lambda_opaque) > 0:
                opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
                loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
                self.log("train/loss_opaque", loss_opaque)
                sub_loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

            # z variance loss proposed in HiFA: http://arxiv.org/abs/2305.18766
            # helps reduce floaters and produce solid geometry
            if self.C(self.cfg.loss.lambda_z_variance) > 0:
                loss_z_variance = out["z_variance"][out["opacity"] > 0.5].mean()
                self.log("train/loss_z_variance", loss_z_variance)
                sub_loss += loss_z_variance * self.C(self.cfg.loss.lambda_z_variance)

            if (
                hasattr(self.cfg.loss, "lambda_eikonal")
                and self.C(self.cfg.loss.lambda_eikonal) > 0
            ):
                loss_eikonal = (
                    (torch.linalg.norm(out["sdf_grad"], ord=2, dim=-1) - 1.0) ** 2
                ).mean()
                self.log("train/loss_eikonal", loss_eikonal)
                sub_loss += loss_eikonal * self.C(self.cfg.loss.lambda_eikonal)

            for name, value in self.cfg.loss.items():
                self.log(f"train_params/{name}", self.C(value))
            loss += sub_loss * loss_weight_list[subpart_idx]
            subpart_idx += 1

            if is_global_step:
                rgbs = out["comp_rgb_fg"]
                rgb_smooth_loss = 0.0
                depth_maps = out["depth"]
                depth_smooth_loss = 0.0
                for idx, depth_map, rgb in zip(range(len(depth_maps)), depth_maps, rgbs):
                    depth_smooth_loss += smooth_loss(depth_map)*2
                    rgb_smooth_loss += smooth_loss(rgb)
                smooth_loss_value = depth_smooth_loss + rgb_smooth_loss
                loss += smooth_loss_value * 5000

        return {"loss": loss,
                "handle": handle}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        is_global_step = batch_idx%len(self.bound) == len(self.bound) - 1
        self.dataset = self.trainer.train_dataloader.dataset
        update_end_if_possible(
            self.dataset, self.true_current_epoch, self.true_global_step
        )
        self.do_update_step_end(self.true_current_epoch, self.true_global_step)
        if outputs["handle"] is not None:
            outputs['handle'].remove()
        if batch_idx > self.cfg.layout_learn_start_step or not self.cfg.use_learnable_layout:
            self.is_layout_step = False
        if batch_idx > self.cfg.layout_learn_start_step and is_global_step and self.cfg.use_learnable_layout\
            and batch_idx < self.cfg.layout_learn_stop_step:
            self.is_layout_step = not self.is_layout_step
    
            layout_regularization_loss = torch.tensor(0.0).to(self.weight_field_mean.device)
            for idx, (regularization, lambda_) in enumerate(zip(self.cfg.layout_regularization, self.cfg.layout_regularization_lambda)):
                if regularization:
                    self.init_weight_field_mean = self.init_weight_field_mean.to(self.weight_field_mean.device)
                    layout_regularization_loss += \
                        torch.mean((self.weight_field_mean[:, idx] - self.init_weight_field_mean[:, idx])**2)*torch.tensor(lambda_).to(self.weight_field_mean.device)

            layout_regularization_loss.backward()
            if self.weight_field_mean.grad is not None and \
                torch.isnan(self.weight_field_mean.grad).sum() == 0 and \
                    torch.isinf(self.weight_field_mean.grad).sum() == 0:
                resolution = self.bound.shape[1]

                # modify the gradient so that there is no net change in the mean
                common_mode_grad = torch.mean(self.weight_field_mean.grad, dim=0)
                self.weight_field_mean.grad = self.weight_field_mean.grad - common_mode_grad


                self.layout_optimizer.step()
           
                print("current mean:", self.weight_field_mean)
                print("last mean:", self.last_weight_field_mean)
                if self.cfg.update_layout:
                    mean_shifts = self.weight_field_mean - self.last_weight_field_mean.to(self.weight_field_mean.device)
                    print("mean shift:", mean_shifts)
                    update_last_mean_idx =torch.zeros_like(self.last_weight_field_mean)
                    for layout_idx, mean_shift in enumerate(mean_shifts):
                        for mean_idx, shift in enumerate(mean_shift):
                            if int(shift.item()*resolution) != 0:
                                update_last_mean_idx[layout_idx, mean_idx] = 1
                    
                    update_last_mean_idx = update_last_mean_idx.bool().to(self.weight_field_mean.device)
                    self.last_weight_field_mean = self.last_weight_field_mean.to(self.weight_field_mean.device)
                    if torch.sum(update_last_mean_idx) > 0:
                        self.last_weight_field_mean[update_last_mean_idx] = \
                            self.weight_field_mean[update_last_mean_idx]
                        for idx, mean_shift in enumerate(mean_shifts):
                            new_bound = torch.roll(self.bound[idx],
                                                        shifts=(int(mean_shift[0].item()*resolution),
                                                                int(mean_shift[1].item()*resolution),
                                                                int(mean_shift[2].item()*resolution)),
                                                        dims=(0, 1, 2))
                            # new_bound = new_bound + self.bound[idx]
                            # new_bound = torch.clamp(new_bound, 0, 1)
                            self.bound[idx] = new_bound
                            stop = 1
                        global_bound = torch.sum(self.bound[:-1], dim=0)
                        self.bound[-1] = torch.clamp(global_bound, 0, 1)
                self.layout_optimizer.zero_grad()
            # with torch.no_grad():
            #     for idx, mean in enumerate(self.weight_field_mean):
            #         self.weight_field_mean[idx] = torch.clamp(mean, -1, 1)
            stop = 1

    def validation_step(self, batch, batch_idx):
        bg_color = torch.tensor([1.0, 1.0, 1.0]).to(self.device)
        out = self(batch, bound=self.bound[-1], bg_color=bg_color)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="validation_step",
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        bg_color = torch.tensor([1.0, 1.0, 1.0]).to(self.device)
        out = self(batch, bound=self.bound[-1], bg_color=bg_color)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="test_step",
            step=self.true_global_step,
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )



@threestudio.register("mvdream-deepfloyd-system")
class MVDreamDeepfloydSystem(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        visualize_samples: bool = False

        part_expert_guidance_type: str = ""
        part_expert_guidance: dict = field(default_factory=dict)

        part_expert_prompt_processor_type: str = ""
        part_expert_prompt_processor: dict = field(default_factory=dict)

        guidance_weight: List[float] = None

    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.prompt_utils = self.prompt_processor()


        self.part_expert_guidance = threestudio.find(self.cfg.part_expert_guidance_type)(self.cfg.part_expert_guidance)
        self.cfg.part_expert_prompt_processor.prompt = self.cfg.prompt_processor.prompt
        self.part_expert_prompt_processor = threestudio.find(self.cfg.part_expert_prompt_processor_type)(
            self.cfg.part_expert_prompt_processor
        )

        self.part_expert_prompt_utils = self.part_expert_prompt_processor()

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return self.renderer(**batch)

    def training_step(self, batch, batch_idx):
        out = self(batch)

        guidance_out = self.guidance(out["comp_rgb"], self.prompt_utils, **batch)
        part_expert_guidance_out = self.part_expert_guidance(out["comp_rgb"], self.part_expert_prompt_utils, **batch)

        loss = 0.0

        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")]) * self.C(self.cfg.guidance_weight[0])
            
        for name, value in part_expert_guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")]) * self.C(self.cfg.guidance_weight[1])

        if self.C(self.cfg.loss.lambda_orient) > 0:
            if "normal" not in out:
                raise ValueError(
                    "Normal is required for orientation loss, no normal is found in the output."
                )
            loss_orient = (
                out["weights"].detach()
                * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
            ).sum() / (out["opacity"] > 0).sum()
            self.log("train/loss_orient", loss_orient)
            loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

        if self.C(self.cfg.loss.lambda_sparsity) > 0:
            loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
            self.log("train/loss_sparsity", loss_sparsity)
            loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        if self.C(self.cfg.loss.lambda_opaque) > 0:
            opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
            loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
            self.log("train/loss_opaque", loss_opaque)
            loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

        # z variance loss proposed in HiFA: http://arxiv.org/abs/2305.18766
        # helps reduce floaters and produce solid geometry
        if self.C(self.cfg.loss.lambda_z_variance) > 0:
            loss_z_variance = out["z_variance"][out["opacity"] > 0.5].mean()
            self.log("train/loss_z_variance", loss_z_variance)
            loss += loss_z_variance * self.C(self.cfg.loss.lambda_z_variance)

        if (
            hasattr(self.cfg.loss, "lambda_eikonal")
            and self.C(self.cfg.loss.lambda_eikonal) > 0
        ):
            loss_eikonal = (
                (torch.linalg.norm(out["sdf_grad"], ord=2, dim=-1) - 1.0) ** 2
            ).mean()
            self.log("train/loss_eikonal", loss_eikonal)
            loss += loss_eikonal * self.C(self.cfg.loss.lambda_eikonal)

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="validation_step",
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="test_step",
            step=self.true_global_step,
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )



@threestudio.register("mvdream-system")
class MVDreamSystem(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        visualize_samples: bool = False

    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.prompt_utils = self.prompt_processor()

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return self.renderer(**batch)

    def training_step(self, batch, batch_idx):
        out = self(batch)

        guidance_out = self.guidance(out["comp_rgb"], self.prompt_utils, **batch)

        loss = 0.0

        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        if self.C(self.cfg.loss.lambda_orient) > 0:
            if "normal" not in out:
                raise ValueError(
                    "Normal is required for orientation loss, no normal is found in the output."
                )
            loss_orient = (
                out["weights"].detach()
                * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
            ).sum() / (out["opacity"] > 0).sum()
            self.log("train/loss_orient", loss_orient)
            loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

        if self.C(self.cfg.loss.lambda_sparsity) > 0:
            loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
            self.log("train/loss_sparsity", loss_sparsity)
            loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        if self.C(self.cfg.loss.lambda_opaque) > 0:
            opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
            loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
            self.log("train/loss_opaque", loss_opaque)
            loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

        # z variance loss proposed in HiFA: http://arxiv.org/abs/2305.18766
        # helps reduce floaters and produce solid geometry
        if self.C(self.cfg.loss.lambda_z_variance) > 0:
            loss_z_variance = out["z_variance"][out["opacity"] > 0.5].mean()
            self.log("train/loss_z_variance", loss_z_variance)
            loss += loss_z_variance * self.C(self.cfg.loss.lambda_z_variance)

        if (
            hasattr(self.cfg.loss, "lambda_eikonal")
            and self.C(self.cfg.loss.lambda_eikonal) > 0
        ):
            loss_eikonal = (
                (torch.linalg.norm(out["sdf_grad"], ord=2, dim=-1) - 1.0) ** 2
            ).mean()
            self.log("train/loss_eikonal", loss_eikonal)
            loss += loss_eikonal * self.C(self.cfg.loss.lambda_eikonal)

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="validation_step",
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="test_step",
            step=self.true_global_step,
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
