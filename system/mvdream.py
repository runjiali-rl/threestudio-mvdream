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



def smooth_loss(inputs):
    if inputs.shape[0] > 3:
        inputs = inputs.permute(2, 0, 1)
    L1 = torch.mean(torch.abs((inputs[:,:,:-1] - inputs[:,:,1:])))
    L2 = torch.mean(torch.abs((inputs[:,:-1,:] - inputs[:,1:,:])))
    L3 = torch.mean(torch.abs((inputs[:,:-1,:-1] - inputs[:,1:,1:])))
    L4 = torch.mean(torch.abs((inputs[:,1:,:-1] - inputs[:,:-1,1:])))
    return (L1 + L2 + L3 + L4) / 4
  

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

        use_single_view: bool = False

        single_view_guidance_type: str = ""
        single_view_guidance: dict = field(default_factory=dict)

        single_view_prompt_processor_type: str = ""
        single_view_prompt_processor: dict = field(default_factory=dict)

        use_geometry_sds: bool = False
        geometry_guidance_type: str = ""


        visualize: bool = False

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
        for idx in range(num_of_bound):
            bound = self.bound[idx]
            # get the indices of the bounding box
            i_indices, j_indices, k_indices = torch.nonzero(bound.float(), as_tuple=True)
            resolution = bound.shape[0]
            # get the mean of the bounding box
            mean = torch.tensor([i_indices.float().mean(), j_indices.float().mean(), k_indices.float().mean()])
            mean = (mean-resolution//2)/resolution
            self.weight_field_mean.append(mean)
            height, width, depth = i_indices.max()-i_indices.min(), j_indices.max()-j_indices.min(), k_indices.max()-k_indices.min()
            var = torch.tensor([height, width, depth])/resolution * 0.04
            self.weight_field_var.append(var)
        # convert mean and variance to leanernable parameter
        self.weight_field_mean = torch.stack(self.weight_field_mean)
        self.weight_field_var = torch.stack(self.weight_field_var)
        self.weight_field_mean = torch.nn.Parameter(self.weight_field_mean, requires_grad=True)
        self.weight_field_var = torch.nn.Parameter(self.weight_field_var, requires_grad=False)
        # set up sds guidance models
        if self.cfg.use_geometry_sds:
            self.guidance = threestudio.find(self.cfg.geometry_guidance_type)(self.cfg.guidance)
        else:
            self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

        # initialize mvdream prompt_processor
        self.prompt_processor_list = []
        for prompt in self.cfg.prompt_list:
            self.cfg.prompt_processor.prompt=prompt

            self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
                self.cfg.prompt_processor
            )
            self.prompt_processor_list.append(self.prompt_processor)
        self.prompt_utils_list = []
        for prompt_processor in self.prompt_processor_list:
            prompt_processor = prompt_processor()
            self.prompt_utils_list.append(prompt_processor)

        if self.cfg.use_single_view:
            self.single_view_guidance = threestudio.find(self.cfg.single_view_guidance_type)(self.cfg.single_view_guidance)

            # initialize the single view prompt_processor
            self.single_view_prompt_processor_list = []
            for prompt in self.cfg.prompt_list:
                self.cfg.single_view_prompt_processor.prompt=prompt
                self.single_view_prompt_processor = threestudio.find(self.cfg.single_view_prompt_processor_type)(
                    self.cfg.single_view_prompt_processor
                )
                self.single_view_prompt_processor_list.append(self.single_view_prompt_processor)
            self.single_view_prompt_utils_list = []
            for single_view_prompt_processor in self.single_view_prompt_processor_list:
                single_view_prompt_processor = single_view_prompt_processor()
                self.single_view_prompt_utils_list.append(single_view_prompt_processor)
            
        # initialize the loss weight
        self.loss_weight = self.cfg.loss_weight


    def forward(self, batch: Dict[str, Any],
                gaussian_mean: torch.Tensor = None,
                gaussian_var: torch.Tensor = None,
                bg_color: torch.Tensor = None,
                bound: torch.Tensor = None) -> Dict[str, Any]:
        if self.cfg.renderer_type == "nerf-gaussian-volume-bounded-renderer":
            if bg_color is None:
                return self.renderer(**batch,
                                    gaussian_mean=gaussian_mean,
                                    gaussian_var=gaussian_var)
            return self.renderer(**batch,
                                gaussian_mean=gaussian_mean,
                                gaussian_var=gaussian_var,
                                bg_color=bg_color)
        elif self.cfg.renderer_type == "nerf-multi-volume-bounded-renderer":
            if bg_color is None:
                return self.renderer(**batch,
                                    bound=bound)
            return self.renderer(**batch,
                                bound=bound,
                                bg_color=bg_color)

    def training_step(self, batch, batch_idx):
        if self.use_iterative:
            if self.cfg.renderer_type == "nerf-gaussian-volume-bounded-renderer":
                iter_idx = batch_idx%len(self.weight_field_mean)
                if not iter_idx == len(self.weight_field_mean) - 1:
                    # only learn the layout for the global step
                    weight_mean_list = [self.weight_field_mean[iter_idx].detach()]
                    weight_var_list = [self.weight_field_var[iter_idx].detach()]
                else:
                    weight_mean_list = [self.weight_field_mean[iter_idx]]
                    weight_var_list = [self.weight_field_var[iter_idx]]
            elif self.cfg.renderer_type == "nerf-multi-volume-bounded-renderer":
                iter_idx = batch_idx%len(self.bound)
                bound_list = [self.bound[iter_idx]]

            loss_weight_list = [self.loss_weight[iter_idx]]
            prompt_utils_list = [self.prompt_utils_list[iter_idx]]
            if self.cfg.use_single_view:
                single_view_prompt_utils_list = [self.single_view_prompt_utils_list[iter_idx]]
        if self.cfg.renderer_type == "nerf-gaussian-volume-bounded-renderer":
            out_list = [self(batch, gaussian_mean=weight_mean, gaussian_var=weight_var) \
                        for weight_mean, weight_var in zip(weight_mean_list, weight_var_list)]
        elif self.cfg.renderer_type == "nerf-multi-volume-bounded-renderer":
            out_list = [self(batch, bound=bound) for bound in bound_list]
        if self.cfg.visualize:
            for idx, out in enumerate(out_list):
                rendered_images = out["comp_rgb"]
                rendered_images_to_save = [Image.fromarray((rendered_image_to_save * 255).astype(np.uint8))\
                                            for rendered_image_to_save in rendered_images.cpu().detach().numpy()]
                # save
                for jdx, rendered_image_to_save in enumerate(rendered_images_to_save):
                    rendered_image_to_save.save(f"rendered_images_{idx}_{jdx}.png")
        
        stop = 1
        # crop image legacy code

        # shift the image to the center
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
                for jdx, rendered_image_to_save in enumerate(rendered_images):
                    single_view_opacity_mask = opacity_mask[jdx, :, :, 0]
                    opacity_i, opacity_j = torch.nonzero(single_view_opacity_mask, as_tuple=True)
                    if len(opacity_i) != 0:
                        opacity_j_min, opacity_j_max = opacity_j.min(), opacity_j.max()
                        # single_view_min_x = max(image_h - x_max, opacity_j_min-int(0.1*image_h))
                        # single_view_max_x = min(image_h - x_min, opacity_j_max+int(0.1*image_h))
                        single_view_min_x = opacity_j_min-int(0.1*image_h)
                        single_view_max_x =  opacity_j_max+int(0.1*image_h)
                        single_view_min_y = opacity_i.min()-int(0.1*image_h)
                        single_view_max_y = opacity_i.max()+int(0.1*image_h)
                    else:
                        single_view_min_x, single_view_max_x = image_h-x_max, image_h-x_min
                        single_view_min_y, single_view_max_y = image_h-z_max, image_h-z_min

                    cropped_rendered_image_to_save = \
                        rendered_image_to_save[single_view_min_y:single_view_max_y, single_view_min_x:single_view_max_x]
                    h, w = cropped_rendered_image_to_save.shape[0], cropped_rendered_image_to_save.shape[1]
                    if h==0 or w==0:
                        cropped_rendered_image_to_save = rendered_image_to_save[image_h-z_max:image_h-z_min, image_h-x_max:image_h-x_min]
                    rendered_image_to_save = F.interpolate(cropped_rendered_image_to_save.permute(2, 0, 1).unsqueeze(0),
                                                        (image_h, image_w),
                                                        mode='bilinear',
                                                        align_corners=True)
                    cropped_images.append(rendered_image_to_save.squeeze(0).permute(1, 2, 0))

                    # rendered_image_to_save = cropped_images[-1].cpu().detach().numpy()
                    # rendered_image_to_save = Image.fromarray((rendered_image_to_save * 255).astype(np.uint8))
                    # rendered_image_to_save.save(f"rendered_images_{idx}_{jdx}.png")   
                #covert list to tensor
                cropped_images = torch.stack(cropped_images)
                out_list[idx]["comp_rgb"] = cropped_images
        


        guidance_out_list = [self.guidance(out["comp_rgb"], prompt_utils_list[idx], **batch) \
                             for idx, out in enumerate(out_list)]
        if self.cfg.use_single_view:
            single_view_guidance_out_list = [self.single_view_guidance(out["comp_rgb"],single_view_prompt_utils_list[idx], **batch) \
                                            for idx, out in enumerate(out_list)]
        else:
            single_view_guidance_out_list = [{} for _ in range(len(out_list))]

        loss = 0.0
        subpart_idx = 0
        handle = None
        # iterate the loss of each compositional part
        for out, guidance_out, single_view_guidance_out in zip(out_list, guidance_out_list, single_view_guidance_out_list):
            sub_loss = 0.0
            for name, value in guidance_out.items():
                if name == "handle":
                    handle = value
                else:
                    self.log(f"train/{name}", value)
                    if name.startswith("loss_"):
                        sub_loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])
        
            
            if self.cfg.use_single_view:
                for name, value in single_view_guidance_out.items():
                    self.log(f"train/{name}", value)
                    if name.startswith("loss_"):
                        sub_loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])*5

            if self.C(self.cfg.loss.lambda_orient) > 0:
                if (self.use_iterative and iter_idx == len(self.bound) - 1) or not self.use_iterative:
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

            if iter_idx == len(self.bound) - 1:
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
        self.dataset = self.trainer.train_dataloader.dataset
        update_end_if_possible(
            self.dataset, self.true_current_epoch, self.true_global_step
        )
        self.do_update_step_end(self.true_current_epoch, self.true_global_step)
        if outputs["handle"] is not None:
            outputs['handle'].remove()

    def validation_step(self, batch, batch_idx):
        bg_color = torch.tensor([1.0, 1.0, 1.0]).to(self.device)
        if self.cfg.renderer_type == "nerf-gaussian-volume-bounded-renderer":
            out = self(batch, self.weight_field_mean[-1], self.weight_field_var[-1], bg_color=bg_color)
        elif self.cfg.renderer_type == "nerf-multi-volume-bounded-renderer":
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
        if self.cfg.renderer_type == "nerf-gaussian-volume-bounded-renderer":
            out = self(batch, self.weight_field_mean[-1], self.weight_field_var[-1], bg_color=bg_color)
        elif self.cfg.renderer_type == "nerf-multi-volume-bounded-renderer":
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



@threestudio.register("mvdream-system-bounded")
class MVDreamBoundedSystem(BaseLift3DSystem):
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
