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

torch.set_float32_matmul_precision('medium')


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
        self.global_mean_shift = 0
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
        self.optimizer = torch.optim.Adam([self.weight_field_mean], lr=1e-5)

        # set up the weighting field function


        
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


    def render_gaussian(self,
                        batch: Dict[str, Any],
                        means: torch.Tensor, # N, 3 
                        vars: torch.Tensor, # N, 3
                        ) -> Dict[str, Any]:
        
        c2w = batch["c2w"] # B, 4, 4
        fovy = batch["fovy"] # B
        H, W = batch["height"], batch["width"] # B
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

        projected_vars = vars[:, None, :] / torch.abs(camera_means[:, :, 2]).unsqueeze(-1) # N, 1, 3 / N, B, 3 --> N, B, 3
        projected_vars = projected_vars * torch.sqrt(torch.tensor(W**2 + H**2)).to(projected_vars.device)


        return projected_means, projected_vars
        


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
            if self.cfg.use_single_view:
                single_view_prompt_utils_list = [self.single_view_prompt_utils_list[iter_idx]]
        

        if is_global_step:
            projected_means, projected_vars = self.render_gaussian(batch,
                                                                    self.weight_field_mean,
                                                                    self.weight_field_var) # N, B, 2
            def gaussian_filter(H, W, mean, var):
                var = var * 2 #Hardcoded
                x = torch.arange(0, W).to(mean)
                y = torch.arange(0, H).to(mean)
                x, y = torch.meshgrid(x, y)
                x = x - mean[0]
                y = y - mean[1]
                weight = torch.exp(-((x**2)/(2*torch.max(var)**2) + (y**2)/(2*torch.max(var)**2)))
                return weight
            weight_filters = []
            for idx, (projected_mean, projected_var) in enumerate(zip(projected_means, projected_vars)):
                weight_filters_for_one_bound = []
                for jdx, (mean, var) in enumerate(zip(projected_mean, projected_var)):
                    weight = gaussian_filter(batch['height'], batch['width'], mean, var)
                    weight_filters_for_one_bound.append(weight)
                weight_filters_for_one_bound = torch.stack(weight_filters_for_one_bound) # B, H, W
                weight_filters.append(weight_filters_for_one_bound)
            weight_filters = torch.stack(weight_filters) # N, B, H, W
            global_weight_filters = torch.sum(weight_filters, dim=0)/len(self.bound) # B, H, W

        # conduct rendering for each bounding box ========================================
        out_list = [self(batch, bound=bound) for bound in bound_list]


        if self.cfg.visualize:
            if is_global_step:
                for idx, (projected_mean, projected_var) in enumerate(zip(projected_means, projected_vars)):
                    for jdx, (mean, var) in enumerate(zip(projected_mean, projected_var)):
                        mean = mean.cpu().detach().numpy().astype(np.int32)
                        var = var.cpu().detach().numpy().astype(np.int32)
                        image = np.zeros((batch['height'], batch['width'], 3), dtype=np.uint8)
                        image = cv2.circle(image, (mean[0], mean[1]), 5, (255, 0, 0), -1)
                        image = cv2.ellipse(image, (mean[0], mean[1]), (var[0], var[1]), 0, 0, 360, (0, 255, 0), 2)
                        cv2.imwrite(f"gaussian_{idx}_{jdx}.png", image)
                        weight_filter = weight_filters[idx, jdx].cpu().detach().numpy()
                        cv2.imwrite(f"weight_filter_{idx}_{jdx}.png", weight_filter*255)

                for jdx, weight_filter in enumerate(global_weight_filters):
                    cv2.imwrite(f"global_weight_filter_{jdx}.png", weight_filter.cpu().detach().numpy()*255)

            for idx, out in enumerate(out_list):
                rendered_images = out["comp_rgb"]
                rendered_images_to_save = [Image.fromarray((rendered_image_to_save * 255).astype(np.uint8))\
                                            for rendered_image_to_save in rendered_images.cpu().detach().numpy()]
                # save
                for jdx, rendered_image_to_save in enumerate(rendered_images_to_save):
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

                    cropped_rendered_image = \
                        rendered_image[single_view_min_y:single_view_max_y, single_view_min_x:single_view_max_x]
          
                    h, w = cropped_rendered_image.shape[0], cropped_rendered_image.shape[1]
                    if is_global_step:
                        global_weight_filter = global_weight_filter[single_view_min_y:single_view_max_y, single_view_min_x:single_view_max_x]
                    if h==0 or w==0:
                        cropped_rendered_image = rendered_image_to_save[image_h-z_max:image_h-z_min, image_h-x_max:image_h-x_min]
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
            guidance_out_list = [self.guidance(out["comp_rgb"],prompt_utils_list[idx], **batch, weight_filters=cropped_weight_filters) \
                                for idx, out in enumerate(out_list)]
        else:
            guidance_out_list = [self.guidance(out["comp_rgb"],prompt_utils_list[idx], **batch) \
                                for idx, out in enumerate(out_list)]
        if self.cfg.use_single_view:
            single_view_guidance_out_list = [self.single_view_guidance(out["comp_rgb"],single_view_prompt_utils_list[idx], **batch) \
                                            for idx, out in enumerate(out_list)]
        else:
            single_view_guidance_out_list = [{} for _ in range(len(out_list))]

        # iterate the loss of each compositional part ========================================
        loss = 0.0
        subpart_idx = 0
        handle = None
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
        if self.weight_field_mean.grad is not None and batch_idx > 2000 and is_global_step:
            self.optimizer.step()
            self.optimizer.zero_grad()
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
