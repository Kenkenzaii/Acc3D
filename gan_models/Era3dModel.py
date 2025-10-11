import functools
import logging
import math
import os
import shutil
from pathlib import Path
from PIL import Image

import accelerate
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel, PretrainedConfig
from dataclasses import dataclass
from typing import Dict, Optional, List
from einops import rearrange, repeat
import torchvision.transforms.functional as TF
from diffusers.utils.torch_utils import randn_tensor
from diffusers.models.embeddings import get_timestep_embedding
from peft import PeftModel

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DDIMScheduler,
)
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available

from mvdiffusion.pipelines.pipeline_mvdiffusion_unclip import (
    StableUnCLIPImg2ImgPipeline,
)
from mvdiffusion.schedulers.scheduling_dpmsolver_multistep import (
    DPMSolverMultistepScheduler,
)
from mvdiffusion.models.unet_mv2d_condition import UNetMV2DConditionModel


class BaseModel(nn.Module):
    def __init__(self, args, accelerator):
        super().__init__()
        self.args = args
        self.generator = torch.Generator(device=accelerator.device).manual_seed(
            args.seed
        )
        self.extra_step_kwargs = {"generator": self.generator}
        self.pipeline = StableUnCLIPImg2ImgPipeline.from_pretrained(
            args.pretrained_teacher_model
        )
        self.pipeline = self.pipeline.to(accelerator.device)
        self.vae = self.pipeline.vae
        self.text_encoder = self.pipeline.text_encoder
        self.image_encoder = self.pipeline.image_encoder
        self.feature_extractor = self.pipeline.feature_extractor
        self.image_normalizer = self.pipeline.image_normalizer
        self.image_noising_scheduler = self.pipeline.image_noising_scheduler
        self.image_processor = self.pipeline.image_processor
        self.scheduler = self.pipeline.scheduler

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        if args.distill_checkpoint_path == "":
            print("Initialize lora model...")
            self.student_unet = UNetMV2DConditionModel.from_pretrained(
                args.pretrained_teacher_model, subfolder="unet"
            )
            lora_config = LoraConfig(
                r=args.lora_rank,  # lora rank : 64
                target_modules=[
                    "to_q",
                    "to_k",
                    "to_v",
                    "to_out.0",
                    "proj_in",
                    "proj_out",
                    "ff.net.0.proj",
                    "ff.net.2",
                    "conv1",
                    "conv2",
                    "conv_shortcut",
                    "downsamplers.0.conv",
                    "upsamplers.0.conv",
                    "time_emb_proj",
                ],
            )
            self.student_unet = get_peft_model(self.student_unet, lora_config)
            self.student_unet.print_trainable_parameters()
        else:
            model = PeftModel.from_pretrained(
                self.pipeline.unet, args.distill_checkpoint_path, is_trainable=True
            )
            self.student_unet = model  # self.pipeline.unet
        if accelerator.unwrap_model(self.student_unet).dtype != torch.float32:
            low_precision_error_string = (
                " Please make sure to always have all model weights in full float32 precision when starting training - even if"
                " doing mixed precision training, copy of the weights should still be float32."
            )
            raise ValueError(
                f"Controlnet loaded as datatype {accelerator.unwrap_model(self.unet).dtype}. {low_precision_error_string}"
            )

        self.weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        self.vae.to(accelerator.device)
        self.text_encoder.to(accelerator.device, dtype=self.weight_dtype)
        self.image_encoder.to(accelerator.device, dtype=self.weight_dtype)
        if self.student_unet.device != accelerator.device:
            self.student_unet.to(accelerator.device)
        # Teacher net can be fp16 to save space and speed up.
        # self.student_unet.to(accelerator.device)
        # self.student_unet.train()

    # Prepare text prompt embedding for teacher net (Optional CFG)
    def prepare_text_embedding(self, batch, device, CFG_GUIDANCE=True):
        normal_text_embeds, color_text_embeds = (
            batch["normal_prompt_embeddings"],
            batch["color_prompt_embeddings"],
        )
        prompt_embeds = torch.cat([normal_text_embeds, color_text_embeds], dim=0)
        prompt_embeds = rearrange(prompt_embeds, "B Nv N C -> (B Nv) N C")
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
        prompt_embeds_single = prompt_embeds
        if CFG_GUIDANCE:
            normal_prompt_embeds, color_prompt_embeds = torch.chunk(
                prompt_embeds, 2, dim=0
            )
            prompt_embeds = torch.cat(
                [
                    normal_prompt_embeds,
                    normal_prompt_embeds,
                    color_prompt_embeds,
                    color_prompt_embeds,
                ],
                0,
            )
        return prompt_embeds, prompt_embeds_single

    def encode_image(
        self,
        dtype,
        image_pil,
        device,
        num_images_per_prompt,
        CFG_GUIDANCE: bool = True,
        noise_level: int = 0,
        real: bool = False,
    ):
        image = self.feature_extractor(
            images=image_pil, return_tensors="pt"
        ).pixel_values
        image = image.to(device=device, dtype=dtype)
        image_embeds = self.image_encoder(image).image_embeds

        noise_level = torch.tensor([noise_level], device=device)
        image_embeds = self.noise_image_embeddings(
            image_embeds=image_embeds, noise_level=noise_level
        )
        image_embeds = image_embeds.repeat(num_images_per_prompt, 1)

        # CFG_GUIDANCE = True #guidance_scale > 1.0

        if CFG_GUIDANCE:
            image_embeds_single = image_embeds
            normal_image_embeds, color_image_embeds = torch.chunk(
                image_embeds, 2, dim=0
            )
            negative_prompt_embeds = torch.zeros_like(normal_image_embeds)
            image_embeds = torch.cat(
                [
                    negative_prompt_embeds,
                    normal_image_embeds,
                    negative_prompt_embeds,
                    color_image_embeds,
                ],
                0,
            )

        image_pt = torch.stack([TF.to_tensor(img) for img in image_pil], dim=0).to(
            dtype=self.vae.dtype, device=device
        )  # vae dtype
        image_pt = image_pt * 2.0 - 1.0

        image_latents = (
            self.vae.encode(image_pt).latent_dist.mode()
            * self.vae.config.scaling_factor
        )

        image_latents = image_latents.repeat(num_images_per_prompt, 1, 1, 1)
        # do_classifier_free_guidance
        if CFG_GUIDANCE:
            image_latents_single = image_latents
            normal_image_latents, color_image_latents = torch.chunk(
                image_latents, 2, dim=0
            )
            image_latents = torch.cat(
                [
                    torch.zeros_like(normal_image_latents),
                    normal_image_latents,
                    torch.zeros_like(color_image_latents),
                    color_image_latents,
                ],
                0,
            )
        return image_embeds, image_latents, image_embeds_single, image_latents_single

    def noise_image_embeddings(
        self,
        image_embeds: torch.Tensor,
        noise_level: int,
        generator: Optional[torch.Generator] = None,
        image_normalizer=None,
        image_noising_scheduler=None,
    ):
        noise = randn_tensor(
            image_embeds.shape,
            generator=generator,
            device=image_embeds.device,
            dtype=image_embeds.dtype,
        )
        noise_level = torch.tensor(
            [noise_level] * image_embeds.shape[0], device=image_embeds.device
        )
        image_embeds = self.image_normalizer.scale(image_embeds)
        image_embeds = self.image_noising_scheduler.add_noise(
            image_embeds, timesteps=noise_level, noise=noise
        )
        image_embeds = self.image_normalizer.unscale(image_embeds)
        noise_level = get_timestep_embedding(
            timesteps=noise_level,
            embedding_dim=image_embeds.shape[-1],
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        noise_level = noise_level.to(image_embeds.dtype)
        image_embeds = torch.cat((image_embeds, noise_level), 1)
        return image_embeds


class DiscriminatorMultiFeature(nn.Module):
    def __init__(self, args, accelerator):
        super().__init__()
        self.args = args
        self.conv1 = nn.Conv2d(1280, 640, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(320, 640, kernel_size=1, stride=1, padding=0)
        # feature_transformed = conv(feature)

        self.cls_pred = nn.Sequential(
            nn.Conv2d(
                kernel_size=4, in_channels=1920, out_channels=1920, stride=2, padding=1
            ),  # 32x32 -> 16x16
            nn.GroupNorm(num_groups=32, num_channels=1920),
            nn.SiLU(),
            nn.Conv2d(
                kernel_size=4, in_channels=1920, out_channels=1920, stride=2, padding=1
            ),  # 16x16 -> 8x8
            nn.GroupNorm(num_groups=32, num_channels=1920),
            nn.SiLU(),
            nn.Conv2d(
                kernel_size=4, in_channels=1920, out_channels=1920, stride=2, padding=1
            ),  # 8x8 -> 4x4
            nn.GroupNorm(num_groups=32, num_channels=1920),
            nn.SiLU(),
            nn.Conv2d(
                kernel_size=4, in_channels=1920, out_channels=1920, stride=4, padding=0
            ),  # 4x4 -> 1x1
            nn.GroupNorm(num_groups=32, num_channels=1920),
            nn.SiLU(),
            nn.Conv2d(
                kernel_size=1, in_channels=1920, out_channels=1, stride=1, padding=0
            ),  # 1x1 -> 1x1
        )
        self.cls_pred.requires_grad_(True)
        self.cls_pred.to(accelerator.device)

    def forward(self, features):
        fea_0 = features[0]
        fea_0 = self.conv1(fea_0)
        fea_1 = F.avg_pool2d(features[1], kernel_size=2, stride=2)
        fea_2 = self.conv2(features[2])
        fea_2 = F.avg_pool2d(fea_2, kernel_size=2, stride=2)
        x = torch.cat([fea_0, fea_1, fea_2], dim=1)
        cls = self.cls_pred(x).squeeze(dim=[2, 3])
        return cls


class Model(nn.Module):
    def __init__(self, features=64):
        super().__init__()
        self.cls_pred_deprecated = nn.Sequential(
            nn.Conv2d(
                kernel_size=4, in_channels=1280, out_channels=1280, stride=2, padding=1
            ),  # 32x32 -> 16x16
            nn.GroupNorm(num_groups=32, num_channels=1280),
            nn.SiLU(),
            nn.Conv2d(
                kernel_size=4, in_channels=1280, out_channels=1280, stride=2, padding=1
            ),  # 16x16 -> 8x8
            nn.GroupNorm(num_groups=32, num_channels=1280),
            nn.SiLU(),
            nn.Conv2d(
                kernel_size=4, in_channels=1280, out_channels=1280, stride=2, padding=1
            ),  # 8x8 -> 4x4
            nn.GroupNorm(num_groups=32, num_channels=1280),
            nn.SiLU(),
            nn.Conv2d(
                kernel_size=4, in_channels=1280, out_channels=1280, stride=4, padding=0
            ),  # 4x4 -> 1x1
            nn.GroupNorm(num_groups=32, num_channels=1280),
            nn.SiLU(),
            nn.Conv2d(
                kernel_size=1, in_channels=1280, out_channels=1, stride=1, padding=0
            ),  # 1x1 -> 1x1
        )
        self.cls_pred = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=4,
                out_channels=features,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1),
                bias=False,
            ),
            torch.nn.BatchNorm2d(num_features=features),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            torch.nn.Conv2d(
                in_channels=features,
                out_channels=features * 2,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1),
                bias=False,
            ),
            torch.nn.BatchNorm2d(num_features=features * 2),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            torch.nn.Conv2d(
                in_channels=features * 2,
                out_channels=features * 4,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1),
                bias=False,
            ),
            torch.nn.BatchNorm2d(num_features=features * 4),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            torch.nn.Conv2d(
                in_channels=features * 4,
                out_channels=features * 8,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1),
                bias=False,
            ),
            torch.nn.BatchNorm2d(num_features=features * 8),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),
            torch.nn.Conv2d(
                in_channels=features * 8,
                out_channels=1,
                kernel_size=(4, 4),
                stride=(1, 1),
                padding=(0, 0),
                bias=False,
            ),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.cls_pred_deprecated(x).squeeze(dim=[2, 3])
        return x


if __name__ == "__main__":
    x = torch.randn((24, 1280, 32, 32))
    model = Model()
    res = model(x)
    print(res.shape)
