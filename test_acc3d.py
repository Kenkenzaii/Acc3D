import os
import functools
import logging
import math
import shutil
from pathlib import Path
from PIL import Image

import accelerate
import numpy as np
import torch
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
from torchvision.utils import make_grid, save_image
from safetensors.torch import load_file
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DDIMScheduler,
    # UNet2DConditionModel,
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
from utils_tool import *
from mvdiffusion.data.single_image_dataset import SingleImageDatasetGenerator

from mvdiffusion.models.unet_mv2d_condition import UNetMV2DConditionModel
from train_acc3d import (
    TestConfig,
)
from peft import PeftModel, AutoPeftModel
from rembg import remove


def convert_to_numpy(tensor):
    return (
        tensor.mul(255)
        .add_(0.5)
        .clamp_(0, 255)
        .permute(1, 2, 0)
        .to("cpu", torch.uint8)
        .numpy()
    )


def save_image(tensor, fp):
    ndarr = convert_to_numpy(tensor)
    save_image_numpy(ndarr, fp)
    return ndarr


def save_image_numpy(ndarr, fp):
    im = Image.fromarray(ndarr)
    im.save(fp)


def get_module_state_dict(
    module, prefix: str, dtype: torch.dtype, adapter_name: str = "default"
):
    kohya_ss_state_dict = {}
    for peft_key, weight in get_peft_model_state_dict(
        module, adapter_name=adapter_name
    ).items():
        kohya_key = peft_key.replace("base_model.model.", prefix)
        kohya_ss_state_dict[kohya_key] = weight.to(dtype)

    return kohya_ss_state_dict


def create_lora_unet(path, weight, pipeline):
    lora_config = LoraConfig(
        r=64,
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
    unet = get_peft_model(pipeline.unet, lora_config)
    unet.load_adapter(path, weight, "default")
    return unet


def main(args):
    save_path = "./Results/formal_2k"
    os.makedirs(save_path, exist_ok=True)

    val_dataset = SingleImageDatasetGenerator(
        prompt_embeds_path="mvdiffusion/data/fixed_prompt_embeds_6view",
        json_path="examples.json",
        root_dir="examples",
        num_views=6,
        bg_color="white",
        img_wh=[512, 512],
        crop_size=350,
    )

    dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
    )

    pipeline = StableUnCLIPImg2ImgPipeline.from_pretrained(
        args.pretrained_teacher_model
    )
 

    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config
    )

    model = PeftModel.from_pretrained(
        pipeline.unet, "kd5678/acc3d_model"
    ) 

    pipeline.unet = model

    pipeline = pipeline.to(device="cuda", dtype=torch.float16)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()
    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device="cuda").manual_seed(args.seed)

    images_cond = []
    for step, batch in tqdm(enumerate(dataloader)):
        with torch.no_grad():

            scene = batch["filename"][0].split(".")[0]
            scene_dir = os.path.join(save_path, scene)

            images_cond.append(batch["imgs_in"][:, 0])
            imgs_in = torch.cat([batch["imgs_in"]] * 2, dim=0)
            num_views = imgs_in.shape[1]
            imgs_in = rearrange(
                imgs_in, "B Nv C H W -> (B Nv) C H W"
            )  # (B*Nv, 3, H, W)

            normal_prompt_embeddings, clr_prompt_embeddings = (
                batch["normal_prompt_embeddings"],
                batch["color_prompt_embeddings"],
            )
            prompt_embeddings = torch.cat(
                [normal_prompt_embeddings, clr_prompt_embeddings], dim=0
            )
            prompt_embeddings = rearrange(prompt_embeddings, "B Nv N C -> (B Nv) N C")
            with torch.no_grad():
                with torch.autocast("cuda"):

                    unet_out = pipeline(
                        imgs_in,
                        None,
                        prompt_embeds=prompt_embeddings,
                        generator=generator,
                        guidance_scale=1.0,
                        num_inference_steps=3,
                        output_type="pt",
                        num_images_per_prompt=1,
                        eta=1.0,
                        return_dict=False,
                    )

                    images, latents = unet_out[0], unet_out[1]

                    out = images
                    bsz = out.shape[0] // 2

                    normals_pred = out[:bsz]
                    images_pred = out[bsz:]
                    vis_ = []
                    if cfg.save_mode == "concat":

                        for i in range(bsz // num_views):
                            scene = batch["filename"][i].split(".")[0]
                            img_in_ = images_cond[-1][i].to(out.device)
                            vis_ = []
                            for j in range(num_views):
                                view = VIEWS[j]
                                idx = i * num_views + j
                                normal = normals_pred[idx]
                                color = images_pred[idx]
                                save_image(
                                    color, os.path.join(save_path, f"color_{view}.png")
                                )
                                save_image(
                                    normal,
                                    os.path.join(save_path, f"normal_{view}.png"),
                                )

                                vis_.append(color)
                                vis_.append(normal)

                            vis_ = torch.stack(vis_, dim=0)
                            vis_ = make_grid(
                                vis_, nrow=len(vis_) // 2, padding=0, value_range=(0, 1)
                            )
                            save_image(vis_, os.path.join(save_path, scene + ".png"))

                    elif cfg.save_mode == "rgba":

                        for i in range(bsz // num_views):
                            scene = batch["filename"][i].split(".")[0]
                            scene_dir = os.path.join(save_path, scene)
                            os.makedirs(scene_dir, exist_ok=True)

                            img_in_ = images_cond[-1][i].to(out.device)
                            vis_ = [img_in_]
                            for j in range(num_views):
                                view = VIEWS[j]
                                idx = i * num_views + j
                                normal = normals_pred[idx]
                                color = images_pred[idx]
                                vis_.append(color)
                                vis_.append(normal)

                                normal = convert_to_numpy(normal)
                                color = convert_to_numpy(color)
                                rm_normal = remove(normal)
                                rm_color = remove(color)
                                normal_filename = f"normals_{view}_masked.png"
                                rgb_filename = f"color_{view}_masked.png"
                                save_image_numpy(
                                    rm_normal, os.path.join(scene_dir, normal_filename)
                                )
                                save_image_numpy(
                                    rm_color, os.path.join(scene_dir, rgb_filename)
                                )

                torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf
    from utils.misc import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", type=str)
    args, extras = parser.parse_known_args()

    # parse YAML config to OmegaConf
    cfg = load_config(args.config, cli_args=extras)
    schema = OmegaConf.structured(TestConfig)
    cfg = OmegaConf.merge(schema, cfg)

    VIEWS = ["front", "front_right", "right", "back", "left", "front_left"]

    main(cfg)
