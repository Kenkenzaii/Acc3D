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
from accelerate.utils import DistributedDataParallelKwargs

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
from mvdiffusion.schedulers.scheduling_dpmsolver_multistep_pred_alpha import (
    DPMSolverMultistepScheduler,
)
from utils_tool import *
from mvdiffusion.data.single_image_dataset import (
    SingleImageDataset,
    SingleImageDatasetGAN,
    SingleImageDatasetGAN_Normal,
)
from mvdiffusion.models.unet_mv2d_condition import UNetMV2DConditionModel

from gan_models.Era3dModel import (
    BaseModel,
    DiscriminatorMultiFeature,
)


MAX_SEQ_LENGTH = 77
logger = get_logger(__name__)


@dataclass
class TestConfig:
    # -------------------Distillation -------------------------
    pretrained_teacher_model: str
    output_dir: str
    tracker_project_name: str
    mixed_precision: str
    resolution: int
    lora_rank: int
    learning_rate: float
    loss_type: str
    adam_weight_decay: float
    max_train_steps: int
    max_train_samples: int
    dataloader_num_workers: int
    validation_steps: int
    checkpointing_steps: int
    checkpoints_total_limit: int
    train_batch_size: int
    enable_xformers_memory_efficient_attention: bool
    gradient_accumulation_steps: int
    use_8bit_adam: bool
    report_to: str
    resume_from_checkpoint: Optional[str]
    seed: int
    num_ddim_timesteps: int
    gradient_checkpointing: bool
    logging_dir: str
    cast_teacher_unet: bool
    allow_tf32: bool
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    proportion_empty_prompts: float
    lr_scheduler: str
    lr_warmup_steps: int
    num_train_epochs: int
    not_apply_cfg_solver: bool
    huber_c: float
    max_grad_norm: float
    distill_checkpoint_path: str

    # --------------------- Era3d ------------------------
    revision: Optional[str]
    train_dataset: Dict
    save_dir: str
    seed: Optional[int]
    dataloader_num_workers: int

    save_mode: str
    local_rank: int

    pipe_kwargs: Dict
    pipe_validation_kwargs: Dict
    unet_from_pretrained_kwargs: Dict
    validation_guidance_scales: List[float]
    validation_grid_nrow: int

    num_views: int
    pred_type: str
    regress_elevation: bool
    enable_xformers_memory_efficient_attention: bool

    regress_elevation: bool
    regress_focal_length: bool

    discriminator_gradient_freq: int
    use_real_latents: bool
    noise_start_steptime_distill: int
    noise_start_steptime_consistency: int
    interval_num: int


def random_select_input(view, latents):

    latents_normal, latents_color = torch.chunk(latents, 2, dim=0)
    latents_normal = torch.roll(latents_normal, -view, dims=0)
    latents_color = torch.roll(latents_color, -view, dims=0)

    latents = torch.cat([latents_normal, latents_color], 0)

    return latents


def get_x0_from_noise_dpmsolver(sample, model_output, sigma_s):

    alpha_s = torch.sqrt(1 - sigma_s**2)
    sigma_t = 1e-3
    alpha_t = torch.sqrt(1 - sigma_t**2)
    lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
    lambda_s = torch.log(alpha_s) - torch.log(sigma_s)
    h = lambda_t - lambda_s
    x_t = (alpha_t / alpha_s) * sample - (sigma_t * (torch.exp(h) - 1.0)) * model_output
    return x_t


def split_color_normal(tensor_list):
    normal = []
    color = []

    for tensor in tensor_list:
        split1, split2 = torch.split(tensor, tensor.size(0) // 2, dim=0)
        normal.append(split1)
        color.append(split2)

    return normal, color


def reweight(start_steptime, weight):
    from_min = start_steptime
    from_range = 1000 - from_min
    to_range = 1 - 0.2
    scaled_value = (((weight - from_min) * to_range) / from_range) + 0.2
    return scaled_value


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        # this should be with the device id?
        set_seed(args.seed + accelerator.process_index)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    generator_seed = torch.Generator(device=accelerator.device).manual_seed(args.seed)
    extra_step_kwargs = {
        "generator": generator_seed
    }  

    unet_freeze = UNetMV2DConditionModel.from_pretrained(
        args.pretrained_teacher_model, subfolder="unet"
    )
    unet_freeze = PeftModel.from_pretrained(unet_freeze, args.distill_checkpoint_path)
    unet_freeze.requires_grad_(False)

    generator = BaseModel(args, accelerator)

    discriminator_color = DiscriminatorMultiFeature(args, accelerator)
    discriminator_normal = DiscriminatorMultiFeature(args, accelerator)

    discriminator_color.to(accelerator.device)
    discriminator_normal.to(accelerator.device)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    student_dpm_scheduler = DPMSolverMultistepScheduler.from_config(
        generator.scheduler.config
    )
    dpm_scheduler = DPMSolverMultistepScheduler.from_config(generator.scheduler.config)
    noise_scheduler = DPMSolverMultistepScheduler.from_config(
        generator.scheduler.config
    )

    criterion = torch.nn.BCELoss()
    criterion.to(accelerator.device, dtype=weight_dtype)

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                unet_ = accelerator.unwrap_model(generator.student_unet)  # unet
                lora_state_dict = get_peft_model_state_dict(
                    unet_, adapter_name="default"
                )
                StableUnCLIPImg2ImgPipeline.save_lora_weights(
                    os.path.join(output_dir, "unet_lora"), lora_state_dict
                )
                unet_.save_pretrained(output_dir)

                for _, model in enumerate(models):
                    weights.pop()

        def load_model_hook(models, input_dir):
            unet_ = accelerator.unwrap_model(generator.student_unet)  # unet
            unet_.load_adapter(input_dir, "default", is_trainable=True)
            for _ in range(len(models)):
                models.pop()

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            # discriminator.enable_xformers_memory_efficient_attention()
            generator.student_unet.enable_xformers_memory_efficient_attention()
            unet_freeze.enable_xformers_memory_efficient_attention()
            # teacher_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    if args.gradient_checkpointing:
        generator.student_unet.enable_gradient_checkpointing()

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    optimizer_G = optimizer_class(
        generator.student_unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    optimizer_D_color = optimizer_class(
        discriminator_color.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    optimizer_D_normal = optimizer_class(
        discriminator_normal.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = SingleImageDatasetGAN_Normal(**args.train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler_G = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_G,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    lr_scheduler_D_color = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_D_color,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    lr_scheduler_D_normal = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_D_normal,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # generator.student_unet.enable_adapter_layers()
    # generator.student_unet.print_trainable_parameters()

    (
        generator.student_unet,
        discriminator_color,
        discriminator_normal,
        optimizer_G,
        optimizer_D_color,
        optimizer_D_normal,
        lr_scheduler_G,
        lr_scheduler_D_color,
        lr_scheduler_D_normal,
        train_dataloader,
    ) = accelerator.prepare(
        generator.student_unet,
        discriminator_color,
        discriminator_normal,
        optimizer_G,
        optimizer_D_color,
        optimizer_D_normal,
        lr_scheduler_G,
        lr_scheduler_D_color,
        lr_scheduler_D_normal,
        train_dataloader,
    )

    generator.student_unet.train()

    # total_params = sum(p.numel() for p in generator.student_unet.parameters())
    # trainable_params = sum(p.numel() for p in generator.student_unet.parameters() if p.requires_grad)
    # print('Total Parameters:', total_params)
    # print('Trainable Parameters:', trainable_params)

    unet_freeze.to(device=accelerator.device, dtype=weight_dtype)
    unet_freeze.requires_grad_(False)

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        tracker_config = dict(args)
        print("config file: ", tracker_config)
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )
    logger.info("***** Running training *****")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar_total = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    update_discriminator = False
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(discriminator_color), accelerator.accumulate(
                discriminator_normal
            ), accelerator.accumulate(generator.student_unet):

                ## 1. Prepare  images and real images latents
                imgs_in = torch.cat([batch["imgs_in"]] * 2, dim=0)
                imgs_in = rearrange(imgs_in, "B Nv C H W -> (B Nv) C H W")

                real_imgs_in = batch["real_imgs_in"]
                real_imgs_in = rearrange(real_imgs_in, "B Nv C H W -> (B Nv) C H W")

                if generator.vae.dtype != weight_dtype:
                    generator.vae.to(dtype=weight_dtype)
                _, prompt_embeds = generator.prepare_text_embedding(
                    batch=batch, device=accelerator.device
                )

                image_pil = [
                    TF.to_pil_image(imgs_in[i]) for i in range(imgs_in.shape[0])
                ]
                real_image_pil = [
                    TF.to_pil_image(real_imgs_in[i])
                    for i in range(real_imgs_in.shape[0])
                ]

                _, _, image_embeds, image_latents = generator.encode_image(
                    dtype=weight_dtype,
                    image_pil=image_pil,
                    device=accelerator.device,
                    num_images_per_prompt=1,
                )

                _, _, real_image_embeds, real_image_latents = generator.encode_image(
                    dtype=weight_dtype,
                    image_pil=real_image_pil,
                    device=accelerator.device,
                    num_images_per_prompt=1,
                    real=True,
                )

                latents_student = torch.randn_like(image_latents)  # [12, 4, 64, 64]
                noise = latents_student

                bsz = args.train_batch_size  # image_latents.shape[0] # B * Nv
                noisy_latents_student = [latents_student]
                student_dpm_scheduler.set_timesteps(
                    num_inference_steps=1, device=accelerator.device
                )
                t_s_list = [[student_dpm_scheduler.timesteps[0].cpu()]]

                cleaner_start_timestep = torch.randint(
                    args.noise_start_steptime_consistency,
                    noise_scheduler.config.num_train_timesteps - 5,
                    (1,),
                    device=image_latents.device,
                ).long()
                noisier_start_timestep = (
                    cleaner_start_timestep
                    - torch.randint(
                        1, args.interval_num, (1,), device=image_latents.device
                    ).long()
                )

                latents_noisy_cleaner, alpha_t_cleaner = noise_scheduler.add_noise(
                    real_image_latents, noise, cleaner_start_timestep
                )
                noisy_latents_student.append(latents_noisy_cleaner)
                t_s_list.append([cleaner_start_timestep.cpu()[0]])
                latents_noisy_noisier, alpha_t_noiser = noise_scheduler.add_noise(
                    real_image_latents, noise, noisier_start_timestep
                )

                noisy_latents_student.append(latents_noisy_noisier)
                t_s_list.append([noisier_start_timestep.cpu()[0]])

                x0_preds = []

                for i, latents_student_noisy in enumerate(noisy_latents_student):
                    student_dpm_scheduler.set_timesteps(
                        device=accelerator.device, timesteps=t_s_list[i]
                    )
                    step_num = 1
                    for s in range(step_num):
                        latent_model_input = latents_student_noisy
                        latent_model_input = torch.cat(
                            [latent_model_input, image_latents], dim=1
                        )
                        latent_model_input = student_dpm_scheduler.scale_model_input(
                            latent_model_input, t_s_list[i][s]
                        )
                        unet_out = generator.student_unet(
                            latent_model_input,
                            t_s_list[i][s],
                            encoder_hidden_states=prompt_embeds,
                            class_labels=image_embeds,
                            return_dict=False,
                        )
                        noise_pred, student_features_with_grad = (
                            unet_out[0],
                            unet_out[2],
                        )
                        latents_student_noisy, x0 = student_dpm_scheduler.step(
                            noise_pred,
                            t_s_list[i][s],
                            latents_student_noisy,
                            **extra_step_kwargs,
                            return_dict=False,
                        )  # **extra_step_kwargs
                    if i == 0:
                        latents_student = latents_student_noisy
                        latents_student_vis = latents_student
                        output_x0 = x0
                    else:
                        x0_preds.append(x0)

                start_timesteps = torch.randint(
                    args.noise_start_steptime_distill,
                    noise_scheduler.config.num_train_timesteps - 5,
                    (bsz,),
                    device=image_latents.device,
                ).long()
                noisy_estimated_x0, _ = noise_scheduler.add_noise(
                    output_x0, noise, start_timesteps
                )  # real_image_latents estimated_x0

                with torch.no_grad():
                    student_dpm_scheduler.set_timesteps(
                        device=accelerator.device, timesteps=start_timesteps.cpu()
                    )
                    step_num = 1
                    for s in range(step_num):
                        latent_model_input = noisy_estimated_x0
                        latent_model_input = torch.cat(
                            [latent_model_input, image_latents], dim=1
                        )
                        latent_model_input = student_dpm_scheduler.scale_model_input(
                            latent_model_input, start_timesteps.cpu()
                        )
                        unet_out = generator.student_unet(
                            latent_model_input,
                            start_timesteps.cpu(),
                            encoder_hidden_states=prompt_embeds,
                            class_labels=image_embeds,
                            return_dict=False,
                        )
                        noise_pred = unet_out[0]
                        latents_student_noisy, x0_ = student_dpm_scheduler.step(
                            noise_pred,
                            start_timesteps.cpu(),
                            latents_student_noisy,
                            **extra_step_kwargs,
                            return_dict=False,
                        )  # **extra_step_kwargs
                    x0_prime = x0_.detach()

                noise = torch.randn_like(latents_student)

                start_timesteps = torch.randint(
                    950,
                    noise_scheduler.config.num_train_timesteps - 10,
                    (bsz,),
                    device=latents_student.device,
                ).long()

                fake_noisy_model_input_grad, _ = noise_scheduler.add_noise(
                    latents_student, noise, start_timesteps
                )

                t_freeze = start_timesteps  # student_dpm_scheduler.timesteps[0]
                # student_dpm_scheduler.set_timesteps(device=accelerator.device, timesteps=t_freeze)
                latent_model_input = fake_noisy_model_input_grad
                latent_model_input = torch.cat(
                    [latent_model_input, image_latents], dim=1
                )
                latent_model_input = student_dpm_scheduler.scale_model_input(
                    latent_model_input, t_freeze
                )
                unet_freeze.requires_grad_(False)

                fake_output_grad = unet_freeze(
                    latent_model_input.to(weight_dtype),
                    t_freeze,
                    encoder_hidden_states=prompt_embeds.to(weight_dtype),
                    class_labels=image_embeds.to(weight_dtype),
                    return_dict=False,
                )
                fake_features_grad = fake_output_grad[2]

                fake_noisy_model_input, _ = noise_scheduler.add_noise(
                    latents_student.detach(), noise, start_timesteps
                )
                real_noisy_model_input, _ = noise_scheduler.add_noise(
                    real_image_latents, noise, start_timesteps
                )
                with torch.no_grad():
                    with torch.autocast("cuda"):
                        unet_freeze.requires_grad_(False)

                        ## 3. Disabled unet lora, extract mv features
                        t_freeze = start_timesteps  # student_dpm_scheduler.timesteps[0]
                        latent_model_input = fake_noisy_model_input
                        latent_model_input = torch.cat(
                            [latent_model_input, image_latents], dim=1
                        )
                        latent_model_input = student_dpm_scheduler.scale_model_input(
                            latent_model_input, t_freeze
                        )
                        # random_view = random.randint(0, 5)
                        # latent_model_input = random_select_input(random_view, latent_model_input)
                        fake_output = unet_freeze(
                            latent_model_input.to(weight_dtype),
                            t_freeze,
                            encoder_hidden_states=prompt_embeds.to(weight_dtype),
                            class_labels=image_embeds.to(weight_dtype),
                            return_dict=False,
                        )
                        # cond_teacher_noise_pred = cond_teacher_output[0]
                        fake_features = fake_output[2]  # [24, 1280, 32, 32]

                        latent_model_input = real_noisy_model_input
                        latent_model_input = torch.cat(
                            [latent_model_input, real_image_latents], dim=1
                        )
                        # latent_model_input = random_select_input(random_view, latent_model_input)
                        latent_model_input = student_dpm_scheduler.scale_model_input(
                            latent_model_input, t_freeze
                        )

                        real_output = unet_freeze(
                            latent_model_input.to(weight_dtype),
                            t_freeze,
                            encoder_hidden_states=prompt_embeds.to(weight_dtype),
                            class_labels=real_image_embeds.to(weight_dtype),
                            return_dict=False,
                        )
                        # cond_teacher_noise_pred = cond_teacher_output[0]
                        real_features = real_output[2]

                # generator.student_unet.enable_adapter_layers()
                # if global_step % args.discriminator_gradient_freq == 0:
                if random.randint(0, 100) <= args.discriminator_gradient_freq:
                    real_features_normal, real_features_color = split_color_normal(
                        [fea.detach() for fea in real_features]
                    )
                    fake_features_normal, fake_features_color = split_color_normal(
                        [fea.detach() for fea in fake_features]
                    )

                    update_discriminator = True
                    real_out_color = discriminator_color(real_features_color)
                    real_out_normal = discriminator_normal(real_features_normal)
                    real_out = torch.cat([real_out_normal, real_out_color], 0)
                    loss_real_D_color = F.softplus(
                        -real_out_color
                    ).mean()  # criterion(real_out, real_label)
                    loss_real_D_normal = F.softplus(
                        -real_out_normal
                    ).mean()  # criterion(real_out, real_label)
                    real_scores = torch.sigmoid(real_out).squeeze(dim=1)

                    fake_out_color = discriminator_color(fake_features_color)
                    fake_out_normal = discriminator_normal(fake_features_normal)
                    fake_out = torch.cat([fake_out_normal, fake_out_color], 0)
                    loss_fake_D_color = F.softplus(fake_out_color).mean()
                    loss_fake_D_normal = F.softplus(fake_out_normal).mean()
                    fake_scores = torch.sigmoid(fake_out).squeeze(dim=1)

                    loss_D_color = 0.5 * (loss_real_D_color + loss_fake_D_color)
                    loss_D_normal = 0.5 * (loss_real_D_normal + loss_fake_D_normal)

                    optimizer_D_color.zero_grad()
                    accelerator.backward(loss_D_color)
                    optimizer_D_color.step()

                    optimizer_D_normal.zero_grad()
                    accelerator.backward(loss_D_normal)
                    optimizer_D_normal.step()

                # Generator Loss
                fake_img = fake_features_grad
                fake_img_normal, fake_img_color = split_color_normal(fake_img)
                output_color = discriminator_color(fake_img_color)
                output_normal = discriminator_normal(fake_img_normal)
                output = torch.cat([output_normal, output_color], 0)
                generator_scores = torch.sigmoid(output).squeeze(dim=1)
                loss_G = F.softplus(-output).mean()  # criterion(output, real_label)
                loss_dmd = F.l1_loss(output_x0, x0_prime).mean()
                loss_consistency = F.l1_loss(x0_preds[1], x0_preds[0].detach()).mean()
                # loss_consistency_gt = reweight(args.noise_start_steptime, cleaner_start_timestep)*F.l1_loss(x0_preds[0], real_image_latents).mean() + reweight(args.noise_start_steptime, noisier_start_timestep)*F.l1_loss(x0_preds[1], real_image_latents).mean()
                # loss_consistency_gt = alpha_t_cleaner * F.l1_loss(x0_preds[0], real_image_latents).mean() + alpha_t_noiser * F.l1_loss(x0_preds[1], real_image_latents).mean()
                loss_consistency_gt = (
                    F.l1_loss(x0_preds[0], real_image_latents).mean()
                    + F.l1_loss(x0_preds[1], real_image_latents).mean()
                )
                optimizer_G.zero_grad()
                accelerator.backward(
                    loss_G + loss_dmd + loss_consistency + loss_consistency_gt
                )

                torch.cuda.empty_cache()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar_total.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = (
                                    len(checkpoints) - args.checkpoints_total_limit + 1
                                )
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )
                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        os.makedirs(save_path, exist_ok=True)
                        accelerator.save_state(save_path)
                        discriminator_color_ = accelerator.unwrap_model(
                            discriminator_color
                        )
                        discriminator_normal_ = accelerator.unwrap_model(
                            discriminator_normal
                        )
                        torch.save(
                            discriminator_color_.state_dict(),
                            os.path.join(save_path, "discriminator_color.pt"),
                        )
                        torch.save(
                            discriminator_normal_.state_dict(),
                            os.path.join(save_path, "discriminator_normal.pt"),
                        )
                        logger.info(f"Saved state to {save_path}")

                    if global_step % args.validation_steps == 0 and global_step != 0:
                        logger.info("Running visualization save...")

                        # dpm_scheduler.set_timesteps(num_inference_steps=40, device=accelerator.device)
                        # latents_fake_grad = dpm_scheduler.step(fake_output_grad[0], t_freeze, fake_noisy_model_input_grad, **extra_step_kwargs, return_dict=False)[0]

                        dpm_scheduler.set_timesteps(
                            num_inference_steps=1000, device=accelerator.device
                        )
                        latents_fake = dpm_scheduler.step(
                            fake_output[0],
                            t_freeze,
                            fake_noisy_model_input,
                            **extra_step_kwargs,
                            return_dict=False,
                        )[0]

                        dpm_scheduler.set_timesteps(
                            num_inference_steps=1000, device=accelerator.device
                        )
                        latents_real = dpm_scheduler.step(
                            real_output[0],
                            t_freeze,
                            real_noisy_model_input,
                            **extra_step_kwargs,
                            return_dict=False,
                        )[0]

                        log_visualization_gan(
                            # latents_student_vis=latents_fake_grad,
                            latents_student_vis=latents_student_vis,
                            latents_fake=latents_fake,
                            latents_real=latents_real,
                            vae=generator.vae,
                            image_processor=generator.image_processor,
                            accelerator=accelerator,
                        )
            if update_discriminator:
                logs = {
                    "loss_G": loss_G.detach().item(),
                    "loss_dmd": loss_dmd.detach().item(),
                    "loss_consistency": loss_consistency.detach().item(),
                    "loss_consistency_gt": loss_consistency_gt.detach().item(),
                    "loss_D_color": loss_D_color.detach().item(),
                    "loss_D_normal": loss_D_normal.detach().item(),
                    "generator_scores": generator_scores[0].detach(),
                    "fake_scores": fake_scores[0].detach(),  # maybe no item()
                    "real_scores": real_scores[0].detach(),  # maybe no item()
                }
                update_discriminator = False
            else:
                logs = {
                    "loss_G": loss_G.detach().item(),
                    "loss_dmd": loss_dmd.detach().item(),
                    "loss_consistency": loss_consistency.detach().item(),
                    "loss_consistency_gt": loss_consistency_gt.detach().item(),
                    "generator_scores": generator_scores[0].detach(),
                }

            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet_ = accelerator.unwrap_model(generator.student_unet)
        unet_.save_pretrained(args.output_dir)
        lora_state_dict = get_peft_model_state_dict(unet_, adapter_name="default")
        StableDiffusionPipeline.save_lora_weights(
            os.path.join(args.output_dir, "unet_lora"), lora_state_dict
        )

    accelerator.end_training()


if __name__ == "__main__":
    import argparse, yaml
    from omegaconf import OmegaConf
    from utils.misc import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", type=str)
    args, extras = parser.parse_known_args()
    with open(args.config, "r") as stream:
        data = yaml.safe_load(stream)
    exp_name = data.get("tracker_project_name", None)
    exp_path = os.path.join("./outputs_formal", "lora_64_" + exp_name)
    os.makedirs(exp_path, exist_ok=True)
    with open(os.path.join(exp_path, "config.yaml"), "w") as outfile:
        yaml.safe_dump(data, outfile)

    # parse YAML config to OmegaConf
    cfg = load_config(args.config, cli_args=extras)
    print(cfg)
    schema = OmegaConf.structured(TestConfig)
    cfg = OmegaConf.merge(schema, cfg)

    VIEWS = ["front", "front_right", "right", "back", "left", "front_left"]
    main(cfg)
