import gc
import wandb
import random
import numpy as np
import torch
import torch.utils.checkpoint

from accelerate.logging import get_logger
from peft import get_peft_model_state_dict, LoraConfig, get_peft_model
from transformers import PretrainedConfig
from tqdm import tqdm
from einops import rearrange
from torchvision.utils import make_grid, save_image
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput

from diffusers import (
    DDIMScheduler,
    StableDiffusionPipeline,
)

logger = get_logger(__name__)



def cycle(dl):
    while True:
        for data in dl:
            yield data

def get_module_kohya_state_dict(
    module, prefix: str, dtype: torch.dtype, adapter_name: str = "default"
):
    kohya_ss_state_dict = {}
    for peft_key, weight in get_peft_model_state_dict(
        module, adapter_name=adapter_name
    ).items():
        kohya_key = peft_key.replace("base_model.model", prefix)
        kohya_key = kohya_key.replace("lora_A", "lora_down")
        kohya_key = kohya_key.replace("lora_B", "lora_up")
        kohya_key = kohya_key.replace(".", "_", kohya_key.count(".") - 2)
        kohya_ss_state_dict[kohya_key] = weight.to(dtype)

        # Set alpha parameter
        if "lora_down" in kohya_key:
            alpha_key = f'{kohya_key.split(".")[0]}.alpha'
            kohya_ss_state_dict[alpha_key] = torch.tensor(
                module.peft_config[adapter_name].lora_alpha
            ).to(dtype)

    return kohya_ss_state_dict


def extract_lora_state_dict(model):
    lora_state_dict = {}
    for name, module in model.named_modules():
            lora_state_dict[name + ".lora_A"] = module.lora_A
            lora_state_dict[name + ".lora_B"] = module.lora_B
    return lora_state_dict

def visual(bsz, normals, images, num_views=6):
    image_logs = []
    for i in range(bsz // num_views):
        vis_ = []
        for j in range(num_views):
            
            idx = i * num_views + j
            normal = normals[idx]
            color = images[idx]

            vis_.append(color)
            vis_.append(normal)

        vis_ = torch.stack(vis_, dim=0)
        vis_ = make_grid(vis_, nrow=len(vis_), padding=0, value_range=(0, 1))

    image_logs.append({"Results_Grid": vis_})
    return image_logs


def log_visualization_gan(
        latents_student_vis,
        latents_fake,
        latents_real,
        vae,
        image_processor,
        accelerator,
):
    torch.cuda.empty_cache()
    with torch.no_grad():
        v_img = vae.decode(latents_student_vis.to(vae.dtype) / vae.config.scaling_factor, return_dict=False)[0]
        s_img = vae.decode(latents_fake.to(vae.dtype) / vae.config.scaling_factor, return_dict=False)[0]
        t_img = vae.decode(latents_real.to(vae.dtype) / vae.config.scaling_factor, return_dict=False)[0]
    
        # s_img = image_processor.postprocess(student_image, output_type='pil')
        # t_img = image_processor.postprocess(teacher_image, output_type='pil')
   
        # out = unet_out.images  # [12, 3, 512, 512]
       
        bsz = s_img.shape[0] //  2 # s_img.shape[0] // 2
        v_normals_pred = v_img[:bsz]  # [6, 3, 512, 512]
        v_images_pred = v_img[bsz:]  # [6, 3, 512, 512]
        s_normals_pred = s_img[:bsz]  # [6, 3, 512, 512]
        s_images_pred = s_img[bsz:]  # [6, 3, 512, 512]
        t_normals_pred = t_img[:bsz]  # [6, 3, 512, 512]
        t_images_pred = t_img[bsz:]  # [6, 3, 512, 512]
        v_logs = visual(bsz, v_normals_pred, v_images_pred)
        s_logs = visual(bsz, s_normals_pred, s_images_pred)
        t_logs = visual(bsz, t_normals_pred, t_images_pred)
    torch.cuda.empty_cache()
    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            v_formatted_images = []
            for log in v_logs:
                images = log["Results_Grid"]
                # for image in images:
                images = wandb.Image(images)
                v_formatted_images.append(images)
            s_formatted_images = []
            for log in s_logs:
                images = log["Results_Grid"]
                # for image in images:
                images = wandb.Image(images)
                s_formatted_images.append(images)
            t_formatted_images = []
            for log in t_logs:
                images = log["Results_Grid"]
                # for image in images:
                images = wandb.Image(images)
                t_formatted_images.append(images)

            tracker.log({"Visualizations for generator": v_formatted_images, "Visualizations for fake images in discriminator": s_formatted_images, "Visualizations for real images in discriminator": t_formatted_images})
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")


def log_visualization(
        objavere_or_imagenet,
        latents_student_vis,
        latents_fake,
        latents_real,
        vae,
        image_processor,
        accelerator,
):
    torch.cuda.empty_cache()
    with torch.no_grad():
        v_img = vae.decode(latents_student_vis.to(vae.dtype) / vae.config.scaling_factor, return_dict=False)[0]
        s_img = vae.decode(latents_fake.to(vae.dtype) / vae.config.scaling_factor, return_dict=False)[0]
        t_img = vae.decode(latents_real.to(vae.dtype) / vae.config.scaling_factor, return_dict=False)[0]
    
        # s_img = image_processor.postprocess(student_image, output_type='pil')
        # t_img = image_processor.postprocess(teacher_image, output_type='pil')
   
        # out = unet_out.images  # [12, 3, 512, 512]
       
        bsz = s_img.shape[0] //  2 # s_img.shape[0] // 2
        v_normals_pred = v_img[:bsz]  # [6, 3, 512, 512]
        v_images_pred = v_img[bsz:]  # [6, 3, 512, 512]
        s_normals_pred = s_img[:bsz]  # [6, 3, 512, 512]
        s_images_pred = s_img[bsz:]  # [6, 3, 512, 512]
        t_normals_pred = t_img[:bsz]  # [6, 3, 512, 512]
        t_images_pred = t_img[bsz:]  # [6, 3, 512, 512]
        v_logs = visual(bsz, v_normals_pred, v_images_pred)
        s_logs = visual(bsz, s_normals_pred, s_images_pred)
        t_logs = visual(bsz, t_normals_pred, t_images_pred)
    torch.cuda.empty_cache()
    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            v_formatted_images = []
            for log in v_logs:
                images = log["Results_Grid"]
                # for image in images:
                images = wandb.Image(images)
                v_formatted_images.append(images)
            s_formatted_images = []
            for log in s_logs:
                images = log["Results_Grid"]
                # for image in images:
                images = wandb.Image(images)
                s_formatted_images.append(images)
            t_formatted_images = []
            for log in t_logs:
                images = log["Results_Grid"]
                # for image in images:
                images = wandb.Image(images)
                t_formatted_images.append(images)

            tracker.log({f"Visualizations for generator {objavere_or_imagenet}": v_formatted_images, f"Visualizations for fake images in discriminator {objavere_or_imagenet}": s_formatted_images, f"Visualizations for real images in discriminator {objavere_or_imagenet}": t_formatted_images})
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")


def log_visualization_distill(
        latents_student,
        latents_teacher,
        vae,
        image_processor,
        accelerator,
):
    torch.cuda.empty_cache()
    with torch.no_grad():
       
        s_img = vae.decode(latents_student.to(vae.dtype) / vae.config.scaling_factor, return_dict=False)[0]
        t_img = vae.decode(latents_teacher.to(vae.dtype) / vae.config.scaling_factor, return_dict=False)[0]
    
        # s_img = image_processor.postprocess(student_image, output_type='pil')
        # t_img = image_processor.postprocess(teacher_image, output_type='pil')
   
        # out = unet_out.images  # [12, 3, 512, 512]
       
        bsz = s_img.shape[0] //  2 # s_img.shape[0] // 2
        s_normals_pred = s_img[:bsz]  # [6, 3, 512, 512]
        s_images_pred = s_img[bsz:]  # [6, 3, 512, 512]
        t_normals_pred = t_img[:bsz]  # [6, 3, 512, 512]
        t_images_pred = t_img[bsz:]  # [6, 3, 512, 512]
        s_logs = visual(bsz, s_normals_pred, s_images_pred)
        t_logs = visual(bsz, t_normals_pred, t_images_pred)
    torch.cuda.empty_cache()
    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            s_formatted_images = []
            for log in s_logs:
                images = log["Results_Grid"]
                # for image in images:
                images = wandb.Image(images)
                s_formatted_images.append(images)
            t_formatted_images = []
            for log in t_logs:
                images = log["Results_Grid"]
                # for image in images:
                images = wandb.Image(images)
                t_formatted_images.append(images)

            tracker.log({"Visualizations for student": s_formatted_images, "Visualizations for teacher": t_formatted_images})
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

# @torch.no_grad()
def log_validation_era3d(
    vae,
    unet,
    args,
    accelerator,
    weight_dtype,
    step,
    guidance_scale,
    num_inference_step,
    val_dataloader,
    Views,
):
    logger.info("Running validation... ")

    unet = accelerator.unwrap_model(unet)

    pipeline = StableUnCLIPImg2ImgPipeline.from_pretrained(
        args.pretrained_teacher_model,
        vae=vae,
        scheduler=DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear", # "linear" for ori
            timestep_spacing="trailing",  # "leading" for ori
        ),  # DDIM should just work well. See our discussion on parameterization in the paper.
        revision=args.revision,
        torch_dtype=weight_dtype,
        safety_checker=None,
    )
    pipeline.set_progress_bar_config(disable=True)
    
    # print(unet.state_dict)
    # lora_state_dict = get_module_kohya_state_dict(unet, "lora_unet", weight_dtype)
    # lora_state_dict = get_peft_model_state_dict(unet, adapter_name="default")    

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
    unet_ = get_peft_model(pipeline.unet, lora_config)
    pipeline.unet = unet_
    lora_state_dict = get_peft_model_state_dict(
                    unet, adapter_name="default"
                )
    pipeline.load_lora_weights(lora_state_dict)
    pipeline.fuse_lora()
    pipeline = pipeline.to(accelerator.device, dtype=weight_dtype)
    torch.cuda.empty_cache()
    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()
    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    image_logs = []
    images_cond = []
    for step, batch in tqdm(enumerate(val_dataloader)):
        images_cond.append(batch["imgs_in"][:, 0])
        imgs_in = torch.cat([batch["imgs_in"]] * 2, dim=0)
        num_views = imgs_in.shape[1]
        imgs_in = rearrange(imgs_in, "B Nv C H W -> (B Nv) C H W")  # (B*Nv, 3, H, W)

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
                # B*Nv images
                # for guidance_scale in args.validation_guidance_scales:
                unet_out = pipeline(
                    imgs_in,
                    batch,
                    args,
                    Views,
                    None,
                    prompt_embeds=prompt_embeddings,
                    generator=generator,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_step,
                    output_type="pt",
                    num_images_per_prompt=1,
                    eta=1.0,
                )

                out = unet_out.images  # [12, 3, 512, 512]
                bsz = out.shape[0] // 2

                normals_pred = out[:bsz]  # [6, 3, 512, 512]
                images_pred = out[bsz:]  # [6, 3, 512, 512]
                vis_ = []
                for i in range(bsz // num_views):
                    scene = batch["filename"][i].split(".")[0]
                    img_in_ = images_cond[-1][i].to(out.device)
                    vis_ = [img_in_]
                    for j in range(num_views):
                        view = Views[j]
                        idx = i * num_views + j
                        normal = normals_pred[idx]
                        color = images_pred[idx]

                        vis_.append(color)
                        vis_.append(normal)

                    vis_ = torch.stack(vis_, dim=0)
                    vis_ = make_grid(vis_, nrow=len(vis_), padding=0, value_range=(0, 1))

                image_logs.append({"Results_Grid": vis_})
            torch.cuda.empty_cache()
            if step == 3:
                break
        

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["Results_Grid"]
                formatted_images = []
                for image in images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images(
                    f"{guidance_scale}: " + formatted_images, step, dataformats="NHWC"
                )
        elif tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                images = log["Results_Grid"]
                for image in images:
                    import pdb
                    pdb.set_trace()
                    image = wandb.Image(image)
                    formatted_images.append(image)

            tracker.log({f"validation-{guidance_scale}": formatted_images})
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

        return image_logs


def log_validation_sd15(
    vae, unet, args, accelerator, weight_dtype, step, cfg, num_inference_step
):
    logger.info("Running validation... ")

    unet = accelerator.unwrap_model(unet)

    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_teacher_model,
        vae=vae,
        scheduler=DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            timestep_spacing="trailing",
        ),  # DDIM should just work well. See our discussion on parameterization in the paper.
        # revision=args.revision,
        torch_dtype=weight_dtype,
        safety_checker=None,
        cache_dir="/data1/kendong/cache",
    )
    pipeline.set_progress_bar_config(disable=True)
    lora_state_dict = get_module_kohya_state_dict(unet, "lora_unet", weight_dtype)
    pipeline.load_lora_weights(lora_state_dict)
    pipeline.fuse_lora()
    pipeline = pipeline.to(accelerator.device, dtype=weight_dtype)
    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()
    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    validation_prompts = [
        "portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour, style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography",
        "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
        "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
        "A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece",
    ]

    image_logs = []

    for _, prompt in enumerate(validation_prompts):
        images = []
        with torch.autocast("cuda", dtype=weight_dtype):
            images = pipeline(
                prompt=prompt,
                num_inference_steps=num_inference_step,
                num_images_per_prompt=4,
                generator=generator,
                guidance_scale=cfg,
            ).images
        image_logs.append({"validation_prompt": prompt, "images": images})

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                formatted_images = []
                for image in images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images(
                    f"{cfg}: " + validation_prompt,
                    formatted_images,
                    step,
                    dataformats="NHWC",
                )
        elif tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                for image in images:
                    image = wandb.Image(image, caption=validation_prompt)
                    formatted_images.append(image)

            tracker.log({f"validation-{cfg}": formatted_images})
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

        return image_logs


# From LatentConsistencyModel.get_guidance_scale_embedding
def guidance_scale_embedding(w, embedding_dim=512, dtype=torch.float32):
    """
    See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

    Args:
        timesteps (`torch.Tensor`):
            generate embedding vectors at these timesteps
        embedding_dim (`int`, *optional*, defaults to 512):
            dimension of the embeddings to generate
        dtype:
            data type of the generated embeddings

    Returns:
        `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
    """
    assert len(w.shape) == 1
    w = w * 1000.0

    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
    emb = w.to(dtype)[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1))
    assert emb.shape == (w.shape[0], embedding_dim)
    return emb


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


def scalings_for_boundary_conditions_target(index, selected_indices):
    c_skip = torch.isin(index, selected_indices).float()
    c_out = 1.0 - c_skip
    return c_skip, c_out


def scalings_for_boundary_conditions_online(index, selected_indices):
    c_skip = torch.zeros_like(index).float()
    c_out = torch.ones_like(index).float()
    return c_skip, c_out


def scalings_for_boundary_conditions(timestep, sigma_data=0.5, timestep_scaling=10.0):
    c_skip = sigma_data**2 / ((timestep / 0.1) ** 2 + sigma_data**2)
    c_out = (timestep / 0.1) / ((timestep / 0.1) ** 2 + sigma_data**2) ** 0.5
    return c_skip, c_out


def predicted_origin(model_output, timesteps, sample, prediction_type, alphas, sigmas):
    if prediction_type == "epsilon":
        sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
        alphas = extract_into_tensor(alphas, timesteps, sample.shape)
        pred_x_0 = (sample - sigmas * model_output) / alphas
    elif prediction_type == "v_prediction":
        sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
        alphas = extract_into_tensor(alphas, timesteps, sample.shape)
        pred_x_0 = alphas * sample - sigmas * model_output
    else:
        raise ValueError(f"Prediction type {prediction_type} currently not supported.")

    return pred_x_0


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class DDIMSolver:
    def __init__(self, alpha_cumprods, timesteps=1000, ddim_timesteps=50):
        self.step_ratio = timesteps // ddim_timesteps
        self.ddim_timesteps = (
            np.arange(1, ddim_timesteps + 1) * self.step_ratio
        ).round().astype(np.int64) - 1
        self.ddim_alpha_cumprods = alpha_cumprods[self.ddim_timesteps]
        self.ddim_timesteps_prev = np.asarray([0] + self.ddim_timesteps[:-1].tolist())
        self.ddim_alpha_cumprods_prev = np.asarray(
            [alpha_cumprods[0]] + alpha_cumprods[self.ddim_timesteps[:-1]].tolist()
        )  # [0.9915] + [0.9822, ..., 0.005] ([1] + [49])

        self.ddim_timesteps = torch.from_numpy(self.ddim_timesteps).long()
        self.ddim_timesteps_prev = torch.from_numpy(self.ddim_timesteps_prev).long()
        self.ddim_alpha_cumprods = torch.from_numpy(self.ddim_alpha_cumprods)
        self.ddim_alpha_cumprods_prev = torch.from_numpy(self.ddim_alpha_cumprods_prev)

    def to(self, device):
        self.ddim_timesteps = self.ddim_timesteps.to(device)
        self.ddim_timesteps_prev = self.ddim_timesteps_prev.to(device)

        self.ddim_alpha_cumprods = self.ddim_alpha_cumprods.to(device)
        self.ddim_alpha_cumprods_prev = self.ddim_alpha_cumprods_prev.to(device)
        return self

    def ddim_step(self, pred_x0, pred_noise, timestep_index):

        alpha_cumprod_prev = extract_into_tensor(
            self.ddim_alpha_cumprods_prev, timestep_index, pred_x0.shape
        )
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
        x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt
        return x_prev

    def ddim_style_multiphase_pred(
        self, pred_x0, pred_noise, timestep_index, multiphase
    ):

        inference_indices = np.linspace(
            0, len(self.ddim_timesteps), num=multiphase, endpoint=False
        )
        inference_indices = np.floor(inference_indices).astype(np.int64)
        inference_indices = (
            torch.from_numpy(inference_indices).long().to(self.ddim_timesteps.device)
        )
        expanded_timestep_index = timestep_index.unsqueeze(1).expand(
            -1, inference_indices.size(0)
        )
        valid_indices_mask = expanded_timestep_index >= inference_indices
        last_valid_index = valid_indices_mask.flip(dims=[1]).long().argmax(dim=1)
        last_valid_index = inference_indices.size(0) - 1 - last_valid_index
        timestep_index = inference_indices[last_valid_index]
        alpha_cumprod_prev = extract_into_tensor(
            self.ddim_alpha_cumprods_prev, timestep_index, pred_x0.shape
        )
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
        x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt
        return x_prev, self.ddim_timesteps_prev[timestep_index]


@torch.no_grad()
def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder=subfolder,
        revision=revision,
        use_auth_token=True,
        cache_dir="/data/kendong/cache",
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


# Adapted from pipelines.StableDiffusionPipeline.encode_prompt
def encode_prompt(
    prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train=True
):
    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        text_inputs = tokenizer(
            captions,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(text_input_ids.to(text_encoder.device))[0]

    return prompt_embeds
