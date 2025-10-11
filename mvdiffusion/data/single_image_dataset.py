from typing import Dict
import numpy as np
from omegaconf import DictConfig, ListConfig
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from PIL import Image
from torchvision import transforms
from einops import rearrange
from typing import Literal, Tuple, Optional, Any
import cv2
import random

import json
import os, sys
import math

from glob import glob

import PIL.Image
from .normal_utils import trans_normal, normal2img, img2normal
import pdb
from icecream import ic
from tqdm import tqdm

import cv2
import numpy as np


def add_margin(pil_img, color=0, size=256):
    width, height = pil_img.size
    result = Image.new(pil_img.mode, (size, size), color)
    result.paste(pil_img, ((size - width) // 2, (size - height) // 2))
    return result


def scale_and_place_object(image, scale_factor):
    assert np.shape(image)[-1] == 4  # RGBA

    # Extract the alpha channel (transparency) and the object (RGB channels)
    alpha_channel = image[:, :, 3]

    # Find the bounding box coordinates of the object
    coords = cv2.findNonZero(alpha_channel)
    x, y, width, height = cv2.boundingRect(coords)

    # Calculate the scale factor for resizing
    original_height, original_width = image.shape[:2]

    if width > height:
        size = width
        original_size = original_width
    else:
        size = height
        original_size = original_height

    scale_factor = min(scale_factor, size / (original_size + 0.0))

    new_size = scale_factor * original_size
    scale_factor = new_size / size

    # Calculate the new size based on the scale factor
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    center_x = original_width // 2
    center_y = original_height // 2

    paste_x = center_x - (new_width // 2)
    paste_y = center_y - (new_height // 2)

    # Resize the object (RGB channels) to the new size
    rescaled_object = cv2.resize(
        image[y : y + height, x : x + width], (new_width, new_height)
    )

    # Create a new RGBA image with the resized image
    new_image = np.zeros((original_height, original_width, 4), dtype=np.uint8)

    new_image[paste_y : paste_y + new_height, paste_x : paste_x + new_width] = (
        rescaled_object
    )

    return new_image


class SingleImageDatasetGAN(Dataset):
    def __init__(
        self,
        root_dir: str,
        json_path: str,
        num_views: int,
        img_wh: Tuple[int, int],
        bg_color: str,
        crop_size: int = 224,
        single_image: Optional[PIL.Image.Image] = None,
        num_validation_samples: Optional[int] = None,
        filepaths: Optional[list] = None,
        cond_type: Optional[str] = None,
        prompt_embeds_path: Optional[str] = None,
        gt_path: Optional[str] = None,
        examples: Optional[str] = None,
        sample_rate: Optional[float] = 1.0,
    ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.json_path = json_path
        self.root_dir = root_dir
        self.num_views = num_views
        self.img_wh = img_wh
        self.crop_size = crop_size
        self.bg_color = bg_color
        self.cond_type = cond_type
        self.gt_path = gt_path
        self.file_list = None

        with open(self.json_path, "r") as f:
            file_list = json.load(f)
        self.file_list = file_list
        # sample_size = int(len(file_list) * sample_rate)
        # self.file_list = random.sample(file_list, sample_size)

        self.all_images = []
        self.all_alphas = []
        self.bg_color = self.get_bg_color()
        # self.folder_list = os.listdir(self.root_dir)
        self.image_fix_key = [
            "normals_006_front",
            "normals_006_front_right",
            "normals_006_right",
            "normals_006_back",
            "normals_006_left",
            "normals_006_front_left",
            "rgb_006_front",
            "rgb_006_front_right",
            "rgb_006_right",
            "rgb_006_back",
            "rgb_006_left",
            "rgb_006_front_left",
        ]

        self.view_list = self.image_fix_key[6:]

        try:
            self.normal_text_embeds = torch.load(
                f"{prompt_embeds_path}/normal_embeds.pt"
            )
            self.color_text_embeds = torch.load(
                f"{prompt_embeds_path}/clr_embeds.pt"
            )  # 4view
        except:
            self.color_text_embeds = torch.load(f"{prompt_embeds_path}/embeds.pt")
            self.normal_text_embeds = None

    def __len__(self):
        return len(self.file_list)

    def sample_view(self, examples=False):
        if examples:
            return self.view_list[0]
        else:
            if random.random() < 0.5:
                return self.view_list[0]
            else:
                return random.choice(self.view_list[1:])

    def get_bg_color(self):
        if self.bg_color == "white":
            bg_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        elif self.bg_color == "black":
            bg_color = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        elif self.bg_color == "gray":
            bg_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        elif self.bg_color == "random":
            bg_color = np.random.rand(3)
        elif isinstance(self.bg_color, float):
            bg_color = np.array([self.bg_color] * 3, dtype=np.float32)
        else:
            raise NotImplementedError
        return bg_color

    def load_image(self, img_path, bg_color, return_type="np"):
        # not using cv2 as may load in uint16 format
        # img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # [0, 255]
        # img = cv2.resize(img, self.img_wh, interpolation=cv2.INTER_CUBIC)
        # pil always returns uint8
        img = np.array(Image.open(img_path).resize(self.img_wh))
        img = img.astype(np.float32) / 255.0  # [0, 1]
        assert img.shape[-1] == 4  # RGBA

        alpha = img[..., 3:4]
        img = img[..., :3] * alpha + bg_color * (1 - alpha)

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
        else:
            raise NotImplementedError

        return img, alpha

    def load_normal(
        self, img_path, bg_color, alpha, RT_w2c=None, RT_w2c_cond=None, return_type="np"
    ):
        # not using cv2 as may load in uint16 format
        # img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # [0, 255]
        # img = cv2.resize(img, self.img_wh, interpolation=cv2.INTER_CUBIC)
        # pil always returns uint8
        normal = np.array(Image.open(img_path).convert("RGB").resize(self.img_wh))

        assert normal.shape[-1] == 3  # RGB

        # normal = trans_normal(img2normal(normal), RT_w2c, RT_w2c_cond)
        # img = normal2img(normal)

        img = img.astype(np.float32) / 255.0  # [0, 1]

        img = img[..., :3] * alpha + bg_color * (1 - alpha)

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
        else:
            raise NotImplementedError

        return img

    def __getitem__(self, index):

        file = self.file_list[index]

        random_view = self.sample_view(examples=True)
        image, _ = self.load_image(
            os.path.join(self.root_dir, file, random_view + ".webp"),
            self.bg_color,
            return_type="pt",
        )
        img_tensors_in = [image.permute(2, 0, 1)] * self.num_views
        img_tensors_in = torch.stack(img_tensors_in, dim=0).float()  # (Nv, 3, H, W)

        real_img_tensors_in = []
        for key in self.image_fix_key:
            real_image, _ = self.load_image(
                os.path.join(self.root_dir, file, key + ".webp"),
                self.bg_color,
                return_type="pt",
            )
            real_img_tensors_in.append(real_image.permute(2, 0, 1))

        real_img_tensors_in = torch.stack(
            real_img_tensors_in, dim=0
        ).float()  # (Nv, 3, H, W)

        normal_prompt_embeddings = (
            self.normal_text_embeds if hasattr(self, "normal_text_embeds") else None
        )
        color_prompt_embeddings = (
            self.color_text_embeds if hasattr(self, "color_text_embeds") else None
        )

        out = {
            "imgs_in": img_tensors_in,
            "real_imgs_in": real_img_tensors_in,
            "normal_prompt_embeddings": normal_prompt_embeddings,
            "color_prompt_embeddings": color_prompt_embeddings,
        }

        return out


class SingleImageDatasetGenerator(Dataset):
    def __init__(
        self,
        root_dir: str,
        json_path: str,
        num_views: int,
        img_wh: Tuple[int, int],
        bg_color: str,
        crop_size: int = 224,
        single_image: Optional[PIL.Image.Image] = None,
        num_validation_samples: Optional[int] = None,
        filepaths: Optional[list] = None,
        cond_type: Optional[str] = None,
        prompt_embeds_path: Optional[str] = None,
        gt_path: Optional[str] = None,
        sample_rate: Optional[float] = 1.0,
    ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.json_path = json_path
        self.root_dir = root_dir
        self.num_views = num_views
        self.img_wh = img_wh
        self.crop_size = crop_size
        self.bg_color = bg_color
        self.cond_type = cond_type
        self.gt_path = gt_path

        self.file_list = None
        self.image_fix_key = [
            "normals_006_front",
            "normals_006_front_right",
            "normals_006_right",
            "normals_006_back",
            "normals_006_left",
            "normals_006_front_left",
            "rgb_006_front",
            "rgb_006_front_right",
            "rgb_006_right",
            "rgb_006_back",
            "rgb_006_left",
            "rgb_006_front_left",
        ]

        self.view_list = self.image_fix_key[6:]

        with open(self.json_path, "r") as f:
            file_list = json.load(f)
        self.file_list = file_list

        self.all_images = []
        self.all_alphas = []
        self.bg_color = self.get_bg_color()

        try:
            self.normal_text_embeds = torch.load(
                f"{prompt_embeds_path}/normal_embeds.pt"
            )
            self.color_text_embeds = torch.load(
                f"{prompt_embeds_path}/clr_embeds.pt"
            )  # 4view
        except:
            self.color_text_embeds = torch.load(f"{prompt_embeds_path}/embeds.pt")
            self.normal_text_embeds = None

    def __len__(self):
        return len(self.file_list)

    def sample_view(self, examples=False):
        if examples:
            return self.view_list[0]
        else:
            if random.random() < 0.5:
                return self.view_list[0]
            else:
                return random.choice(self.view_list[1:])

    def get_bg_color(self):
        if self.bg_color == "white":
            bg_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        elif self.bg_color == "black":
            bg_color = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        elif self.bg_color == "gray":
            bg_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        elif self.bg_color == "random":
            bg_color = np.random.rand(3)
        elif isinstance(self.bg_color, float):
            bg_color = np.array([self.bg_color] * 3, dtype=np.float32)
        else:
            raise NotImplementedError
        return bg_color

    def load_image(self, img_path, bg_color, return_type="np"):
        image_input = Image.open(img_path)  # .resize(self.img_wh))
        image_size = self.img_wh[0]
        if self.crop_size != -1:
            alpha_np = np.asarray(image_input)[:, :, 3]
            coords = np.stack(np.nonzero(alpha_np), 1)[:, (1, 0)]
            min_x, min_y = np.min(coords, 0)
            max_x, max_y = np.max(coords, 0)
            ref_img_ = image_input.crop((min_x, min_y, max_x, max_y))
            h, w = ref_img_.height, ref_img_.width
            scale = self.crop_size / max(h, w)
            h_, w_ = int(scale * h), int(scale * w)
            ref_img_ = ref_img_.resize((w_, h_))
            image_input = add_margin(ref_img_, size=image_size)
        else:
            image_input = add_margin(
                image_input, size=max(image_input.height, image_input.width)
            )
            image_input = image_input.resize((image_size, image_size))

        img = np.array(image_input).astype(np.float32) / 255.0  # [0, 1]
        assert img.shape[-1] == 4  # RGBA

        alpha = img[..., 3:4]
        img = img[..., :3] * alpha + bg_color * (1 - alpha)

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
        else:
            raise NotImplementedError

        return img, alpha

    def __getitem__(self, index):

        file = self.file_list[index]

        random_view = self.sample_view(examples=True)

        image, _ = self.load_image(
            os.path.join(self.root_dir, file, "color_front_masked.png"),
            self.bg_color,
            return_type="pt",
        )

        img_tensors_in = [image.permute(2, 0, 1)] * self.num_views
        img_tensors_in = torch.stack(img_tensors_in, dim=0).float()  # (Nv, 3, H, W)
        print(img_tensors_in.shape)

        normal_prompt_embeddings = (
            self.normal_text_embeds if hasattr(self, "normal_text_embeds") else None
        )
        color_prompt_embeddings = (
            self.color_text_embeds if hasattr(self, "color_text_embeds") else None
        )

        out = {
            "filename": file,
            "imgs_in": img_tensors_in,
            "normal_prompt_embeddings": normal_prompt_embeddings,
            "color_prompt_embeddings": color_prompt_embeddings,
        }

        return out


class SingleImageDatasetDiscriminator(Dataset):
    def __init__(
        self,
        root_dir: str,
        json_path: str,
        num_views: int,
        img_wh: Tuple[int, int],
        bg_color: str,
        crop_size: int = 224,
        single_image: Optional[PIL.Image.Image] = None,
        num_validation_samples: Optional[int] = None,
        filepaths: Optional[list] = None,
        cond_type: Optional[str] = None,
        prompt_embeds_path: Optional[str] = None,
        gt_path: Optional[str] = None,
        examples: Optional[str] = None,
        sample_rate: Optional[float] = 1.0,
    ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.json_path = json_path
        self.root_dir = root_dir
        self.num_views = num_views
        self.img_wh = img_wh
        self.crop_size = crop_size
        self.bg_color = bg_color
        self.cond_type = cond_type
        self.gt_path = gt_path
        self.file_list = None

        self.image_fix_key = [
            "normals_006_front",
            "normals_006_front_right",
            "normals_006_right",
            "normals_006_back",
            "normals_006_left",
            "normals_006_front_left",
            "rgb_006_front",
            "rgb_006_front_right",
            "rgb_006_right",
            "rgb_006_back",
            "rgb_006_left",
            "rgb_006_front_left",
        ]

        self.view_list = self.image_fix_key[6:]

        with open(self.json_path, "r") as f:
            file_list = json.load(f)
        self.file_list = file_list

        self.all_images = []
        self.all_alphas = []
        self.bg_color = self.get_bg_color()

        try:
            self.normal_text_embeds = torch.load(
                f"{prompt_embeds_path}/normal_embeds.pt"
            )
            self.color_text_embeds = torch.load(
                f"{prompt_embeds_path}/clr_embeds.pt"
            )  # 4view
        except:
            self.color_text_embeds = torch.load(f"{prompt_embeds_path}/embeds.pt")
            self.normal_text_embeds = None

    def __len__(self):
        return len(self.file_list)

    def sample_view(self, examples=False):
        if examples:
            return self.view_list[0]
        else:
            if random.random() < 0.5:
                return self.view_list[0]
            else:
                return random.choice(self.view_list[1:])

    def get_bg_color(self):
        if self.bg_color == "white":
            bg_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        elif self.bg_color == "black":
            bg_color = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        elif self.bg_color == "gray":
            bg_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        elif self.bg_color == "random":
            bg_color = np.random.rand(3)
        elif isinstance(self.bg_color, float):
            bg_color = np.array([self.bg_color] * 3, dtype=np.float32)
        else:
            raise NotImplementedError
        return bg_color

    def load_image(self, img_path, bg_color, return_type="np"):

        img = np.array(Image.open(img_path).resize(self.img_wh))
        img = img.astype(np.float32) / 255.0  # [0, 1]
        assert img.shape[-1] == 4  # RGBA

        alpha = img[..., 3:4]
        img = img[..., :3] * alpha + bg_color * (1 - alpha)

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
        else:
            raise NotImplementedError

        return img, alpha

    def __getitem__(self, index):

        file = self.file_list[index]

        random_view = self.sample_view(examples=True)
        # image, _ = self.load_image(os.path.join(self.root_dir, file, random_view + ".webp"), self.bg_color, return_type='pt')
        # img_tensors_in = [
        #     image.permute(2, 0, 1)
        # ] * self.num_views
        # img_tensors_in = torch.stack(img_tensors_in, dim=0).float() # (Nv, 3, H, W)

        real_img_tensors_in = []
        for key in self.image_fix_key:
            real_image, _ = self.load_image(
                os.path.join(self.root_dir, file, key + ".webp"),
                self.bg_color,
                return_type="pt",
            )
            real_img_tensors_in.append(real_image.permute(2, 0, 1))

        real_img_tensors_in = torch.stack(
            real_img_tensors_in, dim=0
        ).float()  # (Nv, 3, H, W)

        normal_prompt_embeddings = (
            self.normal_text_embeds if hasattr(self, "normal_text_embeds") else None
        )
        color_prompt_embeddings = (
            self.color_text_embeds if hasattr(self, "color_text_embeds") else None
        )

        out = {
            # 'imgs_in': img_tensors_in,
            "real_imgs_in": real_img_tensors_in,
            "normal_prompt_embeddings": normal_prompt_embeddings,
            "color_prompt_embeddings": color_prompt_embeddings,
        }

        return out


class SingleImageDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        json_path: str,
        num_views: int,
        img_wh: Tuple[int, int],
        bg_color: str,
        crop_size: int = 224,
        single_image: Optional[PIL.Image.Image] = None,
        num_validation_samples: Optional[int] = None,
        filepaths: Optional[list] = None,
        cond_type: Optional[str] = None,
        prompt_embeds_path: Optional[str] = None,
        gt_path: Optional[str] = None,
        examples: Optional[str] = None,
        sample_rate: Optional[float] = 1.0,
    ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.json_path = json_path
        self.root_dir = root_dir
        self.num_views = num_views
        self.img_wh = img_wh
        self.crop_size = crop_size
        self.bg_color = bg_color
        self.cond_type = cond_type
        self.gt_path = gt_path
        self.view_list = [
            "rgb_006_front.webp",
            "rgb_006_front_right.webp",
            "rgb_006_left.webp",
            "rgb_006_right.webp",
            "rgb_006_front_left.webp",
            "rgb_006_back.webp",
        ]
        self.file_list = None

        # if single_image is None:
        #     if filepaths is None:
        #         # Get a list of all files in the directory
        #         file_list = os.listdir(self.root_dir)
        #     else:
        #         file_list = filepaths

        #     # Filter the files that end with .png or .jpg
        #     self.file_list = [file for file in file_list if file.endswith(('.png', '.jpg', '.webp'))]
        # else:
        #     self.file_list = None

        # load all images
        with open(self.json_path, "r") as f:
            file_list = json.load(f)

        sample_size = int(len(file_list) * sample_rate)
        self.file_list = random.sample(file_list, sample_size)
        # self.file_list = self.file_list[:6000]

        self.all_images = []
        self.all_alphas = []
        bg_color = self.get_bg_color()
        file_skip_count = 0
        if single_image is not None:
            image, alpha = self.load_image(
                None, bg_color, return_type="pt", Imagefile=single_image
            )
            self.all_images.append(image)
            self.all_alphas.append(alpha)
        else:
            for file in tqdm(self.file_list):
                # random sample a position from given list.
                random_view = self.sample_view(examples=examples)
                image, alpha = self.load_image(
                    os.path.join(self.root_dir, file, random_view),
                    bg_color,
                    return_type="pt",
                )
                # if image == None and alpha == None:
                #     file_skip_count += 1
                #     continue
                self.all_images.append(image)
                self.all_alphas.append(alpha)
        # print(f"[INFO] In total, {file_skip_count} files have been corrupted and dataloader skips these files.")

        # self.all_images = self.all_images[:num_validation_samples]
        # self.all_alphas = self.all_alphas[:num_validation_samples]
        # ic(len(self.all_images))

        try:
            self.normal_text_embeds = torch.load(
                f"{prompt_embeds_path}/normal_embeds.pt"
            )
            self.color_text_embeds = torch.load(
                f"{prompt_embeds_path}/clr_embeds.pt"
            )  # 4view
        except:
            self.color_text_embeds = torch.load(f"{prompt_embeds_path}/embeds.pt")
            self.normal_text_embeds = None

    def __len__(self):
        return len(self.all_images)

    def sample_view(self, examples=False):
        if examples:
            return self.view_list[0]
        else:
            if random.random() < 0.5:
                return self.view_list[0]
            else:
                return random.choice(self.view_list[1:])

    def get_bg_color(self):
        if self.bg_color == "white":
            bg_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        elif self.bg_color == "black":
            bg_color = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        elif self.bg_color == "gray":
            bg_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        elif self.bg_color == "random":
            bg_color = np.random.rand(3)
        elif isinstance(self.bg_color, float):
            bg_color = np.array([self.bg_color] * 3, dtype=np.float32)
        else:
            raise NotImplementedError
        return bg_color

    def load_image(self, img_path, bg_color, return_type="np", Imagefile=None):
        # pil always returns uint8
        if Imagefile is None:
            image_input = Image.open(img_path)  # .convert('RGBA')
            # a = Image.open("/data1/kendong/Phased-Consistency-Model-Era3d/code/text_to_image_sd15/examples/3968940-PH.png")
        else:
            image_input = Imagefile
        image_size = self.img_wh[0]
        # print(np.array(image_input).shape, np.array(a).shape)

        # try:
        if self.crop_size != -1:
            alpha_np = np.asarray(image_input)[:, :, 3]
            coords = np.stack(np.nonzero(alpha_np), 1)[:, (1, 0)]
            min_x, min_y = np.min(coords, 0)
            max_x, max_y = np.max(coords, 0)
            ref_img_ = image_input.crop((min_x, min_y, max_x, max_y))
            h, w = ref_img_.height, ref_img_.width
            scale = self.crop_size / max(h, w)
            h_, w_ = int(scale * h), int(scale * w)
            ref_img_ = ref_img_.resize((w_, h_))
            image_input = add_margin(ref_img_, size=image_size)
        else:
            image_input = add_margin(
                image_input, size=max(image_input.height, image_input.width)
            )
            image_input = image_input.resize((image_size, image_size))
        # except:
        #     return None, None

        # img = scale_and_place_object(img, self.scale_ratio)
        img = np.array(image_input)
        img = img.astype(np.float32) / 255.0  # [0, 1]
        assert img.shape[-1] == 4  # RGBA

        alpha = img[..., 3:4]
        img = img[..., :3] * alpha + bg_color * (1 - alpha)

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
            alpha = torch.from_numpy(alpha)
        else:
            raise NotImplementedError

        return img, alpha

    def __getitem__(self, index):
        image = self.all_images[index % len(self.all_images)]
        alpha = self.all_alphas[index % len(self.all_images)]
        if self.file_list is not None:
            filename = self.file_list[
                index % len(self.all_images)
            ]  # .replace(".png", "")
        else:
            filename = "null"
        img_tensors_in = [image.permute(2, 0, 1)] * self.num_views

        alpha_tensors_in = [alpha.permute(2, 0, 1)] * self.num_views

        img_tensors_in = torch.stack(img_tensors_in, dim=0).float()  # (Nv, 3, H, W)
        alpha_tensors_in = torch.stack(alpha_tensors_in, dim=0).float()  # (Nv, 3, H, W)

        if self.gt_path is not None:
            gt_image = self.gt_images[index % len(self.all_images)]
            gt_alpha = self.gt_alpha[index % len(self.all_images)]
            gt_img_tensors_in = [gt_image.permute(2, 0, 1)] * self.num_views
            gt_alpha_tensors_in = [gt_alpha.permute(2, 0, 1)] * self.num_views
            gt_img_tensors_in = torch.stack(gt_img_tensors_in, dim=0).float()
            gt_alpha_tensors_in = torch.stack(gt_alpha_tensors_in, dim=0).float()

        normal_prompt_embeddings = (
            self.normal_text_embeds if hasattr(self, "normal_text_embeds") else None
        )
        color_prompt_embeddings = (
            self.color_text_embeds if hasattr(self, "color_text_embeds") else None
        )

        out = {
            "imgs_in": img_tensors_in,
            "alphas": alpha_tensors_in,
            "normal_prompt_embeddings": normal_prompt_embeddings,
            "color_prompt_embeddings": color_prompt_embeddings,
            "filename": filename,
        }

        return out


class SingleImageDatasetImagenet(Dataset):
    def __init__(
        self,
        root_dir: str,
        json_path_imagenet: str,
        json_path_unpair_real: str,
        num_views: int,
        img_wh: Tuple[int, int],
        bg_color: str,
        crop_size: int = 224,
        single_image: Optional[PIL.Image.Image] = None,
        num_validation_samples: Optional[int] = None,
        filepaths: Optional[list] = None,
        cond_type: Optional[str] = None,
        prompt_embeds_path: Optional[str] = None,
        gt_path: Optional[str] = None,
        examples: Optional[str] = None,
        sample_rate: Optional[float] = 1.0,
    ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.json_path_imagenet = json_path_imagenet
        self.json_path_unpair_real = json_path_unpair_real
        self.root_dir = root_dir
        self.num_views = num_views
        self.img_wh = img_wh
        self.crop_size = crop_size
        self.bg_color = bg_color
        self.cond_type = cond_type
        self.gt_path = gt_path
        self.file_list = None
        self.image_fix_key = [
            "normals_006_front",
            "normals_006_front_right",
            "normals_006_right",
            "normals_006_back",
            "normals_006_left",
            "normals_006_front_left",
            "rgb_006_front",
            "rgb_006_front_right",
            "rgb_006_right",
            "rgb_006_back",
            "rgb_006_left",
            "rgb_006_front_left",
        ]

        self.view_list = self.image_fix_key[6:]

        with open(self.json_path_imagenet, "r") as f:
            file_list_imagenet = json.load(f)
        self.file_list_imagenet = file_list_imagenet

        with open(self.json_path_unpair_real, "r") as f:
            file_list_unpair_real = json.load(f)
        self.file_list_unpair_real = file_list_unpair_real

        self.bg_color = self.get_bg_color()

        try:
            self.normal_text_embeds = torch.load(
                f"{prompt_embeds_path}/normal_embeds.pt"
            )
            self.color_text_embeds = torch.load(
                f"{prompt_embeds_path}/clr_embeds.pt"
            )  # 4view
        except:
            self.color_text_embeds = torch.load(f"{prompt_embeds_path}/embeds.pt")
            self.normal_text_embeds = None

    def __len__(self):
        return len(self.file_list_imagenet)

    def sample_view(self, examples=False):
        if examples:
            return self.view_list[0]
        else:
            if random.random() < 0.5:
                return self.view_list[0]
            else:
                return random.choice(self.view_list[1:])

    def get_bg_color(self):
        if self.bg_color == "white":
            bg_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        elif self.bg_color == "black":
            bg_color = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        elif self.bg_color == "gray":
            bg_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        elif self.bg_color == "random":
            bg_color = np.random.rand(3)
        elif isinstance(self.bg_color, float):
            bg_color = np.array([self.bg_color] * 3, dtype=np.float32)
        else:
            raise NotImplementedError
        return bg_color

    def load_image(self, img_path, bg_color, return_type="np"):

        img = np.array(Image.open(img_path).resize(self.img_wh))
        img = img.astype(np.float32) / 255.0  # [0, 1]
        assert img.shape[-1] == 4  # RGBA

        alpha = img[..., 3:4]
        img = img[..., :3] * alpha + bg_color * (1 - alpha)

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
        else:
            raise NotImplementedError

        return img, alpha

    def __getitem__(self, index):

        file_imagenet = self.file_list_imagenet[index]
        file_unpair_real = self.file_list_unpair_real[index]

        random_view = self.sample_view(examples=True)
        image_imagenet, _ = self.load_image(
            os.path.join(self.root_dir, file_imagenet, random_view + ".webp"),
            self.bg_color,
            return_type="pt",
        )

        img_tensors_in = [image_imagenet.permute(2, 0, 1)] * self.num_views
        img_tensors_in = torch.stack(img_tensors_in, dim=0).float()  # (Nv, 3, H, W)

        real_img_tensors_in = []
        for key in self.image_fix_key:
            image_unpair_real, _ = self.load_image(
                os.path.join(self.root_dir, file_unpair_real, key + ".webp"),
                self.bg_color,
                return_type="pt",
            )
            real_img_tensors_in.append(image_unpair_real.permute(2, 0, 1))

        real_img_tensors_in = torch.stack(
            real_img_tensors_in, dim=0
        ).float()  # (Nv, 3, H, W)

        normal_prompt_embeddings = (
            self.normal_text_embeds if hasattr(self, "normal_text_embeds") else None
        )
        color_prompt_embeddings = (
            self.color_text_embeds if hasattr(self, "color_text_embeds") else None
        )

        out = {
            "filename": file_imagenet,
            "imgs_in": img_tensors_in,
            "real_imgs_in": real_img_tensors_in,
            "normal_prompt_embeddings": normal_prompt_embeddings,
            "color_prompt_embeddings": color_prompt_embeddings,
        }

        return out


class SingleImageDatasetObjaverse(Dataset):
    def __init__(
        self,
        root_dir: str,
        json_path: str,
        num_views: int,
        img_wh: Tuple[int, int],
        bg_color: str,
        crop_size: int = 224,
        single_image: Optional[PIL.Image.Image] = None,
        num_validation_samples: Optional[int] = None,
        filepaths: Optional[list] = None,
        cond_type: Optional[str] = None,
        prompt_embeds_path: Optional[str] = None,
        gt_path: Optional[str] = None,
        examples: Optional[str] = None,
        sample_rate: Optional[float] = 1.0,
    ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.json_path = json_path
        self.root_dir = root_dir
        self.num_views = num_views
        self.img_wh = img_wh
        self.crop_size = crop_size
        self.bg_color = bg_color
        self.cond_type = cond_type
        self.gt_path = gt_path
        self.file_list = None

        self.image_fix_key = [
            "normals_006_front",
            "normals_006_front_right",
            "normals_006_right",
            "normals_006_back",
            "normals_006_left",
            "normals_006_front_left",
            "rgb_006_front",
            "rgb_006_front_right",
            "rgb_006_right",
            "rgb_006_back",
            "rgb_006_left",
            "rgb_006_front_left",
        ]

        self.view_list = self.image_fix_key[6:]

        with open(self.json_path, "r") as f:
            file_list = json.load(f)
        self.file_list = file_list

        self.bg_color = self.get_bg_color()

        try:
            self.normal_text_embeds = torch.load(
                f"{prompt_embeds_path}/normal_embeds.pt"
            )
            self.color_text_embeds = torch.load(
                f"{prompt_embeds_path}/clr_embeds.pt"
            )  # 4view
        except:
            self.color_text_embeds = torch.load(f"{prompt_embeds_path}/embeds.pt")
            self.normal_text_embeds = None

    def __len__(self):
        return len(self.file_list)

    def sample_view(self, examples=False):
        if examples:
            return self.view_list[0]
        else:
            if random.random() < 0.5:
                return self.view_list[0]
            else:
                return random.choice(self.view_list[1:])

    def get_bg_color(self):
        if self.bg_color == "white":
            bg_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        elif self.bg_color == "black":
            bg_color = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        elif self.bg_color == "gray":
            bg_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        elif self.bg_color == "random":
            bg_color = np.random.rand(3)
        elif isinstance(self.bg_color, float):
            bg_color = np.array([self.bg_color] * 3, dtype=np.float32)
        else:
            raise NotImplementedError
        return bg_color

    def load_image(self, img_path, bg_color, return_type="np"):

        img = np.array(Image.open(img_path).resize(self.img_wh))
        img = img.astype(np.float32) / 255.0  # [0, 1]
        assert img.shape[-1] == 4  # RGBA

        alpha = img[..., 3:4]
        img = img[..., :3] * alpha + bg_color * (1 - alpha)

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
        else:
            raise NotImplementedError

        return img, alpha

    def __getitem__(self, index):

        file = self.file_list[index]

        random_view = self.sample_view(examples=True)
        image, _ = self.load_image(
            os.path.join(self.root_dir, file, random_view + ".webp"),
            self.bg_color,
            return_type="pt",
        )
        img_tensors_in = [image.permute(2, 0, 1)] * self.num_views
        img_tensors_in = torch.stack(img_tensors_in, dim=0).float()  # (Nv, 3, H, W)

        real_img_tensors_in = []
        for key in self.image_fix_key:
            real_image, _ = self.load_image(
                os.path.join(self.root_dir, file, key + ".webp"),
                self.bg_color,
                return_type="pt",
            )
            real_img_tensors_in.append(real_image.permute(2, 0, 1))

        real_img_tensors_in = torch.stack(
            real_img_tensors_in, dim=0
        ).float()  # (Nv, 3, H, W)

        normal_prompt_embeddings = (
            self.normal_text_embeds if hasattr(self, "normal_text_embeds") else None
        )
        color_prompt_embeddings = (
            self.color_text_embeds if hasattr(self, "color_text_embeds") else None
        )

        out = {
            "filename": file,
            "imgs_in": img_tensors_in,
            "real_imgs_in": real_img_tensors_in,
            "normal_prompt_embeddings": normal_prompt_embeddings,
            "color_prompt_embeddings": color_prompt_embeddings,
        }

        return out


class SingleImageDatasetGAN_Normal(Dataset):
    def __init__(
        self,
        root_dir: str,
        json_path: str,
        num_views: int,
        img_wh: Tuple[int, int],
        bg_color: str,
        crop_size: int = 224,
        single_image: Optional[PIL.Image.Image] = None,
        num_validation_samples: Optional[int] = None,
        filepaths: Optional[list] = None,
        cond_type: Optional[str] = None,
        prompt_embeds_path: Optional[str] = None,
        gt_path: Optional[str] = None,
        examples: Optional[str] = None,
        sample_rate: Optional[float] = 1.0,
    ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.json_path = json_path
        self.root_dir = root_dir
        self.num_views = num_views
        self.img_wh = img_wh
        self.crop_size = crop_size
        self.bg_color = bg_color
        self.cond_type = cond_type
        self.gt_path = gt_path
        self.file_list = None

        self.view_types = [
            "front",
            "front_right",
            "right",
            "back",
            "left",
            "front_left",
        ]
        self.fix_cam_pose_dir = "./mvdiffusion/data/fixed_poses/nine_views"
        self.fix_cam_poses = self.load_fixed_poses()  # world2cam matrix

        with open(self.json_path, "r") as f:
            file_list = json.load(f)
        self.file_list = file_list
        # sample_size = int(len(file_list) * sample_rate)
        # self.file_list = random.sample(file_list, sample_size)

        self.all_images = []
        self.all_alphas = []
        self.bg_color = self.get_bg_color()
        # self.folder_list = os.listdir(self.root_dir)
        # self.image_fix_key = ["normals_006_front",
        #                       "normals_006_front_right",
        #                       "normals_006_right",
        #                       "normals_006_back",
        #                       "normals_006_left",
        #                       "normals_006_front_left",
        #                       "rgb_006_front",
        #                       "rgb_006_front_right",
        #                       "rgb_006_right",
        #                       "rgb_006_back",
        #                       "rgb_006_left",
        #                       "rgb_006_front_left"]
        self.image_fix_key = [
            "normals_000_front",
            "normals_000_front_right",
            "normals_000_right",
            "normals_000_back",
            "normals_000_left",
            "normals_000_front_left",
            "rgb_000_front",
            "rgb_000_front_right",
            "rgb_000_right",
            "rgb_000_back",
            "rgb_000_left",
            "rgb_000_front_left",
        ]

        self.view_list = self.image_fix_key[6:]

        try:
            self.normal_text_embeds = torch.load(
                f"{prompt_embeds_path}/normal_embeds.pt"
            )
            self.color_text_embeds = torch.load(
                f"{prompt_embeds_path}/clr_embeds.pt"
            )  # 4view
        except:
            self.color_text_embeds = torch.load(f"{prompt_embeds_path}/embeds.pt")
            self.normal_text_embeds = None

    def __len__(self):
        return len(self.file_list)

    def load_fixed_poses(self):
        poses = {}
        for face in self.view_types:
            RT = np.loadtxt(
                os.path.join(self.fix_cam_pose_dir, "%03d_%s_RT.txt" % (0, face))
            )
            poses[face] = RT

        return poses

    def sample_view(self, examples=False):
        if examples:
            return self.view_list[0]
        else:
            if random.random() < 0.5:
                return self.view_list[0]
            else:
                return random.choice(self.view_list[1:])

    def get_bg_color(self):
        if self.bg_color == "white":
            bg_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        elif self.bg_color == "black":
            bg_color = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        elif self.bg_color == "gray":
            bg_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        elif self.bg_color == "random":
            bg_color = np.random.rand(3)
        elif isinstance(self.bg_color, float):
            bg_color = np.array([self.bg_color] * 3, dtype=np.float32)
        else:
            raise NotImplementedError
        return bg_color

    def load_image(self, img_path, bg_color, return_type="np"):
        # not using cv2 as may load in uint16 format
        # img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # [0, 255]
        # img = cv2.resize(img, self.img_wh, interpolation=cv2.INTER_CUBIC)
        # pil always returns uint8
        img = np.array(Image.open(img_path).resize(self.img_wh))
        img = img.astype(np.float32) / 255.0  # [0, 1]
        assert img.shape[-1] == 4  # RGBA

        alpha = img[..., 3:4]
        img = img[..., :3] * alpha + bg_color * (1 - alpha)

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
        else:
            raise NotImplementedError

        return img, alpha

    def load_normal(
        self,
        img_path,
        bg_color,
        alpha=None,
        RT_w2c=None,
        RT_w2c_cond=None,
        return_type="np",
    ):
        # not using cv2 as may load in uint16 format
        # img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # [0, 255]
        # img = cv2.resize(img, self.img_wh, interpolation=cv2.INTER_CUBIC)
        # pil always returns uint8
        normal = np.array(Image.open(img_path).resize(self.img_wh))

        assert normal.shape[-1] == 3 or normal.shape[-1] == 4  # RGB or RGBA

        if alpha is None and normal.shape[-1] == 4:
            alpha = normal[:, :, 3:] / 255.0
            normal = normal[:, :, :3]

        normal = trans_normal(img2normal(normal), RT_w2c, RT_w2c_cond)

        img = (normal * 0.5 + 0.5).astype(np.float32)  # [0, 1]

        if alpha.shape[-1] != 1:
            alpha = alpha[:, :, None]

        img = img[..., :3] * alpha + bg_color * (1 - alpha)

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
        else:
            raise NotImplementedError

        return img

    def __getitem__(self, index):

        file = self.file_list[index]
        cond_view = "front"
        cond_w2c = self.fix_cam_poses[cond_view]

        tgt_w2cs = [self.fix_cam_poses[view] for view in self.view_types]

        random_view = self.sample_view(examples=True)
        image, _ = self.load_image(
            os.path.join(self.root_dir, file, random_view + ".webp"),
            self.bg_color,
            return_type="pt",
        )
        img_tensors_in = [image.permute(2, 0, 1)] * self.num_views
        img_tensors_in = torch.stack(img_tensors_in, dim=0).float()  # (Nv, 3, H, W)

        real_img_tensors_in = []
        # for key in self.image_fix_key:
        for key, tgt_w2c in zip(self.image_fix_key[:6], tgt_w2cs):
            real_normal = self.load_normal(
                img_path=os.path.join(self.root_dir, file, key + ".webp"),
                bg_color=self.bg_color,
                RT_w2c=tgt_w2c,
                RT_w2c_cond=cond_w2c,
                return_type="pt",
            )

            real_img_tensors_in.append(real_normal.permute(2, 0, 1))

        for key, tgt_w2c in zip(self.image_fix_key[6:], tgt_w2cs):
            real_image, _ = self.load_image(
                os.path.join(self.root_dir, file, key + ".webp"),
                self.bg_color,
                return_type="pt",
            )
            real_img_tensors_in.append(real_image.permute(2, 0, 1))

        real_img_tensors_in = torch.stack(
            real_img_tensors_in, dim=0
        ).float()  # (Nv, 3, H, W)

        normal_prompt_embeddings = (
            self.normal_text_embeds if hasattr(self, "normal_text_embeds") else None
        )
        color_prompt_embeddings = (
            self.color_text_embeds if hasattr(self, "color_text_embeds") else None
        )

        out = {
            "imgs_in": img_tensors_in,
            "real_imgs_in": real_img_tensors_in,
            "normal_prompt_embeddings": normal_prompt_embeddings,
            "color_prompt_embeddings": color_prompt_embeddings,
        }

        return out


class SingleImageDataset_Wonder3d(Dataset):
    def __init__(
        self,
        root_dir: str,
        json_path: str,
        num_views: int,
        img_wh: Tuple[int, int],
        bg_color: str,
        crop_size: int = 224,
        single_image: Optional[PIL.Image.Image] = None,
        num_validation_samples: Optional[int] = None,
        filepaths: Optional[list] = None,
        prompt_embeds_path: Optional[str] = None,
        cond_type: Optional[str] = None,
    ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = root_dir
        self.json_path = json_path
        self.num_views = num_views
        self.img_wh = img_wh
        self.crop_size = crop_size

        self.cond_type = cond_type

        if self.num_views == 4:
            self.view_types = ["front", "right", "back", "left"]
        elif self.num_views == 5:
            self.view_types = ["front", "front_right", "right", "back", "left"]
        elif self.num_views == 6:
            self.view_types = [
                "front",
                "front_right",
                "right",
                "back",
                "left",
                "front_left",
            ]

        self.fix_cam_pose_dir = "./mvdiffusion/data/fixed_poses/nine_views"

        self.fix_cam_poses = self.load_fixed_poses()  # world2cam matrix

        with open(self.json_path, "r") as f:
            file_list = json.load(f)
        self.file_list = file_list

        # load all images
        self.all_images = []
        self.all_alphas = []
        self.bg_color = self.get_bg_color(bg_color)
        self.image_fix_key = [
            "normals_000_front",
            "normals_000_front_right",
            "normals_000_right",
            "normals_000_back",
            "normals_000_left",
            "normals_000_front_left",
            "rgb_000_front",
            "rgb_000_front_right",
            "rgb_000_right",
            "rgb_000_back",
            "rgb_000_left",
            "rgb_000_front_left",
        ]

        self.view_list = self.image_fix_key[6:]
        # if single_image is not None:
        #     image, alpha = self.load_image(None, bg_color, return_type='pt', Imagefile=single_image)
        #     self.all_images.append(image)
        #     self.all_alphas.append(alpha)
        # else:
        #     for file in self.file_list:
        #         # print(os.path.join(self.root_dir, file, "color_front_masked.png"))
        #         try:
        #             image, alpha = self.load_image(os.path.join(self.root_dir, file, "color_front_masked.png"), bg_color, return_type='pt')
        #         except:
        #             image, alpha = self.load_image(os.path.join(self.root_dir, file, "rgb_000_front.webp"), bg_color, return_type='pt')

        #         self.all_images.append(image)
        #         self.all_alphas.append(alpha)

        # self.all_images = self.all_images[:num_validation_samples]
        # self.all_alphas = self.all_alphas[:num_validation_samples]

    def __len__(self):
        return len(self.file_list)

    def load_fixed_poses(self):
        poses = {}
        for face in self.view_types:
            RT = np.loadtxt(
                os.path.join(self.fix_cam_pose_dir, "%03d_%s_RT.txt" % (0, face))
            )
            poses[face] = RT

        return poses

    def cartesian_to_spherical(self, xyz):
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
        z = np.sqrt(xy + xyz[:, 2] ** 2)
        theta = np.arctan2(
            np.sqrt(xy), xyz[:, 2]
        )  # for elevation angle defined from Z-axis down
        # ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
        return np.array([theta, azimuth, z])

    def get_T(self, target_RT, cond_RT):
        R, T = target_RT[:3, :3], target_RT[:, -1]
        T_target = -R.T @ T  # change to cam2world

        R, T = cond_RT[:3, :3], cond_RT[:, -1]
        T_cond = -R.T @ T

        theta_cond, azimuth_cond, z_cond = self.cartesian_to_spherical(T_cond[None, :])
        theta_target, azimuth_target, z_target = self.cartesian_to_spherical(
            T_target[None, :]
        )

        d_theta = theta_target - theta_cond
        d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
        d_z = z_target - z_cond

        # d_T = torch.tensor([d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), d_z.item()])
        return d_theta, d_azimuth

    def get_bg_color(self, bg_color):
        if bg_color == "white":
            ret = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        elif bg_color == "black":
            ret = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        elif bg_color == "gray":
            ret = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        elif bg_color == "random":
            ret = np.random.rand(3)
        elif isinstance(self.bg_color, float):
            ret = np.array([self.bg_color] * 3, dtype=np.float32)
        else:
            raise NotImplementedError
        return ret

    def load_image(self, img_path, bg_color, return_type="np", Imagefile=None):
        # pil always returns uint8
        img = np.array(Image.open(img_path).resize(self.img_wh))
        img = img.astype(np.float32) / 255.0  # [0, 1]
        assert img.shape[-1] == 4  # RGBA

        alpha = img[..., 3:4]
        img = img[..., :3] * alpha + bg_color * (1 - alpha)

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
        else:
            raise NotImplementedError

        return img, alpha

    def load_normal(
        self,
        img_path,
        bg_color,
        alpha=None,
        RT_w2c=None,
        RT_w2c_cond=None,
        return_type="np",
    ):
        # not using cv2 as may load in uint16 format
        # img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # [0, 255]
        # img = cv2.resize(img, self.img_wh, interpolation=cv2.INTER_CUBIC)
        # pil always returns uint8
        normal = np.array(Image.open(img_path).resize(self.img_wh))

        assert normal.shape[-1] == 3 or normal.shape[-1] == 4  # RGB or RGBA

        if alpha is None and normal.shape[-1] == 4:
            alpha = normal[:, :, 3:] / 255.0
            normal = normal[:, :, :3]

        normal = trans_normal(img2normal(normal), RT_w2c, RT_w2c_cond)

        img = (normal * 0.5 + 0.5).astype(np.float32)  # [0, 1]

        if alpha.shape[-1] != 1:
            alpha = alpha[:, :, None]

        img = img[..., :3] * alpha + bg_color * (1 - alpha)

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
        else:
            raise NotImplementedError

        return img

    def __getitem__(self, index):

        file = self.file_list[index]
        cond_view = "front"
        cond_w2c = self.fix_cam_poses[cond_view]

        tgt_w2cs = [self.fix_cam_poses[view] for view in self.view_types]

        elevations = []
        azimuths = []
        image, _ = self.load_image(
            os.path.join(self.root_dir, file, self.view_list[0] + ".webp"),
            self.bg_color,
            return_type="pt",
        )
        img_tensors_in = [image.permute(2, 0, 1)] * self.num_views
        img_tensors_in = torch.stack(img_tensors_in, dim=0).float()

        real_img_tensors_in = []
        for key, tgt_w2c in zip(self.image_fix_key[:6], tgt_w2cs):
            # evelations, azimuths
            real_normal = self.load_normal(
                img_path=os.path.join(self.root_dir, file, key + ".webp"),
                bg_color=self.bg_color,
                RT_w2c=tgt_w2c,
                RT_w2c_cond=cond_w2c,
                return_type="pt",
            )
            real_img_tensors_in.append(real_normal.permute(2, 0, 1))
            elevation, azimuth = self.get_T(tgt_w2c, cond_w2c)
            elevations.append(elevation)
            azimuths.append(azimuth)

        for key, tgt_w2c in zip(self.image_fix_key[6:], tgt_w2cs):
            real_image, _ = self.load_image(
                os.path.join(self.root_dir, file, key + ".webp"),
                self.bg_color,
                return_type="pt",
            )
            real_img_tensors_in.append(real_image.permute(2, 0, 1))

        real_img_tensors_in = torch.stack(
            real_img_tensors_in, dim=0
        ).float()  # (Nv, 3, H, W)

        elevations = torch.as_tensor(elevations).float().squeeze(1)
        azimuths = torch.as_tensor(azimuths).float().squeeze(1)
        elevations_cond = torch.as_tensor([0] * self.num_views).float()

        normal_class = torch.tensor([1, 0]).float()
        normal_task_embeddings = torch.stack(
            [normal_class] * self.num_views, dim=0
        )  # (Nv, 2)
        color_class = torch.tensor([0, 1]).float()
        color_task_embeddings = torch.stack(
            [color_class] * self.num_views, dim=0
        )  # (Nv, 2)

        camera_embeddings = torch.stack(
            [elevations_cond, elevations, azimuths], dim=-1
        )  # (Nv, 3)

        out = {
            "elevations_cond": elevations_cond,
            "elevations_cond_deg": torch.rad2deg(elevations_cond),
            "elevations": elevations,
            "azimuths": azimuths,
            "elevations_deg": torch.rad2deg(elevations),
            "azimuths_deg": torch.rad2deg(azimuths),
            "imgs_in": img_tensors_in,
            "camera_embeddings": camera_embeddings,
            "real_imgs_in": real_img_tensors_in,
            "normal_task_embeddings": normal_task_embeddings,
            "color_task_embeddings": color_task_embeddings,
        }

        return out


class SingleImageDatasetGenerator_wonder3d(Dataset):
    def __init__(
        self,
        root_dir: str,
        json_path: str,
        num_views: int,
        img_wh: Tuple[int, int],
        bg_color: str,
        crop_size: int = 224,
        single_image: Optional[PIL.Image.Image] = None,
        num_validation_samples: Optional[int] = None,
        filepaths: Optional[list] = None,
        prompt_embeds_path: Optional[str] = None,
        cond_type: Optional[str] = None,
    ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = root_dir
        self.json_path = json_path
        self.num_views = num_views
        self.img_wh = img_wh
        self.crop_size = crop_size

        self.cond_type = cond_type

        if self.num_views == 4:
            self.view_types = ["front", "right", "back", "left"]
        elif self.num_views == 5:
            self.view_types = ["front", "front_right", "right", "back", "left"]
        elif self.num_views == 6:
            self.view_types = [
                "front",
                "front_right",
                "right",
                "back",
                "left",
                "front_left",
            ]

        self.fix_cam_pose_dir = "./mvdiffusion/data/fixed_poses/nine_views"

        self.fix_cam_poses = self.load_fixed_poses()  # world2cam matrix

        with open(self.json_path, "r") as f:
            file_list = json.load(f)
        self.file_list = file_list

        # load all images
        self.all_images = []
        self.all_alphas = []
        self.bg_color = self.get_bg_color(bg_color)
        self.image_fix_key = [
            "normals_000_front",
            "normals_000_front_right",
            "normals_000_right",
            "normals_000_back",
            "normals_000_left",
            "normals_000_front_left",
            "rgb_000_front",
            "rgb_000_front_right",
            "rgb_000_right",
            "rgb_000_back",
            "rgb_000_left",
            "rgb_000_front_left",
        ]

        self.view_list = self.image_fix_key[6:]

    def __len__(self):
        return len(self.file_list)

    def load_fixed_poses(self):
        poses = {}
        for face in self.view_types:
            RT = np.loadtxt(
                os.path.join(self.fix_cam_pose_dir, "%03d_%s_RT.txt" % (0, face))
            )
            poses[face] = RT

        return poses

    def cartesian_to_spherical(self, xyz):
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
        z = np.sqrt(xy + xyz[:, 2] ** 2)
        theta = np.arctan2(
            np.sqrt(xy), xyz[:, 2]
        )  # for elevation angle defined from Z-axis down
        # ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
        return np.array([theta, azimuth, z])

    def get_T(self, target_RT, cond_RT):
        R, T = target_RT[:3, :3], target_RT[:, -1]
        T_target = -R.T @ T  # change to cam2world

        R, T = cond_RT[:3, :3], cond_RT[:, -1]
        T_cond = -R.T @ T

        theta_cond, azimuth_cond, z_cond = self.cartesian_to_spherical(T_cond[None, :])
        theta_target, azimuth_target, z_target = self.cartesian_to_spherical(
            T_target[None, :]
        )

        d_theta = theta_target - theta_cond
        d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
        d_z = z_target - z_cond

        # d_T = torch.tensor([d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), d_z.item()])
        return d_theta, d_azimuth

    def get_bg_color(self, bg_color):
        if bg_color == "white":
            ret = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        elif bg_color == "black":
            ret = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        elif bg_color == "gray":
            ret = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        elif bg_color == "random":
            ret = np.random.rand(3)
        elif isinstance(self.bg_color, float):
            ret = np.array([self.bg_color] * 3, dtype=np.float32)
        else:
            raise NotImplementedError
        return ret

    def load_image(self, img_path, bg_color, return_type="np", Imagefile=None):
        # pil always returns uint8
        img = np.array(Image.open(img_path).resize(self.img_wh))
        img = img.astype(np.float32) / 255.0  # [0, 1]
        assert img.shape[-1] == 4  # RGBA

        alpha = img[..., 3:4]
        img = img[..., :3] * alpha + bg_color * (1 - alpha)

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
        else:
            raise NotImplementedError

        return img, alpha

    def load_normal(
        self,
        img_path,
        bg_color,
        alpha=None,
        RT_w2c=None,
        RT_w2c_cond=None,
        return_type="np",
    ):
        # not using cv2 as may load in uint16 format
        # img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # [0, 255]
        # img = cv2.resize(img, self.img_wh, interpolation=cv2.INTER_CUBIC)
        # pil always returns uint8
        normal = np.array(Image.open(img_path).resize(self.img_wh))

        assert normal.shape[-1] == 3 or normal.shape[-1] == 4  # RGB or RGBA

        if alpha is None and normal.shape[-1] == 4:
            alpha = normal[:, :, 3:] / 255.0
            normal = normal[:, :, :3]

        normal = trans_normal(img2normal(normal), RT_w2c, RT_w2c_cond)

        img = (normal * 0.5 + 0.5).astype(np.float32)  # [0, 1]

        if alpha.shape[-1] != 1:
            alpha = alpha[:, :, None]

        img = img[..., :3] * alpha + bg_color * (1 - alpha)

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
        else:
            raise NotImplementedError

        return img

    def __getitem__(self, index):

        file = self.file_list[index]
        cond_view = "front"
        cond_w2c = self.fix_cam_poses[cond_view]

        tgt_w2cs = [self.fix_cam_poses[view] for view in self.view_types]

        elevations = []
        azimuths = []
        image, _ = self.load_image(
            os.path.join(self.root_dir, file, self.view_list[0] + ".webp"),
            self.bg_color,
            return_type="pt",
        )
        img_tensors_in = [image.permute(2, 0, 1)] * self.num_views
        img_tensors_in = torch.stack(img_tensors_in, dim=0).float()

        for key, tgt_w2c in zip(self.image_fix_key[:6], tgt_w2cs):
            # evelations, azimuths
            elevation, azimuth = self.get_T(tgt_w2c, cond_w2c)
            elevations.append(elevation)
            azimuths.append(azimuth)

        elevations = torch.as_tensor(elevations).float().squeeze(1)
        azimuths = torch.as_tensor(azimuths).float().squeeze(1)
        elevations_cond = torch.as_tensor([0] * self.num_views).float()

        normal_class = torch.tensor([1, 0]).float()
        normal_task_embeddings = torch.stack(
            [normal_class] * self.num_views, dim=0
        )  # (Nv, 2)
        color_class = torch.tensor([0, 1]).float()
        color_task_embeddings = torch.stack(
            [color_class] * self.num_views, dim=0
        )  # (Nv, 2)

        camera_embeddings = torch.stack(
            [elevations_cond, elevations, azimuths], dim=-1
        )  # (Nv, 3)

        out = {
            "elevations_cond": elevations_cond,
            "elevations_cond_deg": torch.rad2deg(elevations_cond),
            "elevations": elevations,
            "azimuths": azimuths,
            "elevations_deg": torch.rad2deg(elevations),
            "azimuths_deg": torch.rad2deg(azimuths),
            "imgs_in": img_tensors_in,
            "camera_embeddings": camera_embeddings,
            "normal_task_embeddings": normal_task_embeddings,
            "color_task_embeddings": color_task_embeddings,
        }

        return out
