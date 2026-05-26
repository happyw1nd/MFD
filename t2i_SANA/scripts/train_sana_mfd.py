#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "diffusers @ git+https://github.com/huggingface/diffusers.git",
#     "torch>=2.0.0",
#     "accelerate>=1.0.0",
#     "transformers>=4.47.0",
#     "ftfy",
#     "tensorboard",
#     "Jinja2",
#     "peft>=0.14.0",
#     "sentencepiece",
#     "torchvision",
#     "datasets",
#     "bitsandbytes",
#     "prodigyopt",
# ]
# ///

import argparse
import copy
import itertools
import logging
import math
import os
import random
import shutil
import warnings
from pathlib import Path
import lpips

import numpy as np
import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch.utils.checkpoint
from torch.func import jvp, functional_call
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import AutoTokenizer, Gemma2Model
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple, Union

import diffusers
from diffusers import (
    AutoencoderDC,
    FlowMatchEulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
    SanaPipeline,
    SanaTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _collate_lora_metadata,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_torch_npu_available
from diffusers.utils.torch_utils import is_compiled_module


if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.36.0.dev0")

logger = get_logger(__name__)

if is_torch_npu_available():
    torch.npu.config.allow_internal_format = False


def to_m11(img):
    # img: 0..1 -> -1..1
    return img * 2.0 - 1.0

def to_01(img):
    # img: -1..1 -> 0..1
    return (img + 1.0) / 2.0

def latent_to_pixel(latents, vae):
    latents = latents / vae.config.scaling_factor
    image = vae.decode(latents, return_dict=False)[0]
    return image

def prep_for_lpips(x, target_size=256):
    if x.min() >= 0.0:            # 0..1 -> -1..1
        x = to_m11(x)
    if x.shape[-1] != target_size or x.shape[-2] != target_size:
        x = F.interpolate(x, size=(target_size, target_size),
                          mode='bilinear', align_corners=False)
    return x

def generate_latents_with_student(accelerator, transformer, scheduler, latents, prompt_embeds, prompt_attention_mask, uncond_prompt_embeds, uncond_prompt_attention_mask, timesteps=None, guidance_scale=7.0, num_inference_steps=1):
    if timesteps is None:
        scheduler.set_timesteps(num_inference_steps)
    else:
        scheduler.set_timesteps(num_inference_steps, timesteps=[timesteps])
    timesteps = scheduler.timesteps
    bz = latents.shape[0]

    accelerator.unwrap_model(transformer).set_adapters(["default"])

    if guidance_scale > 1.0:
        _prompt_embeds = torch.cat([uncond_prompt_embeds, prompt_embeds], dim=0)
        _pooled_prompt_mask = torch.cat([uncond_prompt_attention_mask, prompt_attention_mask], dim=0)
    else:
        _prompt_embeds = prompt_embeds
        _pooled_prompt_mask = prompt_attention_mask

    interval_results = []
    for i, t in enumerate(timesteps):
        if guidance_scale > 1.0:
            latent_model_input = torch.cat([latents] * 2)
            timestep = torch.tensor([t]*bz).view(bz,).repeat(2)
        else:
            latent_model_input = latents
            timestep = torch.tensor([t]*bz).view(bz,)

        timestep = (timestep * accelerator.unwrap_model(transformer).config.timestep_scale).to(latent_model_input.device)

        noise_pred = transformer(
            hidden_states=latent_model_input,
            encoder_hidden_states=_prompt_embeds,
            encoder_attention_mask=_pooled_prompt_mask,
            timestep=timestep,
            return_dict=False,
        )[0]

        if guidance_scale > 1.0:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        interval_result = (latents - noise_pred * (timestep/1000.0)).clone()
        interval_results.append(interval_result)
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        

    return latents, interval_results

def generate_latents_with_teacher(accelerator, transformer, scheduler, latents, prompt_embeds, prompt_attention_mask, uncond_prompt_embeds, uncond_prompt_attention_mask, timesteps=None, guidance_scale=7.0, num_inference_steps=1):
    if timesteps is None:
        scheduler.set_timesteps(num_inference_steps)
    else:
        scheduler.set_timesteps(num_inference_steps, timesteps=[timesteps])
    timesteps = scheduler.timesteps
    bz = latents.shape[0]

    accelerator.unwrap_model(transformer).set_adapters([])

    if guidance_scale > 1.0:
        _prompt_embeds = torch.cat([uncond_prompt_embeds, prompt_embeds], dim=0)
        _pooled_prompt_mask = torch.cat([uncond_prompt_attention_mask, prompt_attention_mask], dim=0)
    else:
        _prompt_embeds = prompt_embeds
        _pooled_prompt_mask = prompt_attention_mask

    with torch.no_grad():
        for i, t in enumerate(timesteps):
            if guidance_scale > 1.0:
                latent_model_input = torch.cat([latents] * 2)
                timestep = torch.tensor([t]*bz).view(bz,).repeat(2)
            else:
                latent_model_input = latents
                timestep = torch.tensor([t]*bz).view(bz,)
            timestep = (timestep * accelerator.unwrap_model(transformer).config.timestep_scale).to(latent_model_input.device)

            noise_pred = transformer(
                hidden_states=latent_model_input,
                encoder_hidden_states=_prompt_embeds,
                encoder_attention_mask=_pooled_prompt_mask,
                timestep=timestep,
                return_dict=False,
            )[0]

            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    return latents


def sample_t_r(batch_size, device, dist_type='lognorm', r_not_equal_t_ratio=0.25, min_interval=0.0, dtype=torch.float32):

    if min_interval < 0.0 or min_interval >= 1.0:
        raise ValueError("min_interval must be in [0.0, 1.0)")

    def _sample_base(n):
        if dist_type == 'uniform':
            return torch.rand(n, 1, device=device, dtype=dtype)
        elif dist_type == 'lognorm':
            mu, sigma = -0.4, 1.0
            s = torch.randn(n, 1, device=device, dtype=dtype) * sigma + mu
            return torch.sigmoid(s)
        else:
            raise ValueError("Unsupported distribution type")

    s1 = _sample_base(batch_size)
    s2 = _sample_base(batch_size)

    # Generate mask for samples that require r != t
    mask_neq = torch.rand(batch_size, 1, device=device, dtype=dtype) < r_not_equal_t_ratio

    # Rejection sampling: ensure samples with r != t satisfy the minimum interval
    if min_interval > 0.0:
        # Select samples marked for r!=t whose initial interval is insufficient
        invalid_mask = mask_neq & (torch.abs(s1 - s2) < min_interval)
        
        max_iters = 100  # Safety threshold to prevent infinite loop when min_interval is too large
        iters = 0
        
        while invalid_mask.any():
            if iters >= max_iters:
                raise RuntimeError(f"Rejection sampling did not converge after {max_iters} iterations. "
                                   f"Check if min_interval ({min_interval}) is too large.")
            
            num_invalid = invalid_mask.sum().item()
            
            # Resample only invalid samples
            new_s1 = _sample_base(num_invalid)
            new_s2 = _sample_base(num_invalid)
            
            # Scatter back into the original tensors
            s1[invalid_mask] = new_s1.flatten()
            s2[invalid_mask] = new_s2.flatten()
            
            # Re-evaluate invalid mask
            invalid_mask = mask_neq & (torch.abs(s1 - s2) < min_interval)
            iters += 1

    # Enforce t >= r
    t = torch.max(s1, s2)
    r = torch.min(s1, s2)

    # Handle r == t case
    # For samples not requiring r != t, force r = t
    r = torch.where(mask_neq, r, t)

    return t.view(-1), r.view(-1)

def sample_t(batch_size, device, dist_type='lognorm', dtype=torch.float32):
    
    # Sample a single scalar
    if dist_type == 'uniform':
        # Uniform distribution U(0, 1)
        s = torch.rand(batch_size, 1, device=device, dtype=dtype)
    elif dist_type == 'lognorm':
        # Logit-Normal distribution with default mu=-0.4, sigma=1.0
        mu, sigma = -0.4, 1.0
        s = torch.randn(batch_size, 1, device=device, dtype=dtype) * sigma + mu
        s = torch.sigmoid(s)
    else:
        raise ValueError("Unsupported distribution type")

    return s.view(-1)


def log_validation(
    pipeline,
    args,
    accelerator,
    pipeline_args,
    epoch,
    is_final_validation=False,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    if args.enable_vae_tiling:
        pipeline.vae.enable_tiling(tile_sample_min_height=1024, tile_sample_stride_width=1024)

    pipeline.text_encoder = pipeline.text_encoder.to(torch.bfloat16)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed is not None else None

    images = [pipeline(**pipeline_args, generator=generator).images[0] for _ in range(args.num_validation_images)]

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
                    ]
                }
            )

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return images


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) containing the training data of instance images (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--prompt_dataset",
        type=str,
        default=None,
        help=(
            "Path to diffusionDB prompt-only parquet database."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing the target image. By "
        "default, the standard Image Dataset maps out 'file_name' "
        "to 'image'.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default=None,
        help="The column of the dataset containing the instance prompt for each image",
    )

    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")

    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=300,
        help="Maximum sequence length to use with with the Gemma model",
    )
    parser.add_argument(
        "--complex_human_instruction",
        type=str,
        default=None,
        help="Instructions for complex human attention: https://github.com/NVlabs/Sana/blob/main/configs/sana_app_config/Sana_1600M_app.yaml#L55.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=10,
        help=(
            "Run validation every X steps."
        ),
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=4,
        help="LoRA alpha to be used for additional scaling.",
    )
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="Dropout probability for LoRA layers")
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sana-dreambooth-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate_stu",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--learning_rate_aux",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--ode_step_size",
        type=float,
        default=0.02,
        help="ODE step size for integration",
    )
    parser.add_argument(
        "--r_not_equal_t_ratio",
        type=float,
        default=1.0,
        help="1.0 -> MFD, 0.0 -> VSD",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--mf",
        type=str,
        default="iMF",
        choices=["iMF", "MF"],
        help=('type of flow matching to use. Choose between ["iMF", "MF"]'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodigy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_weight_decay_text_encoder", type=float, default=1e-03, help="Weight decay to use for text_encoder"
    )

    parser.add_argument(
        "--lora_layers",
        type=str,
        default=None,
        help=(
            'The transformer modules to apply LoRA training on. Please specify the layers in a comma separated. E.g. - "to_k,to_q,to_v" will result in lora training of attention layers only'
        ),
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--cache_latents",
        action="store_true",
        default=False,
        help="Cache the VAE latents",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--upcast_before_saving",
        action="store_true",
        default=False,
        help=(
            "Whether to upcast the trained transformer layers to float32 before saving (at the end of training). "
            "Defaults to precision dtype used for training to save memory"
        ),
    )
    parser.add_argument(
        "--offload",
        action="store_true",
        help="Whether to offload the VAE and the text encoder to CPU when they are not used.",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--enable_vae_tiling", action="store_true", help="Enabla vae tiling in log validation")
    parser.add_argument("--enable_npu_flash_attention", action="store_true", help="Enabla Flash Attention for NPU")
    parser.add_argument(
        "--val_resolution",
        type=int,
        default=512,
        help=(
            "val_resolution."
        ),
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=1,
        help=(
            "Inference steps for validation."
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warnings.warn("You need not use --class_data_dir without --with_prior_preservation.")
        if args.class_prompt is not None:
            warnings.warn("You need not use --class_prompt without --with_prior_preservation.")

    return args


def collate_fn(examples, with_prior_preservation=False):
    prompts = [example["prompt"] for example in examples]

    batch = {"prompts": prompts}
    return batch

    
class LAIONDataset_prompt_only(Dataset):
    def __init__(self, laion_path):
        self.laion_path = laion_path
        self.prompts = []

        with open(f'{laion_path}', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:

                parts = line.strip().split('\t')

                if len(parts) < 4:
                    continue
                image_name = parts[0]
                prompt = parts[1]
                try:
                    score = float(parts[2])
                except ValueError:
                    continue
                self.prompts.append(prompt)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        example = {}
        example["prompt"] = self.prompts[idx]
        return example


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `hf auth login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

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
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
            ).repo_id

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )

    # Load scheduler and models
    noise_scheduler = DPMSolverMultistepScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler", revision=args.revision
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    student_scheduler = copy.deepcopy(noise_scheduler)
    text_encoder = Gemma2Model.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderDC.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    transformer = SanaTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
    )

    # We only train the additional adapter LoRA layers
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    # VAE should always be kept in fp32 for SANA
    vae.to(accelerator.device, dtype=torch.float32)
    transformer.to(accelerator.device, dtype=weight_dtype)
    # Gemma2 is particularly suited for bfloat16
    text_encoder.to(dtype=torch.bfloat16)

    if args.enable_npu_flash_attention:
        if is_torch_npu_available():
            logger.info("npu flash attention enabled.")
            for block in transformer.transformer_blocks:
                block.attn2.set_use_npu_flash_attention(True)
        else:
            raise ValueError("npu flash attention requires torch_npu extensions and is supported only on npu device ")

    # Initialize a text encoding pipeline and keep it to CPU for now.
    text_encoding_pipeline = SanaPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=None,
        transformer=None,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
    )

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    if args.lora_layers is not None:
        target_modules = [layer.strip() for layer in args.lora_layers.split(",")]
    else:
        target_modules = ["to_k", "to_q", "to_v"]

    # now we will add new LoRA weights the transformer layers
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer.add_adapter(transformer_lora_config, adapter_name="default")
    transformer.add_adapter(transformer_lora_config, adapter_name="aux")
    transformer.set_adapters(["default", "aux"])

    # load lpips backbone
    lpips_net = lpips.LPIPS(net='vgg').to(accelerator.device)   # 'vgg' | 'squeeze' | 'alex'

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None
            modules_to_save = {}
            for model in models:
                if isinstance(model, type(unwrap_model(transformer))):
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model, adapter_name="default")
                    modules_to_save["transformer"] = model
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            SanaPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
                **_collate_lora_metadata(modules_to_save),
            )

    def load_model_hook(models, input_dir):
        transformer_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(transformer))):
                transformer_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict = SanaPipeline.lora_state_dict(input_dir)

        transformer_state_dict = {
            f"{k.replace('transformer.', '')}": v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            models = [transformer_]
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate_stu = (
            args.learning_rate_stu * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
        args.learning_rate_aux = (
            args.learning_rate_aux * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [transformer]
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    # for param in transformer.aux_fuse_time_embed.parameters():
    #     param.requires_grad = True
    student_lora_parameters = [
        p for name, p in transformer.named_parameters() 
        if "default" in name and p.requires_grad
    ]
    aux_lora_parameters = [
        p for name, p in transformer.named_parameters() 
        if "aux" in name and p.requires_grad
    ]
    print(f"Number of trainable parameters in student LoRA: {sum(p.numel() for p in student_lora_parameters)}")
    print(f"Number of trainable parameters in aux LoRA: {sum(p.numel() for p in aux_lora_parameters)}")

    # Optimization parameters
    transformer_parameters_with_lr_stu = {"params": student_lora_parameters, "lr": args.learning_rate_stu}
    transformer_parameters_with_lr_aux = {"params": aux_lora_parameters, "lr": args.learning_rate_aux}
    params_to_optimize_stu = [transformer_parameters_with_lr_stu]
    params_to_optimize_aux = [transformer_parameters_with_lr_aux]

    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
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

        optimizer_stu = optimizer_class(
            params_to_optimize_stu,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

        optimizer_aux = optimizer_class(
            params_to_optimize_aux,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate_stu <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer_stu = optimizer_class(
            params_to_optimize_stu,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

        optimizer_aux = optimizer_class(
            params_to_optimize_aux,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    # Dataset and DataLoaders creation:
    train_dataset = LAIONDataset_prompt_only(laion_path=args.prompt_dataset)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
        num_workers=args.dataloader_num_workers,
    )

    def compute_text_embeddings(prompt, text_encoding_pipeline):
        text_encoding_pipeline = text_encoding_pipeline.to(accelerator.device)
        with torch.no_grad():
            prompt_embeds, prompt_attention_mask, _, _ = text_encoding_pipeline.encode_prompt(
                prompt,
                max_sequence_length=args.max_sequence_length,
                complex_human_instruction=args.complex_human_instruction,
            )
        # if args.offload:
        #     text_encoding_pipeline = text_encoding_pipeline.to("cpu")
        prompt_embeds = prompt_embeds.to(accelerator.unwrap_model(transformer).dtype)

        uncond_prompts = [""] * (len(prompt) if isinstance(prompt, (list, tuple)) else prompt.shape[0])
        with torch.no_grad():
            uncond_prompt_embeds, uncond_prompt_attention_mask, _, _ = text_encoding_pipeline.encode_prompt(
                uncond_prompts,
                max_sequence_length=args.max_sequence_length,
                complex_human_instruction=args.complex_human_instruction,
            )
        if args.offload:
            text_encoding_pipeline = text_encoding_pipeline.to("cpu")
        uncond_prompt_embeds = uncond_prompt_embeds.to(accelerator.unwrap_model(transformer).dtype)
        return prompt_embeds, prompt_attention_mask, uncond_prompt_embeds, uncond_prompt_attention_mask


    vae_config_scaling_factor = vae.config.scaling_factor

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler_stu = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_stu,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    lr_scheduler_aux = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_aux,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    transformer, optimizer_stu, optimizer_aux, train_dataloader, lr_scheduler_stu, lr_scheduler_aux = accelerator.prepare(
        transformer, optimizer_stu, optimizer_aux, train_dataloader, lr_scheduler_stu, lr_scheduler_aux
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "dreambooth-sana-lora"
        accelerator.init_trackers(tracker_name, config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
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

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    loss_stu = torch.tensor([0.0])
    loss_aux = torch.tensor([0.0])

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]
            with accelerator.accumulate(models_to_accumulate):
                if (global_step+1) % 10 != 0: # train aux
                
                    prompts = batch["prompts"]

                    prompt_embeds, prompt_attention_mask, uncond_prompt_embeds, uncond_prompt_attention_mask = compute_text_embeddings(prompts, text_encoding_pipeline)
                    prompt_embeds = prompt_embeds.to(dtype=weight_dtype)
                    uncond_prompt_embeds = uncond_prompt_embeds.to(dtype=weight_dtype)
                    prompt_attention_mask = prompt_attention_mask.to(dtype=weight_dtype)
                    uncond_prompt_attention_mask = uncond_prompt_attention_mask.to(dtype=weight_dtype)

                    # sample gaussian noise
                    noise_shape = (
                        args.train_batch_size,
                        accelerator.unwrap_model(transformer).config.in_channels,
                        args.val_resolution // (2 ** (len(vae.config.encoder_block_out_channels) - 1)),
                        args.val_resolution // (2 ** (len(vae.config.encoder_block_out_channels) - 1)),
                    )
                    noise = torch.randn(noise_shape, device=accelerator.device, dtype=weight_dtype)
                    bsz = noise.shape[0]

                    with torch.no_grad():
                        Z, _ = generate_latents_with_student(
                            accelerator, transformer, student_scheduler, noise, prompt_embeds, prompt_attention_mask, uncond_prompt_embeds, uncond_prompt_attention_mask, 
                            guidance_scale=1.0, 
                            num_inference_steps=args.num_inference_steps,
                        )

                    t = (sample_t(batch_size=bsz, device=accelerator.device, dtype=noise.dtype, dist_type='uniform') * 1000.0).round() / 1000.0

                    noise_for_Z = torch.randn(noise_shape, device=accelerator.device, dtype=torch.float32)
                    x_t = (1.0 - t) * Z + t * noise_for_Z

                    accelerator.unwrap_model(transformer).set_adapter("aux")
                    noise_pred = transformer(
                        hidden_states=x_t,
                        timestep=t * 1000.0,
                        encoder_hidden_states=prompt_embeds,
                        encoder_attention_mask=prompt_attention_mask,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred.float()

                    target = noise_for_Z - Z.float()
                    loss_aux = torch.nn.MSELoss()(noise_pred, target).float()

                    accelerator.backward(loss_aux)
                    if accelerator.sync_gradients:
                        params_to_clip = transformer.parameters()
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    optimizer_aux.step()
                    lr_scheduler_aux.step()
                    optimizer_aux.zero_grad()

                    

                else: # train student
                
                    prompts = batch["prompts"]

                    prompt_embeds, prompt_attention_mask, uncond_prompt_embeds, uncond_prompt_attention_mask = compute_text_embeddings(prompts, text_encoding_pipeline)
                    prompt_embeds = prompt_embeds.to(dtype=weight_dtype)
                    uncond_prompt_embeds = uncond_prompt_embeds.to(dtype=weight_dtype)
                    prompt_attention_mask = prompt_attention_mask.to(dtype=weight_dtype)
                    uncond_prompt_attention_mask = uncond_prompt_attention_mask.to(dtype=weight_dtype)

                    # sample gaussian noise
                    noise_shape = (
                        args.train_batch_size,
                        accelerator.unwrap_model(transformer).config.in_channels,
                        args.val_resolution // (2 ** (len(vae.config.encoder_block_out_channels) - 1)),
                        args.val_resolution // (2 ** (len(vae.config.encoder_block_out_channels) - 1)),
                    )
                    noise = torch.randn(noise_shape, device=accelerator.device, dtype=weight_dtype)
                    bsz = noise.shape[0]

                    Z, _ = generate_latents_with_student(
                        accelerator, transformer, student_scheduler, noise, prompt_embeds, prompt_attention_mask, uncond_prompt_embeds, uncond_prompt_attention_mask, 
                        guidance_scale=1.0, 
                        num_inference_steps=args.num_inference_steps,
                    )

                    with torch.no_grad():    
                        if accelerator.is_main_process:

                            # visualize Z
                            Z_decoded = latent_to_pixel(Z[:1, ...], vae)
                            Z_decoded = (Z_decoded / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).float().numpy()
                            for i in range(min(1, Z_decoded.shape[0])):
                                img = Image.fromarray((Z_decoded[i] * 255).round().astype("uint8"))
                                img.save(os.path.join("/data_storage/zju/tmp_cache/outputs/pics/tmp", f"Z_epoch{epoch}_step{step}_idx{i}.png"))

                    
                    t, r = sample_t_r(batch_size=args.train_batch_size, device=accelerator.device, dtype=weight_dtype, r_not_equal_t_ratio=args.r_not_equal_t_ratio, dist_type='uniform', min_interval=0.0)
                    t = (t * 1000.0).round() / 1000.0
                    r = (r * 1000.0).round() / 1000.0

                    noise_for_Z = torch.randn(noise_shape, device=accelerator.device, dtype=weight_dtype)
                    x_t = ((1.0 - t) * Z + t * noise_for_Z).to(dtype=weight_dtype)

                    with torch.no_grad():
                        accelerator.unwrap_model(transformer).set_adapter("aux")
                        if t.item() == r.item():
                            x_t_raw_input = x_t
                            x_t_input = x_t_raw_input
                            timesteps_input = t.to(x_t_input.device)
                            u_p = transformer(
                                hidden_states=x_t_input,
                                timestep=timesteps_input * 1000.0,
                                encoder_hidden_states=prompt_embeds,
                                encoder_attention_mask=prompt_attention_mask,
                                return_dict=False,
                            )[0]

                        else:
                            denoise_steps = math.ceil((t.item() - r.item()) / args.ode_step_size)
                            timesteps_intervals = torch.linspace(t.item(), r.item(), denoise_steps + 1)
                            # round all items in timesteps_intervals to interger
                            timesteps_intervals = (timesteps_intervals * 1000.0).round() / 1000.0
                            timestep_pairs = list(zip(timesteps_intervals[:-1], timesteps_intervals[1:]))

                            x_t_raw_input = x_t
                            for t_start, t_end in timestep_pairs:
                                t_start = torch.tensor([t_start], device=x_t.device, dtype=weight_dtype)
                                t_end = torch.tensor([t_end], device=x_t.device, dtype=weight_dtype)

                                x_t_input = x_t_raw_input
                                timesteps_input = t_start.to(x_t_input.device)
                                teacher_pred_noise = transformer(
                                    hidden_states=x_t_input,
                                    timestep=timesteps_input * 1000.0,
                                    encoder_hidden_states=prompt_embeds,
                                    encoder_attention_mask=prompt_attention_mask,
                                    return_dict=False,
                                )[0]

                                x_t_raw_input = x_t_raw_input - (t_start - t_end) * teacher_pred_noise

                            u_p = (x_t - x_t_raw_input) / (t - r)
                    
                    with torch.no_grad():
                        accelerator.unwrap_model(transformer).set_adapters([])
                        _prompt_embeds = torch.cat([uncond_prompt_embeds, prompt_embeds], dim=0)
                        _pooled_prompt_mask = torch.cat([uncond_prompt_attention_mask, prompt_attention_mask], dim=0)

                        assert t.shape[0] == 1, "only batch size 1 is supported for now."

                        if t.item() == r.item():
                            print(f'Teacher: timesteps_intervals - {t}')
                            x_t_raw_input = x_t
                            x_t_input = torch.cat([x_t_raw_input] * 2)
                            timesteps_input = torch.cat([t] * 2).to(x_t_input.device)
                            u_q = transformer(
                                hidden_states=x_t_input,
                                timestep=timesteps_input * 1000.0,
                                encoder_hidden_states=_prompt_embeds,
                                encoder_attention_mask=_pooled_prompt_mask,
                                return_dict=False,
                            )[0]
                            u_q_uncond, u_q_text = u_q.chunk(2)
                            u_q = u_q_uncond + 4.5 * (u_q_text - u_q_uncond)

                        else:
                            denoise_steps = math.ceil((t.item() - r.item()) / args.ode_step_size)
                            timesteps_intervals = torch.linspace(t.item(), r.item(), denoise_steps + 1)
                            # round all items in timesteps_intervals to interger
                            timesteps_intervals = (timesteps_intervals * 1000.0).round() / 1000.0
                            timestep_pairs = list(zip(timesteps_intervals[:-1], timesteps_intervals[1:]))

                            print(f'Teacher: timesteps_intervals - {timesteps_intervals}')

                            x_t_raw_input = x_t
                            for t_start, t_end in timestep_pairs:
                                t_start = torch.tensor([t_start], device=x_t.device, dtype=weight_dtype)
                                t_end = torch.tensor([t_end], device=x_t.device, dtype=weight_dtype)

                                x_t_input = torch.cat([x_t_raw_input] * 2)
                                timesteps_input = torch.cat([t_start] * 2).to(x_t_input.device)
                                teacher_pred_noise = transformer(
                                    hidden_states=x_t_input,
                                    timestep=timesteps_input * 1000.0,
                                    encoder_hidden_states=_prompt_embeds,
                                    encoder_attention_mask=_pooled_prompt_mask,
                                    return_dict=False,
                                )[0]
                                teacher_pred_noise_uncond, teacher_pred_noise_text = teacher_pred_noise.chunk(2)
                                teacher_pred_noise = teacher_pred_noise_uncond + 4.5 * (teacher_pred_noise_text - teacher_pred_noise_uncond)

                                x_t_raw_input = x_t_raw_input - (t_start - t_end) * teacher_pred_noise

                            u_q = (x_t - x_t_raw_input) / (t - r)

                   

                    # Compute regular loss.
                    dist_loss = ((u_q - u_p) * x_t).mean()

                    loss_stu = dist_loss 

                    accelerator.unwrap_model(transformer).set_adapter("default")
                    accelerator.backward(loss_stu)
                    if accelerator.sync_gradients:
                        params_to_clip = transformer.parameters()
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                    optimizer_stu.step()
                    lr_scheduler_stu.step()
                    optimizer_stu.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if args.validation_prompt is not None and (global_step-1) % args.validation_steps == 0:
                        # create pipeline
                        accelerator.unwrap_model(transformer).set_adapter("default")
                        pipeline = SanaPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            transformer=accelerator.unwrap_model(transformer),
                            revision=args.revision,
                            variant=args.variant,
                            torch_dtype=torch.float32,
                        )

                        pipeline_args = {
                            "prompt": args.validation_prompt,
                            "num_inference_steps": args.num_inference_steps, "guidance_scale": 1.0, 
                            "height": args.val_resolution, "width": args.val_resolution
                        }
                        images = log_validation(
                            pipeline=pipeline,
                            args=args,
                            accelerator=accelerator,
                            pipeline_args=pipeline_args,
                            epoch=epoch,
                        )
                        free_memory()

                        images = None
                        del pipeline

            logs = {"loss_stu": loss_stu.detach().item(), "loss_aux": loss_aux.detach().item(),"lr": lr_scheduler_stu.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if args.validation_prompt is not None and epoch % args.validation_epochs == 0:
                # create pipeline
                pipeline = SanaPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    transformer=accelerator.unwrap_model(transformer),
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=torch.float32,
                )
                pipeline_args = {
                    "prompt": args.validation_prompt,
                    "complex_human_instruction": args.complex_human_instruction,
                }
                images = log_validation(
                    pipeline=pipeline,
                    args=args,
                    accelerator=accelerator,
                    pipeline_args=pipeline_args,
                    epoch=epoch,
                )
                free_memory()

                images = None
                del pipeline

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer = unwrap_model(transformer)
        modules_to_save = {}
        if args.upcast_before_saving:
            transformer.to(torch.float32)
        else:
            transformer = transformer.to(weight_dtype)
        transformer_lora_layers = get_peft_model_state_dict(transformer)
        modules_to_save["transformer"] = transformer

        SanaPipeline.save_lora_weights(
            save_directory=args.output_dir,
            transformer_lora_layers=transformer_lora_layers,
            **_collate_lora_metadata(modules_to_save),
        )

        # Final inference
        # Load previous pipeline
        pipeline = SanaPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=torch.float32,
        )
        pipeline.transformer = pipeline.transformer.to(torch.float16)
        # load attention processors
        pipeline.load_lora_weights(args.output_dir)

        # run inference
        images = []
        if args.validation_prompt and args.num_validation_images > 0:
            pipeline_args = {
                "prompt": args.validation_prompt,
                "complex_human_instruction": args.complex_human_instruction,
            }
            images = log_validation(
                pipeline=pipeline,
                args=args,
                accelerator=accelerator,
                pipeline_args=pipeline_args,
                epoch=epoch,
                is_final_validation=True,
            )

        images = None
        del pipeline

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)