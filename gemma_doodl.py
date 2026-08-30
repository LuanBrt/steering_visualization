#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from diffusers import DDIMScheduler, StableDiffusionXLPipeline
from torchvision.utils import save_image


DEFAULT_GEMMA_MODEL = "google/gemma-3-4b-it"
DEFAULT_DIFFUSION_MODEL = "RunDiffusion/Juggernaut-XL-v9"


class GemmaSentenceScorer:
    def __init__(
        self,
        model_id: str,
        device: str,
        dtype: torch.dtype,
        instruction: str,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.instruction = instruction

        print(f"Loading Gemma teacher {model_id} on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer
        self.model = (
            Gemma3ForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=dtype,
            )
            .to(self.device)
            .eval()
        )
        self.model.requires_grad_(False)

        self.image_inputs_template, dummy_pixels = self._build_prefix_template()
        self.model_image_h = int(dummy_pixels.shape[-2])
        self.model_image_w = int(dummy_pixels.shape[-1])

        image_processor = self.processor.image_processor
        self.image_mean = torch.tensor(
            image_processor.image_mean,
            device=self.device,
            dtype=torch.float32,
        ).view(1, 3, 1, 1)
        self.image_std = torch.tensor(
            image_processor.image_std,
            device=self.device,
            dtype=torch.float32,
        ).view(1, 3, 1, 1)

        print(
            f"Gemma image input size: {self.model_image_h}x{self.model_image_w}"
        )

    def _build_prefix_template(self):
        dummy = Image.new("RGB", (448, 448), (128, 128, 128))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": dummy},
                    {"type": "text", "text": self.instruction},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            do_pan_and_scan=False,
        )

        dummy_pixels = inputs.pop("pixel_values")
        inputs.pop("num_crops", None)

        template = {
            key: value.to(self.device)
            for key, value in inputs.items()
            if torch.is_tensor(value)
        }

        return template, dummy_pixels

    def preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        image = F.interpolate(
            image,
            size=(self.model_image_h, self.model_image_w),
            mode="bilinear",
            align_corners=False,
        )
        image = image.clamp(0.0, 1.0)
        image = (image - self.image_mean) / self.image_std
        return image.to(self.device, dtype=self.dtype)

    def build_batch(self, image: torch.Tensor, target_sentence: str):
        image = self.preprocess_image(image)

        prefix = {
            key: value.clone()
            for key, value in self.image_inputs_template.items()
        }
        prefix["pixel_values"] = image

        target_ids = self.tokenizer(
            target_sentence,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(self.device)

        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None:
            eos = torch.tensor([[eos_id]], device=self.device, dtype=target_ids.dtype)
            target_ids = torch.cat([target_ids, eos], dim=1)

        batch_size = image.shape[0]
        if batch_size != 1:
            raise ValueError("This minimal script currently supports batch size 1 only.")

        prefix_ids = prefix["input_ids"]
        prefix_attention = prefix["attention_mask"]
        prefix_token_types = prefix.get("token_type_ids")

        target_attention = torch.ones_like(target_ids, device=self.device)
        if prefix_token_types is None:
            prefix_token_types = torch.zeros_like(prefix_ids, device=self.device)
        target_token_types = torch.zeros_like(target_ids, device=self.device)

        input_ids = torch.cat([prefix_ids, target_ids], dim=1)
        attention_mask = torch.cat([prefix_attention, target_attention], dim=1)
        token_type_ids = torch.cat([prefix_token_types, target_token_types], dim=1)

        labels = torch.full_like(input_ids, fill_value=-100)
        labels[:, prefix_ids.shape[1]:] = target_ids

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "pixel_values": prefix["pixel_values"],
            "labels": labels,
            "use_cache": False,
            "return_dict": True,
        }
        return inputs, target_ids.shape[1]

    def sentence_nll(self, image: torch.Tensor, target_sentence: str):
        inputs, target_len = self.build_batch(image, target_sentence)
        outputs = self.model(**inputs)
        nll = outputs.loss
        geom_token_p = torch.exp(-nll.detach())
        return nll, {
            "geom_token_p": geom_token_p,
            "target_len": target_len,
        }


class SDXLDOODL:
    def __init__(
        self,
        model_id: str,
        device: str,
        dtype: torch.dtype,
        prompt: str,
        negative_prompt: str,
        height: int,
        width: int,
        num_inference_steps: int,
        guidance_scale: float,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.height = height
        self.width = width
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale

        print(f"Loading SDXL model {model_id} on {self.device}...")
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            variant="fp16",
            use_safetensors=True,
        )
        # First-order DDIM has a substantially better-conditioned backward path
        # than multistep DPM++ for latent optimization.
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

        # DOODL optimizes only the starting latent. Freezing model weights avoids
        # allocating parameter gradients and makes the 1024px backward graph fit
        # comfortably on an 80 GB H100.
        self.pipe.unet.requires_grad_(False).eval()
        self.pipe.vae.requires_grad_(False).eval()
        self.pipe.text_encoder.requires_grad_(False).eval()
        self.pipe.text_encoder_2.requires_grad_(False).eval()

        if getattr(self.pipe.unet, "_supports_gradient_checkpointing", False):
            self.pipe.unet.enable_gradient_checkpointing()

        with torch.no_grad():
            encoded_prompt = self.pipe.encode_prompt(
                prompt=self.prompt,
                do_classifier_free_guidance=self.guidance_scale > 1.0,
                device=self.device,
                negative_prompt=self.negative_prompt,
                num_images_per_prompt=1,
            )

        (
            self.prompt_embeds,
            self.negative_prompt_embeds,
            self.pooled_prompt_embeds,
            self.negative_pooled_prompt_embeds,
        ) = tuple(value.detach() for value in encoded_prompt)

        if self.guidance_scale > 1.0:
            self.prompt_embeds = torch.cat([self.negative_prompt_embeds, self.prompt_embeds], dim=0)
            self.pooled_prompt_embeds = torch.cat(
                [self.negative_pooled_prompt_embeds, self.pooled_prompt_embeds], dim=0
            )

        projection_dim = self.pipe.text_encoder_2.config.projection_dim
        self.add_time_ids = self.pipe._get_add_time_ids(
            (self.height, self.width),
            (0, 0),
            (self.height, self.width),
            dtype=self.prompt_embeds.dtype,
            text_encoder_projection_dim=projection_dim,
        ).to(self.device)
        if self.guidance_scale > 1.0:
            self.add_time_ids = torch.cat([self.add_time_ids, self.add_time_ids], dim=0)

        # Free the huge text encoder after prompt encoding.
        del self.pipe.text_encoder
        del self.pipe.text_encoder_2
        del self.pipe.tokenizer
        del self.pipe.tokenizer_2
        self.pipe.text_encoder = None
        self.pipe.text_encoder_2 = None
        self.pipe.tokenizer = None
        self.pipe.tokenizer_2 = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        self.latent_shape = self._latent_shape(height, width)

        print(f"Latent shape: {self.latent_shape}")
        print(f"Diffusion timesteps: {self.num_inference_steps}")

    @staticmethod
    def _latent_shape(height: int, width: int):
        latent_h = math.ceil(height / 8)
        latent_w = math.ceil(width / 8)
        return (1, 4, latent_h, latent_w)

    def initial_latent(self, seed: int) -> torch.Tensor:
        generator = torch.Generator(device=self.device).manual_seed(seed)
        latents = torch.randn(
            self.latent_shape,
            generator=generator,
            device=self.device,
            # Keep the optimized parameter and optimizer state in fp32. Low
            # precision is reserved for the frozen diffusion forward/backward.
            dtype=torch.float32,
        )
        latents = latents * self.pipe.scheduler.init_noise_sigma
        return latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents / self.pipe.vae.config.scaling_factor
        image = self.pipe.vae.decode(latents, return_dict=False)[0]
        image = (image * 0.5 + 0.5).clamp(0.0, 1.0)
        return image

    def render(self, z_t: torch.Tensor) -> torch.Tensor:
        # The fp32 master latent keeps optimizer state numerically stable. The
        # differentiable cast lets the lower-precision diffusion model consume
        # it while gradients still flow back to z_t in fp32.
        latents = z_t.to(dtype=self.prompt_embeds.dtype)

        # Reset scheduler state for every independently rendered iteration.
        self.pipe.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        timesteps = self.pipe.scheduler.timesteps

        for timestep in timesteps:
            if self.guidance_scale > 1.0:
                latent_model_input = torch.cat([latents, latents], dim=0)
            else:
                latent_model_input = latents
            latent_model_input = self.pipe.scheduler.scale_model_input(
                latent_model_input, timestep
            )

            noise_pred = self.pipe.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=self.prompt_embeds,
                added_cond_kwargs={
                    "text_embeds": self.pooled_prompt_embeds,
                    "time_ids": self.add_time_ids,
                },
                return_dict=False,
            )[0]

            if self.guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            latents = self.pipe.scheduler.step(
                noise_pred,
                timestep,
                latents,
                return_dict=True,
            ).prev_sample

        return self.decode_latents(latents)


def parse_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def main():
    parser = argparse.ArgumentParser(description="Gemma + SDXL DOODL optimization")
    parser.add_argument("--sentence", required=True, help="Target sentence whose probability under Gemma should be maximized")
    parser.add_argument("--instruction", default="Describe this image in one word.")
    parser.add_argument(
        "--prompt",
        default="A photograph",
        help="Diffusion prompt",
    )
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--gemma-model", default=DEFAULT_GEMMA_MODEL)
    parser.add_argument("--diffusion-model", default=DEFAULT_DIFFUSION_MODEL)
    parser.add_argument("--gemma-device", default="cuda:0")
    parser.add_argument("--diffusion-device", default="cuda:1")
    parser.add_argument("--gemma-dtype", default="float32")
    parser.add_argument(
        "--diffusion-dtype",
        default="fp32",
        help=(
            "Diffusion compute dtype. BF16 is the safe default for differentiable "
            "denoising; FP16 is automatically promoted because its backward pass "
            "can overflow."
        ),
    )
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--diffusion-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--outer-steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--grad-clip", type=float, default=0.0, help="Optional global grad clip; 0 disables clipping")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--out", default="output_juggernaut_xl_doodl")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    gemma_dtype = parse_dtype(args.gemma_dtype)
    diffusion_dtype = parse_dtype(args.diffusion_dtype)

    # FP16 inference is fine, but backpropagating through the complete denoising
    # trajectory can overflow its small exponent range. Use BF16 where CUDA
    # supports it (Ampere and newer), otherwise fall back to FP32. Do this even
    # when an older launch command explicitly passes `--diffusion-dtype fp16`.
    if diffusion_dtype == torch.float16:
        diffusion_device = torch.device(args.diffusion_device)
        bf16_supported = (
            diffusion_device.type == "cuda"
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability(diffusion_device)[0] >= 8
        )
        diffusion_dtype = torch.bfloat16 if bf16_supported else torch.float32
        print(
            "Promoting diffusion dtype from float16 to "
            f"{str(diffusion_dtype).removeprefix('torch.')} for stable backpropagation."
        )

    diffusion_prompt = args.prompt 
    diffusion = SDXLDOODL(
        model_id=args.diffusion_model,
        device=args.diffusion_device,
        dtype=diffusion_dtype,
        prompt=diffusion_prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.diffusion_steps,
        guidance_scale=args.guidance_scale,
    )

    scorer = GemmaSentenceScorer(
        model_id=args.gemma_model,
        device=args.gemma_device,
        dtype=gemma_dtype,
        instruction=args.instruction,
    )

    z_t = torch.nn.Parameter(diffusion.initial_latent(args.seed))
    # Direct latent optimization is poorly conditioned with tiny globally-clipped SGD
    # steps. Adam is much closer to DOODL/DNO practice and gives useful coordinate-
    # wise steps even when the gradient arriving through Gemma is very small.
    optimizer = torch.optim.Adam([z_t], lr=args.lr, betas=(0.9, 0.999))

    print("\n=== Starting optimization ===")
    print(f"sentence         = {args.sentence!r}")
    print(f"instruction      = {args.instruction!r}")
    print(f"prompt           = {diffusion_prompt!r}")
    print(f"guidance_scale   = {args.guidance_scale}")
    print(f"diffusion_steps  = {args.diffusion_steps}")
    print(f"outer_steps      = {args.outer_steps}")
    print(f"lr               = {args.lr}")
    print(f"grad_clip        = {args.grad_clip}")
    print(f"output dir       = {out_dir}")
    print("=============================\n")

    with torch.no_grad():
        init_image = diffusion.render(z_t)
        previous_image = init_image.detach().clone()
        save_image(init_image.cpu(), out_dir / "step_0000.png")

    for step in range(args.outer_steps):
        optimizer.zero_grad(set_to_none=True)

        image = diffusion.render(z_t)
        if not torch.isfinite(image).all():
            raise FloatingPointError(
                f"Diffusion produced non-finite pixels at optimization step {step + 1}."
            )
        with torch.no_grad():
            image_delta_mean = (image.detach() - previous_image).float().abs().mean()
            image_delta_max = (image.detach() - previous_image).float().abs().max()
        image_for_gemma = image.to(scorer.device, dtype=torch.float32)
        nll, diagnostics = scorer.sentence_nll(image_for_gemma, args.sentence)
        loss = nll
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Gemma produced a non-finite NLL at optimization step {step + 1}."
            )
        loss.backward()

        if z_t.grad is None or not torch.isfinite(z_t.grad).all():
            raise FloatingPointError(
                f"Latent gradient became non-finite at optimization step {step + 1}."
            )

        grad_norm = z_t.grad.float().norm()
        grad_abs_mean = z_t.grad.float().abs().mean()
        grad_abs_max = z_t.grad.float().abs().max()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([z_t], max_norm=args.grad_clip)

        old_z = z_t.detach().clone()
        optimizer.step()
        latent_step_norm = (z_t.detach() - old_z).float().norm()
        latent_step_mean = (z_t.detach() - old_z).float().abs().mean()

        if not torch.isfinite(z_t).all():
            raise FloatingPointError(
                f"The optimizer produced a non-finite latent at optimization step {step + 1}."
            )

        if step == 0 or (step + 1) % 1 == 0 or step == args.outer_steps - 1:
            print(
                f"step {step + 1:03d}/{args.outer_steps}"
                f" | nll={loss.item():.9f}"
                f" | geom_token_p={diagnostics['geom_token_p'].item():.9f}"
                f" | target_len={diagnostics['target_len']}"
                f" | grad_norm={grad_norm.item():.3e}"
                f" | grad_mean={grad_abs_mean.item():.3e}"
                f" | grad_max={grad_abs_max.item():.3e}"
                f" | dz_norm={latent_step_norm.item():.3e}"
                f" | dz_mean={latent_step_mean.item():.3e}"
                f" | dimg_mean={image_delta_mean.item():.3e}"
                f" | dimg_max={image_delta_max.item():.3e}"
            )
        previous_image = image.detach().clone()

        if args.save_every > 0 and ((step + 1) % args.save_every == 0 or step == args.outer_steps - 1):
            with torch.no_grad():
                preview = diffusion.render(z_t)
                save_image(preview.cpu(), out_dir / f"step_{step + 1:04d}.png")

    # Keep gradients enabled for the final score so gradient checkpointing uses
    # the same numerical forward path as it did during optimization.
    final_image = diffusion.render(z_t)
    final_nll, final_diag = scorer.sentence_nll(
        final_image.to(scorer.device, dtype=torch.float32),
        args.sentence,
    )
    save_image(final_image.detach().cpu(), out_dir / "final.png")

    print("\n=== Final result ===")
    print(f"final_nll        = {final_nll.item():.6f}")
    print(f"geom_token_p     = {final_diag['geom_token_p'].item():.6f}")
    print(f"saved final image= {out_dir / 'final.png'}")
    print("====================")


if __name__ == "__main__":
    main()
