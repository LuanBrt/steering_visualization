#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import DDIMScheduler, StableDiffusionXLPipeline
from torchvision.utils import save_image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration


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

        print(f"Gemma image input size: {self.model_image_h}x{self.model_image_w}")

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

        prefix = {key: value.clone() for key, value in self.image_inputs_template.items()}
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
        return nll, {
            "target_len": target_len,
            "geom_token_p": torch.exp(-nll.detach()),
        }


class SDXLMPGD:
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
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

        self.pipe.unet.requires_grad_(False).eval()
        self.pipe.vae.requires_grad_(False).eval()
        self.pipe.text_encoder.requires_grad_(False).eval()
        self.pipe.text_encoder_2.requires_grad_(False).eval()

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

        self.pipe.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        self.timesteps = self.pipe.scheduler.timesteps
        self.latent_shape = self._latent_shape(height, width)

        print(f"Latent shape: {self.latent_shape}")
        print(f"Diffusion timesteps: {len(self.timesteps)}")

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
            dtype=self.dtype,
        )
        latents = latents * self.pipe.scheduler.init_noise_sigma
        return latents

    def predict_noise(self, latents: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        if self.guidance_scale > 1.0:
            latent_model_input = torch.cat([latents, latents], dim=0)
        else:
            latent_model_input = latents

        latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, timestep)

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
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
        return noise_pred

    def pred_x0(self, latents: torch.Tensor, noise_pred: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        alpha_prod_t = self.pipe.scheduler.alphas_cumprod[timestep].to(latents.device, latents.dtype)
        beta_prod_t = 1 - alpha_prod_t
        while alpha_prod_t.ndim < latents.ndim:
            alpha_prod_t = alpha_prod_t.unsqueeze(-1)
            beta_prod_t = beta_prod_t.unsqueeze(-1)
        pred_x0 = (latents - beta_prod_t.sqrt() * noise_pred) / alpha_prod_t.sqrt()
        return pred_x0

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents / self.pipe.vae.config.scaling_factor
        image = self.pipe.vae.decode(latents, return_dict=False)[0]
        image = (image * 0.5 + 0.5).clamp(0.0, 1.0)
        return image

    def denoise_step(self, latents: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            noise_pred = self.predict_noise(latents, timestep)
            latents = self.pipe.scheduler.step(
                noise_pred,
                timestep,
                latents,
                return_dict=True,
            ).prev_sample
        return latents


def parse_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def per_sample_l2_norm(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(1).norm(dim=1)


def main():
    parser = argparse.ArgumentParser(description="Gemma + SDXL MPGD-style optimization")
    parser.add_argument("--sentence", default="Yes", help="Target sentence whose probability under Gemma should be maximized")
    parser.add_argument("--instruction", default="Is there a pirate in the image?")
    parser.add_argument("--prompt", default="A person", help="Diffusion prompt")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--gemma-model", default=DEFAULT_GEMMA_MODEL)
    parser.add_argument("--diffusion-model", default=DEFAULT_DIFFUSION_MODEL)
    parser.add_argument("--gemma-device", default="cuda:0")
    parser.add_argument("--diffusion-device", default="cuda:1")
    parser.add_argument("--gemma-dtype", default="float32")
    parser.add_argument("--diffusion-dtype", default="fp16")
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--diffusion-steps", type=int, default=60)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--mpgd-scale", type=float, default=0.2)
    parser.add_argument("--recurrent-steps", type=int, default=2)
    parser.add_argument("--guide-start", type=int, default=2, help="first denoising step index that receives Gemma guidance")
    parser.add_argument("--guide-end", type=int, default=-3, help="last denoising step index (inclusive if >=0, relative from end if <0)")
    parser.add_argument("--grad-clip", type=float, default=0.0, help="optional clip on latent gradient norm; 0 disables")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--out", default="output_juggernaut_xl_mpgd")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    gemma_dtype = parse_dtype(args.gemma_dtype)
    diffusion_dtype = parse_dtype(args.diffusion_dtype)

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
            f"{str(diffusion_dtype).removeprefix('torch.')} for stable local guidance."
        )

    diffusion = SDXLMPGD(
        model_id=args.diffusion_model,
        device=args.diffusion_device,
        dtype=diffusion_dtype,
        prompt=args.prompt,
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

    latents = diffusion.initial_latent(args.seed)
    initial_latents = latents.detach().clone()

    guide_end = args.guide_end
    if guide_end < 0:
        guide_end = len(diffusion.timesteps) + guide_end
    guide_end = max(0, min(len(diffusion.timesteps) - 1, guide_end))
    guide_start = max(0, min(len(diffusion.timesteps) - 1, args.guide_start))

    print("\n=== Starting MPGD-style denoising ===")
    print(f"sentence         = {args.sentence!r}")
    print(f"instruction      = {args.instruction!r}")
    print(f"prompt           = {args.prompt!r}")
    print(f"guidance_scale   = {args.guidance_scale}")
    print(f"diffusion_steps  = {args.diffusion_steps}")
    print(f"mpgd_scale       = {args.mpgd_scale}")
    print(f"recurrent_steps  = {args.recurrent_steps}")
    print(f"guide window     = [{guide_start}, {guide_end}]")
    print(f"grad_clip        = {args.grad_clip}")
    print(f"output dir       = {out_dir}")
    print("=====================================\n")

    metrics_rows = []
    best_nll = float("inf")
    best_preview = None
    best_step = None
    best_recur = None

    with torch.no_grad():
        noise0 = diffusion.predict_noise(latents, diffusion.timesteps[0])
        x0_0 = diffusion.pred_x0(latents, noise0, diffusion.timesteps[0])
        img0 = diffusion.decode_latents(x0_0)
        save_image(img0.cpu(), out_dir / "preview_step_000.png")

    for step_idx, timestep in enumerate(diffusion.timesteps):
        should_guide = guide_start <= step_idx <= guide_end

        if should_guide:
            for recur_idx in range(args.recurrent_steps):
                latents = latents.detach().requires_grad_(True)

                noise_pred = diffusion.predict_noise(latents, timestep)
                pred_x0 = diffusion.pred_x0(latents, noise_pred, timestep)
                image = diffusion.decode_latents(pred_x0)
                image_for_gemma = image.to(scorer.device, dtype=torch.float32)
                nll, diagnostics = scorer.sentence_nll(image_for_gemma, args.sentence)

                grad = torch.autograd.grad(nll, latents)[0]
                raw_grad_norm = per_sample_l2_norm(grad).mean().item()
                if args.grad_clip > 0:
                    grad_norm = per_sample_l2_norm(grad).clamp_min(1e-8)
                    clip_scale = torch.clamp(args.grad_clip / grad_norm, max=1.0)
                    grad = grad * clip_scale.view(-1, 1, 1, 1)

                grad_norm = per_sample_l2_norm(grad).clamp_min(1e-8)
                grad_unit = grad / grad_norm.view(-1, 1, 1, 1)

                alpha_prod_t = diffusion.pipe.scheduler.alphas_cumprod[timestep].to(latents.device, latents.dtype)
                sigma_t = (1 - alpha_prod_t).sqrt().item()
                step_scale = args.mpgd_scale * max(sigma_t, 1e-4)

                with torch.no_grad():
                    latents_next = latents - step_scale * grad_unit
                    dz = latents_next - latents
                    dz_norm = per_sample_l2_norm(dz).mean().item()
                    dimg_mean = 0.0
                    dimg_max = 0.0
                    if best_nll == float("inf"):
                        pass
                    latents = latents_next.detach()

                nll_value = nll.item()
                metrics_rows.append({
                    "timestep_index": step_idx,
                    "timestep": int(timestep.item()) if hasattr(timestep, "item") else int(timestep),
                    "recur_index": recur_idx,
                    "guided": 1,
                    "nll": nll_value,
                    "geom_token_p": diagnostics["geom_token_p"].item(),
                    "target_len": diagnostics["target_len"],
                    "raw_grad_norm": raw_grad_norm,
                    "dz_norm": dz_norm,
                    "sigma_t": sigma_t,
                    "step_scale": step_scale,
                })

                if nll_value < best_nll:
                    best_nll = nll_value
                    best_preview = image.detach().cpu()
                    best_step = step_idx
                    best_recur = recur_idx

                print(
                    f"denoise {step_idx + 1:03d}/{len(diffusion.timesteps)}"
                    f" | recur={recur_idx + 1}/{args.recurrent_steps}"
                    f" | t={int(timestep)}"
                    f" | nll={nll_value:.6f}"
                    f" | geom_token_p={diagnostics['geom_token_p'].item():.9f}"
                    f" | grad_norm={raw_grad_norm:.3e}"
                    f" | dz_norm={dz_norm:.3e}"
                    f" | sigma_t={sigma_t:.3e}"
                    f" | step_scale={step_scale:.3e}"
                )

        latents = diffusion.denoise_step(latents, timestep)

        if args.save_every > 0 and ((step_idx + 1) % args.save_every == 0 or step_idx == len(diffusion.timesteps) - 1):
            with torch.no_grad():
                next_probe_t = diffusion.timesteps[min(step_idx + 1, len(diffusion.timesteps) - 1)]
                noise_pred = diffusion.predict_noise(latents, next_probe_t)
                pred_x0 = diffusion.pred_x0(latents, noise_pred, next_probe_t)
                preview = diffusion.decode_latents(pred_x0)
                save_image(preview.cpu(), out_dir / f"preview_step_{step_idx + 1:03d}.png")

    with torch.no_grad():
        final_image = diffusion.decode_latents(latents)
        noise_l2 = per_sample_l2_norm(latents - initial_latents).mean().item()
    save_image(final_image.cpu(), out_dir / "final.png")

    final_nll, final_diag = scorer.sentence_nll(
        final_image.to(scorer.device, dtype=torch.float32),
        args.sentence,
    )

    if best_preview is not None:
        save_image(best_preview, out_dir / "best_preview.png")

    with open(out_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestep_index",
                "timestep",
                "recur_index",
                "guided",
                "nll",
                "geom_token_p",
                "target_len",
                "raw_grad_norm",
                "dz_norm",
                "sigma_t",
                "step_scale",
            ],
        )
        writer.writeheader()
        writer.writerows(metrics_rows)

    print("\n=== Final result ===")
    print(f"final_nll        = {final_nll.item():.6f}")
    print(f"geom_token_p     = {final_diag['geom_token_p'].item():.9f}")
    print(f"noise_delta_l2   = {noise_l2:.6f}")
    if best_step is not None:
        print(f"best_preview_nll = {best_nll:.6f} at denoise step {best_step + 1}, recur {best_recur + 1}")
        print(f"saved best image = {out_dir / 'best_preview.png'}")
    print(f"saved final image= {out_dir / 'final.png'}")
    print(f"saved metrics    = {out_dir / 'metrics.csv'}")
    print("====================")


if __name__ == "__main__":
    main()
