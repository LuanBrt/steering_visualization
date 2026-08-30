#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import StableDiffusionXLPipeline
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

        self.template, dummy_pixels = self._build_prefix_template()
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

    def _pil_to_tensor(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        x = torch.from_numpy(__import__("numpy").array(image)).float() / 255.0
        x = x.permute(2, 0, 1).unsqueeze(0).to(self.device)
        x = F.interpolate(
            x,
            size=(self.model_image_h, self.model_image_w),
            mode="bilinear",
            align_corners=False,
        )
        x = (x - self.image_mean) / self.image_std
        return x.to(self.dtype)

    @torch.no_grad()
    def score(self, image: Image.Image, target_sentence: str):
        prefix = {key: value.clone() for key, value in self.template.items()}
        prefix["pixel_values"] = self._pil_to_tensor(image)

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
        if prefix_token_types is None:
            prefix_token_types = torch.zeros_like(prefix_ids)

        target_attention = torch.ones_like(target_ids)
        target_token_types = torch.zeros_like(target_ids)

        input_ids = torch.cat([prefix_ids, target_ids], dim=1)
        attention_mask = torch.cat([prefix_attention, target_attention], dim=1)
        token_type_ids = torch.cat([prefix_token_types, target_token_types], dim=1)

        labels = torch.full_like(input_ids, -100)
        labels[:, prefix_ids.shape[1]:] = target_ids

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=prefix["pixel_values"],
            labels=labels,
            use_cache=False,
            return_dict=True,
        )

        nll = float(outputs.loss.item())
        return {
            "nll": nll,
            "log_p": -nll,
            "log10_p": -nll / math.log(10.0),
            "target_len": int(target_ids.shape[1]),
        }


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
    parser = argparse.ArgumentParser(description="Best-of-N SDXL samples ranked by Gemma sentence likelihood")
    parser.add_argument("--sentence", required=True)
    parser.add_argument("--instruction", default="Describe this image in one sentence.")
    parser.add_argument("--prompt", default="A photograph")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--gemma-model", default=DEFAULT_GEMMA_MODEL)
    parser.add_argument("--diffusion-model", default=DEFAULT_DIFFUSION_MODEL)
    parser.add_argument("--gemma-device", default="cuda:0")
    parser.add_argument("--diffusion-device", default="cuda:1")
    parser.add_argument("--gemma-dtype", default="bfloat16")
    parser.add_argument("--diffusion-dtype", default="float16")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--diffusion-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="output_bestofn")
    args = parser.parse_args()

    out_dir = Path(args.out)
    all_dir = out_dir / "all"
    top_dir = out_dir / "top"
    all_dir.mkdir(parents=True, exist_ok=True)
    top_dir.mkdir(parents=True, exist_ok=True)

    diffusion_dtype = parse_dtype(args.diffusion_dtype)
    gemma_dtype = parse_dtype(args.gemma_dtype)

    print(f"Loading diffusion model {args.diffusion_model} on {args.diffusion_device}...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.diffusion_model,
        torch_dtype=diffusion_dtype,
        variant="fp16" if diffusion_dtype == torch.float16 else None,
        use_safetensors=True,
    ).to(args.diffusion_device)
    pipe.set_progress_bar_config(disable=True)

    scorer = GemmaSentenceScorer(
        model_id=args.gemma_model,
        device=args.gemma_device,
        dtype=gemma_dtype,
        instruction=args.instruction,
    )

    rows = []
    generated = 0
    while generated < args.num_samples:
        current_bs = min(args.batch_size, args.num_samples - generated)
        generators = [
            torch.Generator(device=args.diffusion_device).manual_seed(args.seed + generated + i)
            for i in range(current_bs)
        ]

        with torch.no_grad():
            images = pipe(
                prompt=[args.prompt] * current_bs,
                negative_prompt=[args.negative_prompt] * current_bs,
                num_inference_steps=args.diffusion_steps,
                guidance_scale=args.guidance_scale,
                height=args.height,
                width=args.width,
                generator=generators,
            ).images

        for i, image in enumerate(images):
            index = generated + i
            seed = args.seed + index
            filename = all_dir / f"sample_{index:04d}_seed_{seed}.png"
            image.save(filename)

            metrics = scorer.score(image, args.sentence)
            row = {
                "index": index,
                "seed": seed,
                "path": str(filename),
                **metrics,
            }
            rows.append(row)
            print(
                f"sample {index + 1:04d}/{args.num_samples}"
                f" | seed={seed}"
                f" | nll={metrics['nll']:.6f}"
                f" | log10_p={metrics['log10_p']:.3f}"
            )

        generated += current_bs

    rows.sort(key=lambda x: x["nll"])

    csv_path = out_dir / "scores.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "index", "seed", "path", "nll", "log_p", "log10_p", "target_len"])
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow({"rank": rank, **row})

    top_k = min(args.top_k, len(rows))
    for rank, row in enumerate(rows[:top_k], start=1):
        image = Image.open(row["path"]).convert("RGB")
        image.save(
            top_dir / f"rank_{rank:02d}_nll_{row['nll']:.4f}_seed_{row['seed']}.png"
        )

    print("\n=== Top results ===")
    for rank, row in enumerate(rows[:top_k], start=1):
        print(
            f"#{rank:02d}"
            f" | nll={row['nll']:.6f}"
            f" | log10_p={row['log10_p']:.3f}"
            f" | seed={row['seed']}"
            f" | {row['path']}"
        )

    print(f"\nSaved scores to: {csv_path}")
    print(f"Saved top-{top_k} images to: {top_dir}")


if __name__ == "__main__":
    main()
