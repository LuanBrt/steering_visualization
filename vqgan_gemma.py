#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from diffusers import VQModel
from PIL import Image
from torchvision.utils import save_image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration


GEMMA_MODEL = "google/gemma-3-4b-it"
VQ_REPO = "CompVis/ldm-super-resolution-4x-openimages"
VQ_SUBFOLDER = "vqvae"

# Same broad baseline idea as the paper replication: subtract the average
# activation of unrelated words before aligning text and image representations.
BASELINE_WORDS: List[str] = [
    "desk", "jacket", "gondola", "laughter", "intelligence", "bicycle",
    "chair", "orchestra", "sand", "pottery", "arrowhead", "jewelry",
    "daffodil", "plateau", "estuary", "quilt", "moment", "bamboo",
    "ravine", "archive", "hieroglyph", "star", "clay", "fossil",
    "wildlife", "flour", "traffic", "bubble", "honey", "geode",
    "magnet", "ribbon", "zigzag", "puzzle", "tornado", "anthill",
    "galaxy", "poverty", "diamond", "universe", "vinegar", "nebula",
    "knowledge", "marble", "fog", "river", "scroll", "silhouette",
    "cake", "valley", "whisper", "pendulum", "tower", "table",
    "glacier", "whirlpool", "jungle", "wool", "anger", "rampart",
    "flower", "research", "hammer", "cloud", "justice", "dog",
    "butterfly", "needle", "fortress", "bonfire", "skyscraper",
    "caravan", "patience", "bacon", "velocity", "smoke", "electricity",
    "sunset", "anchor", "parchment", "courage", "statue", "oxygen",
    "time", "fabric", "pasta", "snowflake", "mountain", "echo",
    "piano", "sanctuary", "abyss", "air", "dewdrop", "garden",
    "literature", "rice", "enigma",
]


def parse_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(name)


def clamp_with_grad(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    clamped = x.clamp(lo, hi)
    return x + (clamped - x).detach()


def total_variation(x: torch.Tensor) -> torch.Tensor:
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return dx.abs().mean() + dy.abs().mean()


DAS_SHIFT = 56
DAS_NOISE_STD = 0.1


def make_views(
    image: torch.Tensor,
    n_views: int,
    out_size: int,
    shift: int = DAS_SHIFT,
    noise_std: float = DAS_NOISE_STD,
) -> torch.Tensor:
    """The paper's DAS augmentation: independent shifts and pixel noise.

    The image is enlarged by 2*shift and each view takes an independent
    out_size crop.  This matches the paper's +/-56 pixel translation window
    and Gaussian noise (std=0.1) at the 448x448 working resolution.
    """
    if shift < 0:
        raise ValueError("shift must be non-negative")
    enlarged_size = out_size + 2 * shift
    enlarged = F.interpolate(
        image, size=(enlarged_size, enlarged_size), mode="bilinear", align_corners=False
    )
    views = []
    for _ in range(n_views):
        offset_y = int(torch.randint(0, 2 * shift + 1, (), device=image.device).item())
        offset_x = int(torch.randint(0, 2 * shift + 1, (), device=image.device).item())
        crop = enlarged[:, :, offset_y:offset_y + out_size, offset_x:offset_x + out_size]
        views.append(crop + noise_std * torch.randn_like(crop))
    return torch.cat(views, dim=0)


class GemmaObjective:
    def __init__(
        self,
        model_id: str,
        device: str,
        dtype: torch.dtype,
        target: str,
        sentence: str,
        layer: int,
        instruction: str,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.target = target
        self.sentence = sentence
        self.layer = layer
        self.instruction = instruction

        print(f"Loading Gemma {model_id} on {self.device}")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
        ).to(self.device).eval()
        self.model.requires_grad_(False)

        template, dummy_pixels = self._image_template()
        self.image_template = template
        self.model_image_h = int(dummy_pixels.shape[-2])
        self.model_image_w = int(dummy_pixels.shape[-1])

        proc = self.processor.image_processor
        self.image_mean = torch.tensor(proc.image_mean, device=self.device, dtype=torch.float32).view(1, 3, 1, 1)
        self.image_std = torch.tensor(proc.image_std, device=self.device, dtype=torch.float32).view(1, 3, 1, 1)

        print("Computing language baseline...")
        with torch.no_grad():
            baseline = torch.stack([self.text_activation(w) for w in BASELINE_WORDS]).mean(dim=0)
            target_act = self.text_activation(target)
        self.text_direction = F.normalize(target_act - baseline, dim=0, eps=1e-8)

        print("Computing gray-image baseline...")
        gray = torch.full((1, 3, self.model_image_h, self.model_image_w), 0.5, device=self.device)
        with torch.no_grad():
            gray_patches = self.image_patch_activations(gray)
        self.image_baseline = gray_patches.mean(dim=1)[0]

        print(f"Gemma image size: {self.model_image_h}x{self.model_image_w}")
        print(f"Gemma layer: {self.layer}")
        print(f"Target concept: {self.target!r}")

    def render_text_chat(self, text: str) -> str:
        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def text_activation(self, text: str) -> torch.Tensor:
        rendered = self.render_text_chat(text)
        start = rendered.find(text)
        end = start + len(text)
        enc = self.tokenizer(
            rendered,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        offsets = enc.pop("offset_mapping")[0]
        mask = torch.tensor(
            [int(b) > start and int(a) < end for a, b in offsets.tolist()],
            device=self.device,
            dtype=torch.bool,
        )
        inputs = {k: v.to(self.device) for k, v in enc.items() if torch.is_tensor(v)}
        out = self.model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
        hidden = out.hidden_states[self.layer][0]
        return hidden[mask].mean(dim=0).float()

    def _image_template(self):
        dummy = Image.new("RGB", (448, 448), (128, 128, 128))
        messages = [{"role": "user", "content": [{"type": "image", "image": dummy}]}]
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
        inputs = {k: v.to(self.device) for k, v in inputs.items() if torch.is_tensor(v)}
        return inputs, dummy_pixels

    def preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        image = F.interpolate(
            image,
            size=(self.model_image_h, self.model_image_w),
            mode="bilinear",
            align_corners=False,
        )
        image = clamp_with_grad(image, 0.0, 1.0)
        return ((image - self.image_mean) / self.image_std).to(self.dtype)

    def image_patch_activations(self, image: torch.Tensor) -> torch.Tensor:
        batch = image.shape[0]
        inputs = {
            k: v.repeat(batch, *([1] * (v.ndim - 1)))
            for k, v in self.image_template.items()
        }
        inputs["pixel_values"] = self.preprocess_image(image)
        out = self.model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden = out.hidden_states[self.layer]
        mask = inputs["token_type_ids"].bool()
        n_tokens = int(mask[0].sum().item())
        return hidden[mask].view(batch, n_tokens, hidden.shape[-1]).float()

    def representation_loss(
        self,
        image: torch.Tensor,
        tau: float = 0.25,
        spatial_sigma: float = 2.0,
    ) -> tuple[torch.Tensor, dict]:
        patches = self.image_patch_activations(image)
        centered = patches - self.image_baseline[None, None, :]
        scores = F.cosine_similarity(
            centered,
            self.text_direction[None, None, :],
            dim=-1,
            eps=1e-8,
        )
        # Match paper_experiment.py: add a centered spatial Gaussian prior
        # to the patch logits before the temperature softmax.
        n_patches = centered.shape[1]
        side = int(round(math.sqrt(n_patches)))
        if side * side == n_patches:
            coords = torch.arange(side, device=centered.device, dtype=centered.dtype)
            yy, xx = torch.meshgrid(coords, coords, indexing="ij")
            center = (side - 1) / 2.0
            log_g = -((xx - center).square() + (yy - center).square()) / (
                2.0 * spatial_sigma * spatial_sigma
            )
            log_g = log_g.flatten()
        else:
            # Gemma currently produces a square patch grid; retain a safe
            # fallback for processors that do not.
            log_g = torch.zeros(n_patches, device=centered.device, dtype=centered.dtype)

        weights = F.softmax((scores + log_g[None, :]) / tau, dim=-1)
        rep = (weights[..., None] * centered).sum(dim=1)
        cosine = F.cosine_similarity(rep, self.text_direction[None, :], dim=-1, eps=1e-8)
        return -cosine.mean(), {
            "rep_cos": cosine.mean().detach(),
            "patch_max": scores.max().detach(),
            "weight_max": weights.max().detach(),
        }

    def sentence_loss(self, image: torch.Tensor) -> tuple[torch.Tensor, dict]:
        batch = image.shape[0]
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": Image.new("RGB", (448, 448), (128, 128, 128))},
                {"type": "text", "text": self.instruction},
            ],
        }]
        prefix = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            do_pan_and_scan=False,
        )
        prefix.pop("pixel_values")
        prefix.pop("num_crops", None)
        prefix = {k: v.to(self.device) for k, v in prefix.items() if torch.is_tensor(v)}

        target_ids = self.tokenizer(self.sentence, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.device)
        prefix_ids = prefix["input_ids"].repeat(batch, 1)
        prefix_attn = prefix["attention_mask"].repeat(batch, 1)
        prefix_types = prefix.get("token_type_ids", torch.zeros_like(prefix["input_ids"]))
        prefix_types = prefix_types.repeat(batch, 1)
        target_ids = target_ids.repeat(batch, 1)

        input_ids = torch.cat([prefix_ids, target_ids], dim=1)
        attention_mask = torch.cat([prefix_attn, torch.ones_like(target_ids)], dim=1)
        token_type_ids = torch.cat([prefix_types, torch.zeros_like(target_ids)], dim=1)
        labels = torch.full_like(input_ids, -100)
        labels[:, prefix_ids.shape[1]:] = target_ids

        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=self.preprocess_image(image),
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
        return out.loss, {
            "sentence_nll": out.loss.detach(),
            "sentence_log10p": (-out.loss.detach() / math.log(10.0)),
        }


class VQGANGenerator:
    def __init__(
        self,
        repo: str,
        subfolder: str,
        device: str,
        dtype: torch.dtype,
        image_size: int,
        seed: int,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.image_size = image_size

        print(f"Loading VQ model {repo}/{subfolder} on {self.device}")
        self.model = VQModel.from_pretrained(repo, subfolder=subfolder, torch_dtype=dtype).to(self.device).eval()
        self.model.requires_grad_(False)

        # Decoder checkpoint has len(block_out_channels)-1 spatial downsamples.
        factor = 2 ** (len(self.model.config.block_out_channels) - 1)
        self.latent_h = math.ceil(image_size / factor)
        self.latent_w = math.ceil(image_size / factor)
        self.embed_dim = self.model.quantize.embedding.weight.shape[1]

        g = torch.Generator(device=self.device).manual_seed(seed)
        codebook = self.model.quantize.embedding.weight.detach()
        indices = torch.randint(
            0,
            codebook.shape[0],
            (self.latent_h * self.latent_w,),
            generator=g,
            device=self.device,
        )
        init = codebook[indices].view(self.latent_h, self.latent_w, self.embed_dim)
        init = init.permute(2, 0, 1).unsqueeze(0).contiguous().float()
        self.z = torch.nn.Parameter(init)

        print(f"Codebook: {codebook.shape[0]} x {codebook.shape[1]}")
        print(f"Latent grid: {self.latent_h}x{self.latent_w}")

    def decode(self) -> tuple[torch.Tensor, torch.Tensor]:
        z_model = self.z.to(self.dtype)
        quant, commit_loss, _ = self.model.quantize(z_model)
        quant2 = self.model.post_quant_conv(quant)
        raw = self.model.decoder(quant2)
        image = clamp_with_grad((raw.float() + 1.0) * 0.5, 0.0, 1.0)
        return image, commit_loss.float()

    @torch.no_grad()
    def codebook_usage(self) -> int:
        z_model = self.z.to(self.dtype)
        _, _, info = self.model.quantize(z_model)
        indices = info[2]
        return int(indices.unique().numel())


class EMA:
    def __init__(self, tensor: torch.Tensor, decay: float):
        self.decay = decay
        self.value = tensor.detach().clone()

    @torch.no_grad()
    def update(self, tensor: torch.Tensor):
        self.value.mul_(self.decay).add_(tensor.detach(), alpha=1.0 - self.decay)


def main():
    parser = argparse.ArgumentParser(description="Experimental VQGAN + Gemma concept synthesis")
    parser.add_argument("--target", default="lion")
    parser.add_argument("--sentence", default="A person")
    parser.add_argument("--objective", choices=["representation", "sentence", "hybrid"], default="representation")
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--instruction", default="Describe this image in one sentence.")
    parser.add_argument("--gemma-model", default=GEMMA_MODEL)
    parser.add_argument("--vq-repo", default=VQ_REPO)
    parser.add_argument("--vq-subfolder", default=VQ_SUBFOLDER)
    parser.add_argument("--gemma-device", default="cuda:0")
    parser.add_argument("--vq-device", default="cuda:1")
    parser.add_argument("--gemma-dtype", default="bfloat16")
    parser.add_argument("--vq-dtype", default="float32")
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--views", type=int, default=4)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--spatial-sigma-start", type=float, default=2.0)
    parser.add_argument("--spatial-sigma-end", type=float, default=16.0)
    parser.add_argument("--das-shift", type=int, default=DAS_SHIFT)
    parser.add_argument("--das-noise-std", type=float, default=DAS_NOISE_STD)
    parser.add_argument("--rep-weight", type=float, default=4.0)
    parser.add_argument("--sentence-weight", type=float, default=0.15)
    parser.add_argument("--commit-weight", type=float, default=0.20)
    parser.add_argument("--tv-weight", type=float, default=0.04)
    parser.add_argument("--latent-l2-weight", type=float, default=0.001)
    parser.add_argument("--ema-decay", type=float, default=0.98)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--out", default="output_vqgan_gemma")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    gemma = GemmaObjective(
        model_id=args.gemma_model,
        device=args.gemma_device,
        dtype=parse_dtype(args.gemma_dtype),
        target=args.target,
        sentence=args.sentence,
        layer=args.layer,
        instruction=args.instruction,
    )

    generator = VQGANGenerator(
        repo=args.vq_repo,
        subfolder=args.vq_subfolder,
        device=args.vq_device,
        dtype=parse_dtype(args.vq_dtype),
        image_size=args.image_size,
        seed=args.seed,
    )

    optimizer = torch.optim.AdamW([generator.z], lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.08)
    ema = EMA(generator.z, args.ema_decay)

    best_score = float("inf")
    best_z = None
    rows = []

    with torch.no_grad():
        initial, _ = generator.decode()
        save_image(initial.cpu(), out_dir / "step_0000.png")

    print("\n=== VQGAN + Gemma optimization ===")
    print(f"target           = {args.target!r}")
    print(f"sentence         = {args.sentence!r}")
    print(f"objective        = {args.objective}")
    print(f"steps            = {args.steps}")
    print(f"lr               = {args.lr}")
    print(f"views            = {args.views}")
    print(f"spatial sigma    = {args.spatial_sigma_start} -> {args.spatial_sigma_end}")
    print(f"DAS shift/noise  = +/-{args.das_shift} / {args.das_noise_std}")
    print(f"rep weight       = {args.rep_weight}")
    print(f"sentence weight  = {args.sentence_weight}")
    print(f"commit weight    = {args.commit_weight}")
    print(f"tv weight        = {args.tv_weight}")
    print("===================================\n")

    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)

        image, commit_loss = generator.decode()
        views = make_views(
            image,
            args.views,
            gemma.model_image_h,
            shift=args.das_shift,
            noise_std=args.das_noise_std,
        )
        views = views.to(gemma.device)

        progress = step / max(args.steps - 1, 1)
        spatial_sigma = args.spatial_sigma_start + progress * (
            args.spatial_sigma_end - args.spatial_sigma_start
        )

        rep_loss = torch.zeros((), device=gemma.device)
        sentence_loss = torch.zeros((), device=gemma.device)
        rep_diag = {
            "rep_cos": torch.tensor(float("nan")),
            "patch_max": torch.tensor(float("nan")),
            "weight_max": torch.tensor(float("nan")),
        }
        sent_diag = {"sentence_nll": torch.tensor(float("nan")), "sentence_log10p": torch.tensor(float("nan"))}

        if args.objective in {"representation", "hybrid"}:
            rep_loss, rep_diag = gemma.representation_loss(
                views, tau=args.tau, spatial_sigma=spatial_sigma
            )
        if args.objective in {"sentence", "hybrid"}:
            sentence_loss, sent_diag = gemma.sentence_loss(views)

        # Regularizers live on the VQ device. Move scalar semantic losses back
        # before combining so autograd crosses the device-to-device copy.
        semantic = torch.zeros((), device=generator.device)
        if args.objective in {"representation", "hybrid"}:
            semantic = semantic + args.rep_weight * rep_loss.to(generator.device)
        if args.objective in {"sentence", "hybrid"}:
            semantic = semantic + args.sentence_weight * sentence_loss.to(generator.device)

        tv = total_variation(image)
        latent_l2 = generator.z.square().mean()
        loss = (
            semantic
            + args.commit_weight * commit_loss
            + args.tv_weight * tv
            + args.latent_l2_weight * latent_l2
        )

        loss.backward()
        grad_norm = generator.z.grad.norm().item()
        torch.nn.utils.clip_grad_norm_([generator.z], 10.0)
        optimizer.step()
        scheduler.step()
        ema.update(generator.z)

        # Keep the latent numerically near the scale of the learned codebook.
        with torch.no_grad():
            codebook = generator.model.quantize.embedding.weight.float()
            lo = codebook.amin(dim=0).view(1, -1, 1, 1)
            hi = codebook.amax(dim=0).view(1, -1, 1, 1)
            generator.z.clamp_(lo, hi)

        score = float(semantic.detach().item())
        if score < best_score:
            best_score = score
            best_z = generator.z.detach().clone()

        row = {
            "step": step + 1,
            "loss": float(loss.detach().item()),
            "semantic": score,
            "rep_cos": float(rep_diag["rep_cos"].item()),
            "patch_max": float(rep_diag["patch_max"].item()),
            "weight_max": float(rep_diag["weight_max"].item()),
            "spatial_sigma": spatial_sigma,
            "sentence_nll": float(sent_diag["sentence_nll"].item()),
            "sentence_log10p": float(sent_diag["sentence_log10p"].item()),
            "commit": float(commit_loss.detach().item()),
            "tv": float(tv.detach().item()),
            "latent_l2": float(latent_l2.detach().item()),
            "grad_norm": grad_norm,
            "lr": optimizer.param_groups[0]["lr"],
        }
        rows.append(row)

        if step == 0 or (step + 1) % 10 == 0 or step == args.steps - 1:
            print(
                f"step {step + 1:04d}/{args.steps}"
                f" | loss={row['loss']:+.5f}"
                f" | sem={row['semantic']:+.5f}"
                f" | rep_cos={row['rep_cos']:+.4f}"
                f" | weight_max={row['weight_max']:.4f}"
                f" | sent_nll={row['sentence_nll']:.4f}"
                f" | commit={row['commit']:.5f}"
                f" | tv={row['tv']:.5f}"
                f" | grad={row['grad_norm']:.3e}"
                f" | lr={row['lr']:.5f}"
            )

        if args.save_every > 0 and ((step + 1) % args.save_every == 0 or step == args.steps - 1):
            with torch.no_grad():
                preview, _ = generator.decode()
                save_image(preview.cpu(), out_dir / f"step_{step + 1:04d}.png")

    with torch.no_grad():
        final_z = generator.z.detach().clone()

        generator.z.copy_(best_z)
        best_image, _ = generator.decode()
        save_image(best_image.cpu(), out_dir / "best.png")

        generator.z.copy_(ema.value)
        ema_image, _ = generator.decode()
        save_image(ema_image.cpu(), out_dir / "ema.png")

        generator.z.copy_(final_z)
        final_image, _ = generator.decode()
        save_image(final_image.cpu(), out_dir / "final.png")

        usage = generator.codebook_usage()

    with open(out_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    torch.save(
        {
            "best_z": best_z.cpu(),
            "final_z": final_z.cpu(),
            "ema_z": ema.value.cpu(),
            "args": vars(args),
        },
        out_dir / "latents.pt",
    )

    print("\n=== Done ===")
    print(f"best semantic loss = {best_score:+.6f}")
    print(f"codebook entries used = {usage}")
    print(f"best image = {out_dir / 'best.png'}")
    print(f"ema image  = {out_dir / 'ema.png'}")
    print(f"final image= {out_dir / 'final.png'}")
    print(f"metrics    = {out_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
