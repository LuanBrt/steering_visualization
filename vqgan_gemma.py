#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
from pathlib import Path
import sys
import types
from typing import List

# The upstream setup.py does not discover taming's namespace packages. Keep
# the editable source checkout importable when installed from requirements.txt.
_taming_source = Path(__file__).resolve().parent / "src" / "taming-transformers"
if _taming_source.is_dir():
    sys.path.insert(0, str(_taming_source))

if "main" not in sys.modules:
    _compat_main = types.ModuleType("main")

    def instantiate_from_config(config):
        module_name, class_name = config["target"].rsplit(".", 1)
        cls = getattr(importlib.import_module(module_name), class_name)
        return cls(**config.get("params", {}))

    _compat_main.instantiate_from_config = instantiate_from_config
    sys.modules["main"] = _compat_main

from huggingface_hub import hf_hub_download
from taming.models.vqgan import VQModel

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.checkpoint import checkpoint
from torchvision.utils import save_image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration


GEMMA_MODEL = "google/gemma-3-4b-it"
TAMING_VQ_REPO = "valhalla/vqgan_imagenet_f16_16384"

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

    # household / objects
    "lamp", "mirror", "shelf", "cupboard", "blanket", "pillow",
    "mug", "bottle", "basket", "bucket", "ladder", "scissors",
    "notebook", "pencil", "clock", "key", "lock", "window",
    "door", "carpet", "curtain", "candle", "spoon", "fork",

    # tools / machines
    "wrench", "screwdriver", "drill", "saw", "engine", "tractor",
    "camera", "telescope", "microscope", "computer", "printer",
    "radio", "telephone", "satellite", "robot", "generator", "pump",

    # transport
    "airplane", "helicopter", "train", "bus", "truck", "motorcycle",
    "sailboat", "canoe", "submarine", "rocket", "wagon", "scooter",

    # animals
    "cat", "horse", "cow", "sheep", "goat", "rabbit", "fox",
    "wolf", "bear", "deer", "elephant", "giraffe", "zebra",
    "monkey", "dolphin", "whale", "shark", "eagle", "owl",
    "sparrow", "penguin", "frog", "snake", "lizard", "turtle",

    # plants / nature
    "tree", "moss", "fern", "cactus", "orchid", "rose", "tulip",
    "mushroom", "forest", "meadow", "desert", "ocean", "lake",
    "waterfall", "canyon", "volcano", "island", "coast", "swamp",

    # weather / physical phenomena
    "rain", "snow", "thunder", "lightning", "storm", "hail",
    "rainbow", "breeze", "frost", "steam", "shadow", "reflection",
    "flame", "spark", "wave", "current", "gravity", "friction",

    # materials / substances
    "wood", "steel", "glass", "plastic", "rubber", "leather",
    "paper", "concrete", "ceramic", "copper", "silver", "gold",
    "salt", "sugar", "oil", "water", "ink", "paint",

    # food
    "bread", "cheese", "apple", "banana", "orange", "grape",
    "tomato", "potato", "carrot", "onion", "coffee", "tea",
    "chocolate", "soup", "salad", "pizza", "noodle", "cookie",

    # architecture / places
    "bridge", "castle", "temple", "church", "palace", "stadium",
    "school", "hospital", "factory", "warehouse", "harbor", "village",
    "city", "street", "alley", "plaza", "tunnel", "dam",

    # shapes / visual properties
    "circle", "triangle", "square", "spiral", "stripe", "checkerboard",
    "curve", "line", "grid", "pattern", "texture", "symmetry",
    "brightness", "darkness", "color", "contrast", "depth", "motion",

    # abstract concepts
    "freedom", "memory", "truth", "beauty", "fear", "hope",
    "wisdom", "chaos", "order", "peace", "conflict", "energy",
    "distance", "balance", "chance", "growth", "decay", "change",
    "identity", "meaning", "language", "history", "culture", "science",

    # activities / processes
    "running", "writing", "painting", "cooking", "dancing", "flying",
    "swimming", "reading", "building", "melting", "freezing",
    "falling", "rising", "spinning", "burning", "growing",

    # music / arts
    "violin", "guitar", "drum", "trumpet", "melody", "rhythm",
    "painting", "sculpture", "portrait", "poetry", "theater", "dance",

    # geography / geology
    "continent", "peninsula", "delta", "basin", "ridge", "cliff",
    "cave", "crater", "dune", "reef", "lagoon", "geyser",

    # astronomy
    "planet", "moon", "comet", "asteroid", "meteor", "eclipse",
    "constellation", "orbit", "cosmos", "sun", "space", "horizon",

    # miscellaneous semantic diversity
    "signal", "network", "algorithm", "equation", "number", "map",
    "letter", "book", "coin", "mask", "rope", "feather",
    "shell", "crown", "sword", "shield", "lantern", "fountain",
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


def make_cutouts(
    image: torch.Tensor,
    n_cutouts: int,
    out_size: int,
    min_scale: float = 0.25,
    max_scale: float = 1.0,
    noise_std: float = 0.01,
) -> torch.Tensor:
    """VQGAN-CLIP-style random cutout ensemble.

    Each cutout samples an independent square crop, resizes it to the
    scorer's input size, and applies lightweight differentiable augmentations.
    Averaging the scorer loss over these views reduces single-patch shortcuts
    and follows the augmentation strategy used by VQGAN-CLIP.
    """
    if n_cutouts <= 0:
        return image.new_empty((0, image.shape[1], out_size, out_size))
    if not 0.0 < min_scale <= max_scale <= 1.0:
        raise ValueError("cutout scales must satisfy 0 < min_scale <= max_scale <= 1")

    _, _, h, w = image.shape
    short_side = min(h, w)
    cutouts = []
    for _ in range(n_cutouts):
        scale = min_scale + (max_scale - min_scale) * torch.rand((), device=image.device)
        side = max(8, int(round(float(scale.item()) * short_side)))
        side = min(side, short_side)
        y0 = int(torch.randint(0, h - side + 1, (), device=image.device).item())
        x0 = int(torch.randint(0, w - side + 1, (), device=image.device).item())
        crop = image[:, :, y0:y0 + side, x0:x0 + side]
        crop = F.interpolate(crop, size=(out_size, out_size), mode="bilinear", align_corners=False)

        if bool(torch.rand((), device=image.device) < 0.5):
            crop = crop.flip(-1)

        # Lightweight color jitter, as used in the VQGAN-CLIP cutout path.
        brightness = 1.0 + (torch.rand((), device=image.device) * 0.2 - 0.1)
        contrast = 1.0 + (torch.rand((), device=image.device) * 0.2 - 0.1)
        mean = crop.mean(dim=(-2, -1), keepdim=True)
        crop = (crop - mean) * contrast + mean
        crop = crop * brightness
        if noise_std > 0:
            crop = crop + noise_std * torch.randn_like(crop)
        cutouts.append(clamp_with_grad(crop, 0.0, 1.0))
    return torch.cat(cutouts, dim=0)


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
        if layer < 1:
            raise ValueError("layer must be at least 1 (the embedding output is layer 0)")

        print(f"Loading Gemma {model_id} on {self.device}")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
        ).to(self.device).eval()
        self.model.requires_grad_(False)

        # Capture only the requested transformer block instead of asking the
        # model to retain every intermediate hidden state. `layer` preserves
        # the old output_hidden_states indexing: layer 1 is block 0.
        self._hook_hidden = None

        def hook_fn(_module, _inputs, output):
            self._hook_hidden = output[0] if isinstance(output, tuple) else output

        language_model = self.model.model.language_model
        try:
            layer_module = language_model.layers[layer - 1]
        except (AttributeError, IndexError) as exc:
            raise ValueError(f"layer {layer} is not available in Gemma") from exc
        self._layer_hook = layer_module.register_forward_hook(hook_fn)

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

    def _forward_with_hook(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run Gemma while retaining only the hooked layer activation.

        During optimization, checkpointing recomputes Gemma during backward
        instead of retaining its full transformer activation graph. The hook
        output is returned from the checkpoint wrapper, so the image gradient
        still flows through the selected layer.
        """
        keys = tuple(inputs)
        values = tuple(inputs[key] for key in keys)

        def forward(*args):
            self._hook_hidden = None
            self.model(
                **dict(zip(keys, args)),
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )
            if self._hook_hidden is None:
                raise RuntimeError("Gemma layer hook did not capture an activation")
            hidden = self._hook_hidden
            self._hook_hidden = None
            return hidden

        if any(value.requires_grad for value in values):
            return checkpoint(forward, *values, use_reentrant=False)
        return forward(*values)

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
        hidden = self._forward_with_hook(inputs)[0]
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
        hidden = self._forward_with_hook(inputs)
        mask = inputs["token_type_ids"].bool()
        n_tokens = int(mask[0].sum().item())
        return hidden[mask].view(batch, n_tokens, hidden.shape[-1]).float()

    def representation_loss(
        self,
        image: torch.Tensor,
        tau: float = 0.25,
        spatial_sigma: float = 2.0,
        distance: str = "cosine",
        batch_size: int | None = None,
    ) -> tuple[torch.Tensor, dict]:
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive or None")
        if batch_size is not None and image.shape[0] > batch_size:
            # Keep all views/cutouts in one logical objective while limiting
            # the peak Gemma memory used by each forward pass. Weighting by
            # chunk size makes this identical to one large batch.
            image_chunks = list(image.split(batch_size, dim=0))
            chunks = [
                self.representation_loss(
                    chunk,
                    tau=tau,
                    spatial_sigma=spatial_sigma,
                    distance=distance,
                    batch_size=None,
                )
                for chunk in image_chunks
            ]
            chunk_sizes = [chunk.shape[0] for chunk in image_chunks]
            total = image.shape[0]
            weighted = lambda key: sum(
                result[1][key] * size for result, size in zip(chunks, chunk_sizes)
            ) / total
            return (
                sum(result[0] * size for result, size in zip(chunks, chunk_sizes)) / total,
                {
                    "rep_cos": weighted("rep_cos"),
                    "rep_distance": weighted("rep_distance"),
                    "patch_max": max(result[1]["patch_max"] for result in chunks),
                    "weight_max": max(result[1]["weight_max"] for result in chunks),
                },
            )

        if distance == "geodesic":
            distance = "spherical"
        if distance not in {"cosine", "spherical"}:
            raise ValueError(
                f"unknown representation distance {distance!r}; "
                "expected 'cosine' or 'spherical'"
            )

        patches = self.image_patch_activations(image)
        centered = patches - self.image_baseline[None, None, :]
        target = self.text_direction[None, None, :]
        if distance == "cosine":
            scores = F.cosine_similarity(centered, target, dim=-1, eps=1e-8)
        else:
            centered_normalized = F.normalize(centered, dim=-1, eps=1e-8)
            target_normalized = F.normalize(target, dim=-1, eps=1e-8)
            chord_half = (centered_normalized - target_normalized).norm(dim=-1) / 2.0
            # For unit vectors this is theta^2 / 2, where theta is the
            # geodesic angle. Negate it because larger scores get larger
            # softmax weights below.
            scores = -2.0 * torch.asin(chord_half.clamp(max=1.0)).square()
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
        target = self.text_direction[None, :]
        if distance == "cosine":
            cosine = F.cosine_similarity(rep, target, dim=-1, eps=1e-8)
            loss = -cosine.mean()
            rep_distance = torch.zeros_like(cosine)
        else:
            rep_normalized = F.normalize(rep, dim=-1, eps=1e-8)
            target_normalized = F.normalize(target, dim=-1, eps=1e-8)
            chord_half = (rep_normalized - target_normalized).norm(dim=-1) / 2.0
            rep_distance = 2.0 * torch.asin(chord_half.clamp(max=1.0)).square()
            cosine = F.cosine_similarity(rep, target, dim=-1, eps=1e-8)
            loss = rep_distance.mean()
        return loss, {
            "rep_cos": cosine.mean().detach(),
            "rep_distance": rep_distance.mean().detach(),
            "patch_max": scores.max().detach(),
            "weight_max": weights.max().detach(),
        }


class VQGANGenerator:
    def __init__(
        self,
        repo: str,
        device: str,
        dtype: torch.dtype,
        image_size: int,
        seed: int,
        init_scale: float,
        init_method: str,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.image_size = image_size
        if init_scale < 0:
            raise ValueError("init-scale must be non-negative")
        if init_method not in {"gaussian", "codebook"}:
            raise ValueError("init-method must be 'gaussian' or 'codebook'")

        # The Hugging Face repository contains the original taming-transformers
        # weights as a config.json + pytorch_model.bin pair.  Instantiate the
        # original VQModel so its quantizer and decoder remain differentiable.
        print(f"Loading taming-transformers VQGAN {repo} on {self.device}")
        config_path = hf_hub_download(repo_id=repo, filename="config.json")
        weights_path = hf_hub_download(repo_id=repo, filename="pytorch_model.bin")
        with open(config_path) as f:
            config = json.load(f)

        # Support both the original taming names and the equivalent names in
        # the Hugging Face conversion of this checkpoint.
        get = lambda old, new, default=None: config.get(old, config.get(new, default))
        ddconfig = {
            "double_z": get("double_z", "double_z", False),
            "z_channels": get("z_channels", "z_channels"),
            "resolution": get("resolution", "resolution"),
            "in_channels": get("in_channels", "num_channels", 3),
            "out_ch": get("out_ch", "num_channels", 3),
            "ch": get("ch", "hidden_channels"),
            "ch_mult": get("ch_mult", "channel_mult"),
            "num_res_blocks": get("num_res_blocks", "num_res_blocks"),
            "attn_resolutions": get("attn_resolutions", "attn_resolutions"),
            "dropout": get("dropout", "dropout", 0.0),
        }
        # The loss is not used during inference/optimization. Identity avoids
        # constructing the discriminator and its extra perceptual dependencies.
        self.model = VQModel(
            ddconfig=ddconfig,
            lossconfig={"target": "torch.nn.Identity", "params": {}},
            n_embed=get("n_embed", "num_embeddings", 16384),
            embed_dim=get("embed_dim", "quantized_embed_dim", 256),
        )
        state = torch.load(weights_path, map_location="cpu", weights_only=False)
        if "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise ValueError(f"Unsupported VQGAN checkpoint format in {weights_path}")
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise ValueError(
                "VQGAN checkpoint does not match its config: "
                f"{len(missing)} missing, {len(unexpected)} unexpected keys"
            )
        self.model = self.model.to(device=self.device, dtype=dtype).eval()
        self.model.requires_grad_(False)

        factor = 16
        if image_size % factor != 0:
            raise ValueError(f"image-size must be divisible by VQGAN compression factor f={factor}")
        self.latent_h = image_size // factor
        self.latent_w = image_size // factor
        self.embed_dim = self.model.quantize.embedding.weight.shape[1]

        g = torch.Generator(device=self.device).manual_seed(seed)
        codebook = self.model.quantize.embedding.weight.detach().float()
        if init_method == "codebook":
            indices = torch.randint(
                0,
                codebook.shape[0],
                (self.latent_h * self.latent_w,),
                generator=g,
                device=self.device,
            )
            init = codebook[indices].view(self.latent_h, self.latent_w, self.embed_dim)
            init = init.permute(2, 0, 1).unsqueeze(0).contiguous()
        else:
            codebook_mean = codebook.mean(dim=0).view(1, self.embed_dim, 1, 1)
            codebook_std = codebook.std(dim=0, unbiased=False).view(1, self.embed_dim, 1, 1)
            init = codebook_mean + init_scale * codebook_std * torch.randn(
                (1, self.embed_dim, self.latent_h, self.latent_w),
                generator=g,
                device=self.device,
                dtype=torch.float32,
            )
        init = init.float()
        self.z = torch.nn.Parameter(init)

        print(f"Codebook: {codebook.shape[0]} x {codebook.shape[1]}")
        print(f"Latent grid: {self.latent_h}x{self.latent_w}")

    def decode(self) -> tuple[torch.Tensor, torch.Tensor]:
        z_model = self.z.to(self.dtype)
        quant, commit_loss, _ = self.model.quantize(z_model)
        raw = self.model.decode(quant)
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
    parser.add_argument("--target", default="fear")
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--instruction", default="Describe this image in one sentence.")
    parser.add_argument("--gemma-model", default=GEMMA_MODEL)
    parser.add_argument("--vq-repo", default=TAMING_VQ_REPO)
    parser.add_argument("--gemma-device", default="cuda:0")
    parser.add_argument("--vq-device", default="cuda:1")
    parser.add_argument("--gemma-dtype", default="bfloat16")
    parser.add_argument("--vq-dtype", default="float32")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--init-scale",
        type=float,
        default=0.5,
        help="standard-deviation multiplier for continuous codebook-space initialization",
    )
    parser.add_argument(
        "--init-method",
        choices=["gaussian", "codebook"],
        default="codebook",
        help="latent initialization method; codebook restores independent random entries",
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--views", type=int, default=4)
    parser.add_argument("--cutouts", type=int, default=16, help="additional VQGAN-CLIP random cutouts per step")
    parser.add_argument(
        "--gemma-batch-size",
        type=int,
        default=8,
        help="views/cutouts per Gemma forward pass; 0 uses the full batch",
    )
    parser.add_argument("--cutout-min-scale", type=float, default=0.45)
    parser.add_argument("--cutout-max-scale", type=float, default=1.0)
    parser.add_argument("--cutout-noise-std", type=float, default=0.01)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument(
        "--representation-distance",
        "--rep-distance",
        choices=["cosine", "spherical", "geodesic"],
        default="spherical",
        dest="representation_distance",
        help="distance used by the representation objective (default: cosine)",
    )
    parser.add_argument("--spatial-sigma-start", type=float, default=2.0)
    parser.add_argument("--spatial-sigma-end", type=float, default=16.0)
    parser.add_argument("--das-shift", type=int, default=DAS_SHIFT)
    parser.add_argument("--das-noise-std", type=float, default=DAS_NOISE_STD)
    parser.add_argument("--rep-weight", type=float, default=4.0)
    parser.add_argument("--sentence-weight", type=float, default=0.15)
    parser.add_argument("--commit-weight", type=float, default=0.02)
    parser.add_argument("--tv-weight", type=float, default=0.02)
    parser.add_argument("--latent-l2-weight", type=float, default=0)
    parser.add_argument(
        "--latent-l2-decay",
        type=float,
        default=0,
        help="multiplicative per-step decay (0.005 means 0.995x per step)",
    )
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--out", default="output_vqgan_gemma_fear_codebook")
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
        sentence="",
        layer=args.layer,
        instruction=args.instruction,
    )

    generator = VQGANGenerator(
        repo=args.vq_repo,
        device=args.vq_device,
        dtype=parse_dtype(args.vq_dtype),
        image_size=args.image_size,
        seed=args.seed,
        init_scale=args.init_scale,
        init_method=args.init_method,
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
    print(f"steps            = {args.steps}")
    print(f"lr               = {args.lr}")
    print(f"views            = {args.views}")
    print(f"cutouts          = {args.cutouts} ({args.cutout_min_scale} -> {args.cutout_max_scale})")
    print(f"Gemma batch size  = {args.gemma_batch_size or 'full'}")
    print(f"spatial sigma    = {args.spatial_sigma_start} -> {args.spatial_sigma_end}")
    print(f"DAS shift/noise  = +/-{args.das_shift} / {args.das_noise_std}")
    print(f"latent L2 decay  = {args.latent_l2_decay} per step")
    print(f"rep weight       = {args.rep_weight}")
    print(f"rep distance     = {args.representation_distance}")
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
        if args.cutouts > 0:
            cutouts = make_cutouts(
                image,
                args.cutouts,
                gemma.model_image_h,
                min_scale=args.cutout_min_scale,
                max_scale=args.cutout_max_scale,
                noise_std=args.cutout_noise_std,
            )
            views = torch.cat([views, cutouts], dim=0)
        views = views.to(gemma.device)

        progress = step / max(args.steps - 1, 1)
        spatial_sigma = args.spatial_sigma_start + progress * (
            args.spatial_sigma_end - args.spatial_sigma_start
        )

        rep_loss = torch.zeros((), device=gemma.device)
        rep_diag = {
            "rep_cos": torch.tensor(float("nan")),
            "patch_max": torch.tensor(float("nan")),
            "weight_max": torch.tensor(float("nan")),
        }
        sent_diag = {"sentence_nll": torch.tensor(float("nan")), "sentence_log10p": torch.tensor(float("nan"))}

        rep_loss, rep_diag = gemma.representation_loss(
            views,
            tau=args.tau,
            spatial_sigma=spatial_sigma,
            distance=args.representation_distance,
            batch_size=args.gemma_batch_size or None,
        )

        # Regularizers live on the VQ device. Move scalar semantic losses back
        # before combining so autograd crosses the device-to-device copy.
        semantic = torch.zeros((), device=generator.device)
        semantic = semantic + args.rep_weight * rep_loss.to(generator.device)

        tv = total_variation(image)
        latent_l2 = generator.z.square().mean()
        if not 0.0 <= args.latent_l2_decay < 1.0:
            raise ValueError("latent-l2-decay must be in [0, 1)")
        # Decay the latent penalty so it constrains early texture formation
        # while allowing semantic optimization more freedom later.
        latent_l2_weight = args.latent_l2_weight * (
            1.0 - args.latent_l2_decay
        ) ** step
        loss = (
            semantic
            + args.commit_weight * commit_loss
            + args.tv_weight * tv
            + latent_l2_weight * latent_l2
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
            "latent_l2_weight": latent_l2_weight,
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
