from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration


MODEL_ID = "google/gemma-3-4b-it"

IMAGE_SIZE = 448
DAS_SHIFT = 56
DAS_NOISE_STD = 0.1
BATCH_SIZE = 8
STEPS = 600
GRAD_CLIP = 1.0
MOMENTUM = 0.9


# Appendix C of arXiv:2601.08017.
# Keep duplicate entries because they are duplicated in the paper's list.
BASELINE_WORDS: List[str] = [
    "desk",
    "jacket",
    "gondola",
    "laughter",
    "intelligence",
    "bicycle",
    "chair",
    "orchestra",
    "sand",
    "pottery",
    "arrowhead",
    "jewelry",
    "daffodil",
    "plateau",
    "estuary",
    "quilt",
    "moment",
    "bamboo",
    "ravine",
    "archive",
    "hieroglyph",
    "star",
    "clay",
    "fossil",
    "wildlife",
    "flour",
    "traffic",
    "bubble",
    "honey",
    "geode",
    "magnet",
    "ribbon",
    "zigzag",
    "puzzle",
    "tornado",
    "anthill",
    "galaxy",
    "poverty",
    "diamond",
    "universe",
    "vinegar",
    "nebula",
    "knowledge",
    "marble",
    "fog",
    "river",
    "scroll",
    "silhouette",
    "marble",
    "cake",
    "valley",
    "whisper",
    "pendulum",
    "tower",
    "table",
    "glacier",
    "whirlpool",
    "jungle",
    "wool",
    "anger",
    "rampart",
    "flower",
    "research",
    "hammer",
    "cloud",
    "justice",
    "dog",
    "butterfly",
    "needle",
    "fortress",
    "bonfire",
    "skyscraper",
    "caravan",
    "patience",
    "bacon",
    "velocity",
    "smoke",
    "electricity",
    "sunset",
    "anchor",
    "parchment",
    "courage",
    "statue",
    "oxygen",
    "time",
    "butterfly",
    "fabric",
    "pasta",
    "snowflake",
    "mountain",
    "echo",
    "piano",
    "sanctuary",
    "abyss",
    "air",
    "dewdrop",
    "garden",
    "literature",
    "rice",
    "enigma",
]


def paper_hparams(layer: int):
    if layer in {1, 5, 30}:
        return 0.04, 0.005

    if layer in {10, 15, 20, 25}:
        return 0.15, 0.5

    raise ValueError(
        "The paper reports synthesis hyperparameters only for "
        "layers 1, 5, 10, 15, 20, 25, and 30."
    )


def save_tensor_image(image: torch.Tensor, path: Path):
    image = image.detach().float().cpu().clamp(0.0, 1.0)
    array = (
        image[0]
        .permute(1, 2, 0)
        .mul(255.0)
        .round()
        .byte()
        .numpy()
    )
    Image.fromarray(array).save(path)


class MultiResolutionDAS(nn.Module):
    """
    Paper equation:

        perturbation = sum_r upscale_448(layer_r)
        I_final = I_gray + 1/2 tanh(perturbation)

    with:
        r in {448, 428, 408, ..., 8}
        I_gray = 0.5
    """

    def __init__(self):
        super().__init__()

        self.resolutions = list(
            range(
                IMAGE_SIZE,
                7,
                -20,
            )
        )

        if self.resolutions[-1] != 8:
            self.resolutions.append(8)

        self.layers = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(
                        1,
                        3,
                        resolution,
                        resolution,
                    )
                )
                for resolution in self.resolutions
            ]
        )

        print(
            "DAS resolutions:",
            self.resolutions,
        )

    def forward(self):
        perturbation = 0.0

        for layer in self.layers:
            perturbation = (
                perturbation
                + F.interpolate(
                    layer,
                    size=(IMAGE_SIZE, IMAGE_SIZE),
                    mode="bilinear",
                    align_corners=False,
                )
            )

        return (
            0.5
            + 0.5 * torch.tanh(perturbation)
        )


class GemmaAlignmentModel:
    def __init__(
        self,
        model_id: str,
        layer: int,
        target: str,
        device: str,
        dtype: torch.dtype,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.layer = layer
        self.target = target

        print(
            f"Loading {model_id} on {self.device}"
        )

        self.processor = AutoProcessor.from_pretrained(
            model_id
        )
        self.tokenizer = self.processor.tokenizer

        self.model = (
            Gemma3ForConditionalGeneration
            .from_pretrained(
                model_id,
                torch_dtype=dtype,
            )
            .to(self.device)
            .eval()
        )

        self.model.requires_grad_(False)

        n_layers = (
            self.model.config
            .text_config
            .num_hidden_layers
        )

        if not 1 <= layer <= n_layers:
            raise ValueError(
                f"layer must be in [1, {n_layers}]"
            )

        (
            self.image_inputs,
            dummy_pixels,
        ) = self._build_image_template()

        self.model_image_h = int(
            dummy_pixels.shape[-2]
        )
        self.model_image_w = int(
            dummy_pixels.shape[-1]
        )

        image_processor = (
            self.processor.image_processor
        )

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
            "\nComputing language baseline..."
        )

        self.language_baseline = (
            self.compute_language_baseline()
        )

        self.target_activation = (
            self.text_activation(
                target,
                verbose=True,
            )
        )

        self.target_representation_raw = (
            self.target_activation
            - self.language_baseline
        )

        self.target_representation = (
            F.normalize(
                self.target_representation_raw,
                dim=0,
                eps=1e-8,
            )
        )

        print(
            "\n=== Target representation ==="
        )
        print(
            f"target: {target!r}"
        )
        print(
            f"||activation|| = "
            f"{self.target_activation.norm().item():.6f}"
        )
        print(
            f"||language baseline|| = "
            f"{self.language_baseline.norm().item():.6f}"
        )
        print(
            f"||centered target|| = "
            f"{self.target_representation_raw.norm().item():.6f}"
        )
        print(
            "=============================\n"
        )

        print(
            "Computing gray-image baseline..."
        )

        self.image_baseline = (
            self.gray_image_baseline()
        )

        print(
            f"||image baseline|| = "
            f"{self.image_baseline.norm().item():.6f}\n"
        )

    def render_text_chat(
        self,
        text: str,
    ):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text,
                    }
                ],
            }
        ]

        return (
            self.processor
            .apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    def text_inputs_and_mask(
        self,
        text: str,
    ):
        rendered = self.render_text_chat(text)

        start_char = rendered.find(text)

        if start_char < 0:
            raise RuntimeError(
                f"Could not locate {text!r} "
                "inside rendered chat template."
            )

        end_char = start_char + len(text)

        encoded = self.tokenizer(
            rendered,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=False,
        )

        offsets = encoded.pop(
            "offset_mapping"
        )[0]

        mask = torch.tensor(
            [
                int(end) > start_char
                and int(start) < end_char
                for start, end
                in offsets.tolist()
            ],
            dtype=torch.bool,
        )

        if mask.sum() == 0:
            raise RuntimeError(
                f"No token positions matched {text!r}"
            )

        model_inputs = {
            key: value.to(self.device)
            for key, value in encoded.items()
            if torch.is_tensor(value)
        }

        return (
            model_inputs,
            mask.to(self.device),
        )

    @torch.no_grad()
    def text_activation(
        self,
        text: str,
        verbose: bool = False,
    ):
        inputs, mask = (
            self.text_inputs_and_mask(text)
        )

        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

        hidden = (
            outputs.hidden_states[
                self.layer
            ][0]
        )

        activation = (
            hidden[mask]
            .mean(dim=0)
            .float()
        )

        if verbose:
            token_ids = (
                inputs["input_ids"][0][mask]
            )

            tokens = (
                self.tokenizer
                .convert_ids_to_tokens(
                    token_ids.tolist()
                )
            )

            print(
                f"target tokens: {tokens}"
            )

        return activation

    @torch.no_grad()
    def compute_language_baseline(self):
        activations = []

        for i, word in enumerate(
            BASELINE_WORDS,
            start=1,
        ):
            activation = (
                self.text_activation(word)
            )

            activations.append(
                activation
            )

            if (
                i == 1
                or i % 10 == 0
                or i == len(BASELINE_WORDS)
            ):
                print(
                    f"  baseline "
                    f"{i:03d}/"
                    f"{len(BASELINE_WORDS)}"
                )

        return (
            torch.stack(activations)
            .mean(dim=0)
        )

    def _build_image_template(self):
        dummy = Image.new(
            "RGB",
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            ),
            (
                128,
                128,
                128,
            ),
        )

        # Image-only user turn.
        # No semantic text is added, so text conditioning does not
        # leak the target concept into image-token activations.
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": dummy,
                    },
                ],
            }
        ]

        inputs = (
            self.processor
            .apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
                do_pan_and_scan=False,
            )
        )

        dummy_pixels = inputs.pop(
            "pixel_values"
        )

        inputs.pop(
            "num_crops",
            None,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
            if torch.is_tensor(value)
        }

        if "token_type_ids" not in inputs:
            raise RuntimeError(
                "Gemma processor did not "
                "return token_type_ids."
            )

        n_image_tokens = int(
            (
                inputs["token_type_ids"]
                == 1
            )
            .sum()
            .item()
        )

        print(
            f"image-token positions: "
            f"{n_image_tokens}"
        )

        return inputs, dummy_pixels

    def preprocess_image(
        self,
        image: torch.Tensor,
    ):
        image = F.interpolate(
            image,
            size=(
                self.model_image_h,
                self.model_image_w,
            ),
            mode="bilinear",
            align_corners=False,
        )

        image = (
            image
            - self.image_mean
        ) / self.image_std

        return image.to(self.dtype)

    def image_patch_activations(
        self,
        image: torch.Tensor,
    ):
        batch = image.shape[0]

        inputs = {
            key: value.repeat(
                batch,
                *(
                    [1]
                    * (value.ndim - 1)
                ),
            )
            for key, value
            in self.image_inputs.items()
        }

        inputs["pixel_values"] = (
            self.preprocess_image(
                image
            )
        )

        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

        hidden = (
            outputs.hidden_states[
                self.layer
            ]
        )

        image_mask = (
            inputs["token_type_ids"]
            .bool()
        )

        counts = (
            image_mask.sum(dim=1)
        )

        if not torch.all(
            counts == counts[0]
        ):
            raise RuntimeError(
                "Different image-token "
                "counts in batch."
            )

        n_tokens = int(
            counts[0].item()
        )

        return (
            hidden[image_mask]
            .view(
                batch,
                n_tokens,
                hidden.shape[-1],
            )
            .float()
        )

    @torch.no_grad()
    def gray_image_baseline(self):
        gray = torch.full(
            (
                1,
                3,
                IMAGE_SIZE,
                IMAGE_SIZE,
            ),
            0.5,
            device=self.device,
            dtype=torch.float32,
        )

        patches = (
            self.image_patch_activations(
                gray
            )
        )

        return (
            patches
            .mean(dim=1)[0]
        )

    def image_representation(
        self,
        image: torch.Tensor,
        spatial_sigma: float,
        tau: float,
    ):
        patches = (
            self.image_patch_activations(
                image
            )
        )

        centered = (
            patches
            - self.image_baseline[
                None,
                None,
                :
            ]
        )

        semantic_scores = (
            F.cosine_similarity(
                centered,
                self.target_representation[
                    None,
                    None,
                    :
                ],
                dim=-1,
                eps=1e-8,
            )
        )

        n_patches = centered.shape[1]

        side = int(
            round(
                math.sqrt(n_patches)
            )
        )

        if side * side == n_patches:
            coords = torch.arange(
                side,
                device=centered.device,
                dtype=centered.dtype,
            )

            yy, xx = torch.meshgrid(
                coords,
                coords,
                indexing="ij",
            )

            center = (
                side - 1
            ) / 2.0

            log_g = -(
                (
                    xx - center
                ).square()
                + (
                    yy - center
                ).square()
            ) / (
                2.0
                * spatial_sigma
                * spatial_sigma
            )

            log_g = log_g.flatten()

        else:
            log_g = torch.zeros(
                n_patches,
                device=centered.device,
                dtype=centered.dtype,
            )

        weights = F.softmax(
            (
                semantic_scores
                + log_g[None, :]
            ) / tau,
            dim=-1,
        )

        representation = (
            weights[..., None]
            * centered
        ).sum(dim=1)

        return (
            representation,
            semantic_scores,
            weights,
        )

    def loss(
        self,
        image: torch.Tensor,
        spatial_sigma: float,
        tau: float,
    ):
        (
            representation,
            patch_scores,
            weights,
        ) = self.image_representation(
            image,
            spatial_sigma,
            tau,
        )

        cosine = (
            F.cosine_similarity(
                representation,
                self.target_representation[
                    None,
                    :
                ],
                dim=-1,
                eps=1e-8,
            )
        )

        loss = -cosine.mean()

        diagnostics = {
            "cosine_mean": (
                cosine.mean().detach()
            ),
            "cosine_min": (
                cosine.min().detach()
            ),
            "cosine_max": (
                cosine.max().detach()
            ),
            "patch_mean": (
                patch_scores
                .mean()
                .detach()
            ),
            "patch_max": (
                patch_scores
                .max()
                .detach()
            ),
            "weight_max": (
                weights
                .max()
                .detach()
            ),
        }

        return loss, diagnostics


def das_augment(
    image: torch.Tensor,
    batch_size: int,
):
    """
    Paper augmentation:
    - independently random shift horizontally/vertically in [-56, 56]
    - first upscale sufficiently to permit shifting, then crop to 448x448
    - Gaussian pixel noise std = 0.1

    We implement this by resizing to:
        448 + 2*56 = 560
    then choosing an independent 448x448 crop for each batch element.
    The crop offset in [0,112] corresponds to a shift in [-56,56]
    around the centered crop.
    """
    enlarged_size = (
        IMAGE_SIZE
        + 2 * DAS_SHIFT
    )

    enlarged = F.interpolate(
        image,
        size=(
            enlarged_size,
            enlarged_size,
        ),
        mode="bilinear",
        align_corners=False,
    )

    augmented = []

    for _ in range(batch_size):
        offset_y = int(
            torch.randint(
                low=0,
                high=2 * DAS_SHIFT + 1,
                size=(),
                device=image.device,
            ).item()
        )

        offset_x = int(
            torch.randint(
                low=0,
                high=2 * DAS_SHIFT + 1,
                size=(),
                device=image.device,
            ).item()
        )

        crop = enlarged[
            :,
            :,
            offset_y:
                offset_y
                + IMAGE_SIZE,
            offset_x:
                offset_x
                + IMAGE_SIZE,
        ]

        crop = (
            crop
            + DAS_NOISE_STD
            * torch.randn_like(crop)
        )

        augmented.append(crop)

    return torch.cat(
        augmented,
        dim=0,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target",
        default="lion",
    )

    parser.add_argument(
        "--layer",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--model",
        default=MODEL_ID,
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=STEPS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--out",
        default="output",
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(
        args.seed
    )

    learning_rate, tau = (
        paper_hparams(
            args.layer
        )
    )

    print(
        "\n=== Paper hyperparameters ==="
    )
    print(
        f"target       = {args.target}"
    )
    print(
        f"layer        = {args.layer}"
    )
    print(
        f"steps        = {args.steps}"
    )
    print(
        f"batch size   = {args.batch_size}"
    )
    print(
        f"learning rate= {learning_rate}"
    )
    print(
        f"tau          = {tau}"
    )
    print(
        f"momentum     = {MOMENTUM}"
    )
    print(
        f"grad clip    = {GRAD_CLIP}"
    )
    print(
        f"DAS shift    = +/-{DAS_SHIFT}"
    )
    print(
        f"DAS noise    = {DAS_NOISE_STD}"
    )
    print(
        "spatial sigma= 2 -> 16"
    )
    print(
        "=============================\n"
    )

    alignment = GemmaAlignmentModel(
        model_id=args.model,
        layer=args.layer,
        target=args.target,
        device=args.device,
        dtype=torch.bfloat16,
    )

    das = MultiResolutionDAS().to(
        args.device
    )

    optimizer = torch.optim.SGD(
        das.parameters(),
        lr=learning_rate,
        momentum=MOMENTUM,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "target": args.target,
            "layer": args.layer,
            "language_baseline": (
                alignment
                .language_baseline
                .cpu()
            ),
            "target_activation": (
                alignment
                .target_activation
                .cpu()
            ),
            "target_representation": (
                alignment
                .target_representation
                .cpu()
            ),
            "baseline_words": (
                BASELINE_WORDS
            ),
        },
        out_dir
        / "concept_representation.pt",
    )

    print(
        "\n=== Starting DAS optimization ===\n"
    )

    for step in range(
        args.steps
    ):
        optimizer.zero_grad(
            set_to_none=True
        )

        clean_image = das()

        augmented_images = (
            das_augment(
                clean_image,
                args.batch_size,
            )
        )

        progress = (
            step
            / max(
                args.steps - 1,
                1,
            )
        )

        spatial_sigma = (
            2.0
            + progress
            * (
                16.0 - 2.0
            )
        )

        loss, diagnostics = (
            alignment.loss(
                image=augmented_images,
                spatial_sigma=(
                    spatial_sigma
                ),
                tau=tau,
            )
        )

        loss.backward()

        grad_norm = (
            torch.nn.utils
            .clip_grad_norm_(
                das.parameters(),
                max_norm=GRAD_CLIP,
            )
        )

        optimizer.step()

        if (
            step == 0
            or (step + 1) % 10 == 0
            or step
            == args.steps - 1
        ):
            print(
                f"step "
                f"{step + 1:04d}/"
                f"{args.steps}"
                f" | loss="
                f"{loss.item():+.6f}"
                f" | cosine="
                f"{diagnostics['cosine_mean'].item():+.6f}"
                f" | cos_min="
                f"{diagnostics['cosine_min'].item():+.6f}"
                f" | cos_max="
                f"{diagnostics['cosine_max'].item():+.6f}"
                f" | patch_mean="
                f"{diagnostics['patch_mean'].item():+.6f}"
                f" | patch_max="
                f"{diagnostics['patch_max'].item():+.6f}"
                f" | weight_max="
                f"{diagnostics['weight_max'].item():.6f}"
                f" | sigma="
                f"{spatial_sigma:.3f}"
                f" | grad_norm="
                f"{float(grad_norm):.6f}"
            )

        if (
            args.save_every > 0
            and (
                step == 0
                or (
                    step + 1
                ) % args.save_every == 0
                or step
                == args.steps - 1
            )
        ):
            save_tensor_image(
                clean_image,
                out_dir
                / (
                    f"step_"
                    f"{step + 1:04d}"
                    f".png"
                ),
            )

    final_image = das()

    final_path = (
        out_dir
        / (
            f"{args.target.replace(' ', '_')}"
            f"_layer{args.layer}"
            f"_final.png"
        )
    )

    save_tensor_image(
        final_image,
        final_path,
    )

    # Evaluate final unaugmented image with final sigma.
    with torch.no_grad():
        final_loss, final_diag = (
            alignment.loss(
                image=final_image,
                spatial_sigma=16.0,
                tau=tau,
            )
        )

    print(
        "\n=== Final unaugmented alignment ==="
    )
    print(
        f"loss   = "
        f"{final_loss.item():+.6f}"
    )
    print(
        f"cosine = "
        f"{final_diag['cosine_mean'].item():+.6f}"
    )
    print(
        f"patch max = "
        f"{final_diag['patch_max'].item():+.6f}"
    )
    print(
        "===================================\n"
    )

    print(
        f"Saved final image to: "
        f"{final_path}"
    )


if __name__ == "__main__":
    main()
