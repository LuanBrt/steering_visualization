# Paper Method vs. `vqgan_gemma.py`

`vqgan_gemma.py` keeps the paper's core idea—optimize an image so its Gemma image representation aligns with a text-concept direction—but changes the synthesis process substantially.

The paper defines the goal as maximizing cosine similarity between a text representation and an image representation at a selected Gemma layer. It uses weighted patch aggregation, direct ascent synthesis (DAS) augmentations, and a multi-resolution pixel perturbation parameterization.

## Main differences

| Component | Paper method | `vqgan_gemma.py` |
|---|---|---|
| Image parameterization | Directly optimizes a sum of trainable RGB perturbations at resolutions from `8x8` to `448x448`, added to a gray image through `tanh`. | Optimizes a continuous latent tensor `z` in a pretrained VQGAN codebook space, then decodes it into an image. |
| Image prior | Neutral gray base image plus a multi-resolution perturbation. | VQGAN decoder and quantizer provide the image prior. |
| Resolution | Optimizes a `448x448` image. | Defaults to a `256x256` VQGAN image, which is resized for Gemma/DAS processing. |
| Quantization | No VQGAN quantization. | Uses VQGAN quantization during decoding while keeping `z` continuous and differentiable. |
| Initialization | Perturbation initialized around a gray image. | Either Gaussian initialization based on codebook statistics or random codebook entries. |
| Image baseline | Computes a separate Gemma activation baseline from a neutral gray image and subtracts it from every image patch. | Does not compute the paper's gray-image patch baseline. It instead applies the text-derived background transform to image patches. This is a substantive methodological difference. |
| Text concept vector | Target activation minus the mean activation of approximately 100 baseline words. | Uses the same general baseline-centering idea, but with a much larger/custom `BASELINE_WORDS` list. |
| Background correction | Only mean subtraction. | Adds optional top-principal-component removal or regularized whitening; whitening is the default. |
| Patch aggregation | Cosine similarity between each centered patch and the target, combined with a Gaussian spatial prior and softmax weighting. | Retains this mechanism. It also adds an optional spherical/geodesic distance objective instead of cosine similarity. |
| Augmentation | DAS random shifts of up to `+/-56` pixels plus Gaussian noise with standard deviation `0.1`. | Retains the DAS augmentation with the same defaults. |
| Additional augmentation | None described in the paper's main method. | Adds VQGAN-CLIP-style random cutouts, resizing, horizontal flips, brightness/contrast jitter, and noise. |
| Optimization variable | Multi-resolution RGB perturbation tensors. | One trainable VQGAN latent tensor. |
| Optimizer | SGD with momentum `0.9`. | AdamW with betas `(0.9, 0.99)` and zero weight decay. |
| Learning-rate schedule | Fixed layer-dependent learning rate: `0.04` for layers 1, 5, and 30; `0.15` for the other tested layers. | Default learning rate `0.08`, followed by cosine annealing. |
| Number of steps | 600 steps. | Defaults to 1,000 steps. |
| Gradient clipping | Maximum norm `1.0`. | Maximum norm `10.0`. |
| Batch handling | Batch size 8. | Generates multiple DAS views and additional cutouts; Gemma forwards can be chunked using `--gemma-batch-size`. |
| Loss | Representation alignment loss. | Representation loss plus VQGAN commitment loss, total variation loss, and optional latent L2 regularization. |
| Latent constraints | `tanh` guarantees valid pixels in `[0, 1]`. | Uses straight-through pixel clamping and clamps latent channels to the per-channel VQGAN codebook min/max after every step. |
| Stabilization/output | Saves synthesized images. | Adds EMA latents, best/final/EMA image selection, intermediate checkpoints, a metrics CSV, and latent serialization. |

## Effective loss

The implementation's effective loss is approximately:

```text
total_loss =
    rep_weight * Gemma_representation_loss
  + commit_weight * VQGAN_commitment_loss
  + tv_weight * total_variation(image)
  + latent_l2_weight * mean(z^2)
```

## Core conceptual change

The paper performs DAS optimization in multi-resolution pixel space. This implementation performs optimization in VQGAN latent/codebook space and uses DAS plus VQGAN-CLIP-style cutouts to make that latent optimization work.

In short:

```text
Paper:       multi-resolution pixel perturbation -> Gemma representation matching
This file:   VQGAN latent optimization -> decoded image -> DAS/cutouts -> Gemma matching
```

## Important implementation notes

### Image centering differs from the paper

The paper centers image patches using the representation of a neutral gray image. This file does not compute that gray-image patch baseline. Instead, it applies statistics derived from the text baseline-word activations to image patches.

Therefore, even when using the same Gemma layer and the same patch-weighting formula, the image representation is not exactly the representation used in the paper.

### Sentence-related arguments are currently unused

`sentence`, `sentence_weight`, and `instruction` are present in the interface, but they are not currently used to compute an auxiliary language-model loss. The optimization is driven by the representation objective and the VQGAN/image regularizers.

## Reference

Wybitul et al., [“Representations of Text and Images Align From Layer One”](https://arxiv.org/abs/2601.08017), especially Sections 2.1–2.3 and Appendix E.
