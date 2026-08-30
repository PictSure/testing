# Testing of PictSure

An independent test harness for [PictSure](https://pictsure.eu/), a few-shot
image classification method that uses in-context learning: you give the
model a handful of labeled "context" images per class, and it classifies new
images against those classes directly, with no fine-tuning or gradient
updates. It works by combining a pre-trained image encoder (ViT, ResNet,
DINOv2 or CLIP) with a transformer that attends over the context set, as
described in

> Schiesser, L., Wolff, C., Haas, S., & Pukrop, S. (2025). *PictSure:
> Pretraining Embeddings Matters for In-Context Learning Image Classifiers*.
> [arXiv:2506.14842](https://arxiv.org/abs/2506.14842)

The pre-trained models are published on Hugging Face as the
[`pictsure/pictsure-10`](https://huggingface.co/collections/pictsure/pictsure-10)
collection. Code and usage examples live in
[pictsure-library](https://github.com/PictSure/pictsure-library).

This repo validates that the usage pattern shown in each model's README
actually works, for every published encoder, across multiple datasets and
shot counts - not just the single example the docs happen to show.

## Layout

- `repos/` - scratch space for local clones of the Hugging Face model repos,
  used when reviewing or pushing README/model-card updates. Not committed
  (see `.gitignore`) and not needed to run the tests below. If your local
  `git` has no `git-lfs` installed, `model.safetensors` in these clones will
  be a pointer file only, not the real weights - fine for editing docs, not
  for loading the models directly from a clone.
- `model_testing/` - the test harness itself: scripts to fetch datasets and
  to run every model through the few-shot classification usage pattern from
  its README, so doc changes get validated against real model behavior
  before being published.

## Running the model tests

Requires Python 3.11+, `torch`, `torchvision`, `transformers`, `Pillow`, and
the [`PictSure`](https://pypi.org/project/PictSure/) package:

```bash
pip install torch torchvision transformers Pillow PictSure
```

Then:

```bash
./model_testing/download_datasets.sh   # fetches datasets into model_testing/datasets/ (gitignored)
python3 model_testing/test_models.py
```

`test_models.py` loads each model straight from the Hub and runs it through
`PictSure.from_pretrained(...)` -> `set_context_images(...)` ->
`predict(...)`, exactly as documented in the model cards. All models used
here are public, so no Hugging Face token is required; if you hit anonymous
rate limits, set the `HF_TOKEN` environment variable or run
`huggingface-cli login` first.

Models tested:

- `pictsure/pictsure-vit`
- `pictsure/pictsure-resnet`
- `pictsure/pictsure-dinov2`
- `pictsure/pictsure-dinov2-large`
- `pictsure/pictsure-clip`

Datasets tested:

| Dataset | Classes | Source | What it tests |
|---|---|---|---|
| CatsDogs | cat, dog | [pictsure-library/Examples](https://github.com/PictSure/pictsure-library/tree/main/Examples/CatsDogs) | the exact 2-shot example from the README |
| BrainTumor_preprocessed | glioma, meningioma, notumor, pituitary | [pictsure-library/Examples](https://github.com/PictSure/pictsure-library/tree/main/Examples/BrainTumor_preprocessed) | 4-way MRI classification, 1/3/5/10-shot |
| PlantDoc | 8 tomato leaf disease subtypes | [PlantDoc-Dataset](https://github.com/pratikkayal/PlantDoc-Dataset) | fine-grained, visually similar classes, 1/3-shot |
| SwedishFlowers | 5 wildflower species | [swedish-flowers-dataset](https://huggingface.co/datasets/renukadevichappidi/swedish-flowers-dataset) | visually distinct natural classes, 1/3/5-shot |

Shot counts are capped per dataset so every class always keeps at least one
held-out query image.

## Results

All models ran every task end-to-end with no errors, confirming the README
usage pattern works for every repo, not just `pictsure-vit`. Accuracy shown
as percentage, with the exact correct/total count behind it; the best model
per row is **bold** (ties all bolded):

| Task | vit | resnet | dinov2 | dinov2-large | clip |
|---|---|---|---|---|---|
| CatsDogs (2-shot) | **100% (1/1)** | **100% (1/1)** | **100% (1/1)** | **100% (1/1)** | **100% (1/1)** |
| BrainTumor (1-shot) | 40% (8/20) | 50% (10/20) | 45% (9/20) | 40% (8/20) | **60% (12/20)** |
| BrainTumor (3-shot) | 70% (14/20) | 70% (14/20) | 45% (9/20) | 50% (10/20) | **75% (15/20)** |
| BrainTumor (5-shot) | 65% (13/20) | 50% (10/20) | 45% (9/20) | 55% (11/20) | **85% (17/20)** |
| BrainTumor (10-shot) | 70% (14/20) | 75% (15/20) | **80% (16/20)** | **80% (16/20)** | **80% (16/20)** |
| PlantDoc (1-shot) | 21% (8/38) | 18% (7/38) | **39% (15/38)** | 24% (9/38) | 26% (10/38) |
| PlantDoc (3-shot) | 21% (7/33) | 9% (3/33) | 30% (10/33) | 30% (10/33) | **39% (13/33)** |
| SwedishFlowers (1-shot) | 52% (13/25) | 24% (6/25) | **100% (25/25)** | **100% (25/25)** | 84% (21/25) |
| SwedishFlowers (3-shot) | 60% (15/25) | 20% (5/25) | 96% (24/25) | **100% (25/25)** | 88% (22/25) |
| SwedishFlowers (5-shot) | 67% (16/24) | 21% (5/24) | 96% (23/24) | **100% (24/24)** | 92% (22/24) |

Takeaways:

- **SwedishFlowers** (visually distinct species) is easy for every encoder
  except plain ResNet; DINOv2 and DINOv2-large get it essentially perfect.
- **PlantDoc** (subtle differences between tomato disease subtypes) is
  genuinely hard for all models - low double-digit accuracy is expected for
  a fine-grained few-shot task with only 1-3 examples per class.
- **BrainTumor** accuracy generally trends upward with more shots, as
  expected for in-context learning.
- Single query images (CatsDogs, 1-shot tasks) are noisy signals on their
  own - treat individual numbers as sanity checks, not benchmarks.

## License

[MIT](LICENSE)
