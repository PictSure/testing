# Testing of PictSure

A working repo for validating and maintaining the model cards of the
[PictSure](https://pictsure.eu/) model collection on Hugging Face
([`pictsure/pictsure-10`](https://huggingface.co/collections/pictsure/pictsure-10)),
plus a few models outside that collection (`pictsure-resnet-base`).

PictSure is a few-shot image classification model that uses in-context
learning: you give it a handful of labeled "context" images and it classifies
new images against those classes, no fine-tuning required. See the
[paper](https://arxiv.org/abs/2506.14842) and the
[pictsure-library](https://github.com/PictSure/pictsure-library) for details.

## Layout

- `repos/` - local clones of the Hugging Face model repos (`pictsure-vit`,
  `pictsure-resnet`, `pictsure-dinov2`, `pictsure-clip`, `pictsure-dinov2-large`).
  Used to review and push README updates. Note: this machine doesn't have
  `git-lfs` installed, so `model.safetensors` in these clones is a pointer
  file only, not the real weights - fine for editing docs, not for loading
  the models directly from `repos/`.
- `model_testing/` - a standalone harness that actually runs every model
  through the few-shot classification usage pattern shown in its README,
  across several datasets and shot counts, so README changes get validated
  against real model behavior before being pushed live.

## Running the model tests

```bash
./model_testing/download_datasets.sh   # fetches datasets into model_testing/datasets/ (gitignored)
python3 model_testing/test_models.py
```

`test_models.py` loads each model straight from the Hub (using the token in
`~/.hf_credentials`) and runs it through `PictSure.from_pretrained(...)` ->
`set_context_images(...)` -> `predict(...)`, exactly as documented in the
model cards.

Models tested:

- `pictsure/pictsure-vit`
- `pictsure/pictsure-resnet`
- `pictsure/pictsure-resnet-base`
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

| Task | vit | resnet | resnet-base | dinov2 | dinov2-large | clip |
|---|---|---|---|---|---|---|
| CatsDogs (2-shot) | **100% (1/1)** | **100% (1/1)** | 0% (0/1) | **100% (1/1)** | **100% (1/1)** | **100% (1/1)** |
| BrainTumor (1-shot) | 40% (8/20) | 50% (10/20) | 50% (10/20) | 45% (9/20) | 40% (8/20) | **60% (12/20)** |
| BrainTumor (3-shot) | 70% (14/20) | 70% (14/20) | 40% (8/20) | 45% (9/20) | 50% (10/20) | **75% (15/20)** |
| BrainTumor (5-shot) | 65% (13/20) | 50% (10/20) | 70% (14/20) | 45% (9/20) | 55% (11/20) | **85% (17/20)** |
| BrainTumor (10-shot) | 70% (14/20) | 75% (15/20) | 55% (11/20) | **80% (16/20)** | **80% (16/20)** | **80% (16/20)** |
| PlantDoc (1-shot) | 21% (8/38) | 18% (7/38) | 18% (7/38) | **39% (15/38)** | 24% (9/38) | 26% (10/38) |
| PlantDoc (3-shot) | 21% (7/33) | 9% (3/33) | 27% (9/33) | 30% (10/33) | 30% (10/33) | **39% (13/33)** |
| SwedishFlowers (1-shot) | 52% (13/25) | 24% (6/25) | 56% (14/25) | **100% (25/25)** | **100% (25/25)** | 84% (21/25) |
| SwedishFlowers (3-shot) | 60% (15/25) | 20% (5/25) | 52% (13/25) | 96% (24/25) | **100% (25/25)** | 88% (22/25) |
| SwedishFlowers (5-shot) | 67% (16/24) | 21% (5/24) | 54% (13/24) | 96% (23/24) | **100% (24/24)** | 92% (22/24) |

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
