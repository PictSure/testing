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
- `model_testing/` - the test harness itself. One directory per dataset, plus
  two top-level entry points that drive all of them:

```
model_testing/
├── download_datasets.sh          # downloads every dataset (runs each download.py)
├── test_models.py                # runs every dataset's tasks against every model
├── harness/                      # shared plumbing: HTTP, task shapes, runner, discovery
└── datasets/
    ├── beans/
    │   ├── download.py           # fetches just this dataset
    │   ├── test_beans.py         # declares just this dataset's tasks; runnable alone
    │   └── data/                 # the images (gitignored)
    ├── braintumor/
    └── ...                       # 16 dataset directories
```

Each dataset directory is self-contained and independently runnable:

```bash
python3 model_testing/datasets/dtd/download.py
python3 model_testing/datasets/dtd/test_dtd.py
```

A dataset is discovered purely by being a directory under `datasets/` with a
`download.py` and a `test_*.py` that defines `build_tasks()`, so adding one
means adding a directory - neither entry point needs editing.

## Running the model tests

Requires Python 3.11+, `torch`, `torchvision`, `transformers`, `Pillow`,
`certifi`, and the [`PictSure`](https://pypi.org/project/PictSure/) package:

```bash
pip install torch torchvision transformers Pillow certifi PictSure
```

Then:

```bash
./model_testing/download_datasets.sh   # ~1200 images into datasets/*/data/ (gitignored)
python3 model_testing/test_models.py
```

`test_models.py` loads each model straight from the Hub and runs it through
`PictSure.from_pretrained(...)` -> `set_context_images(...)` -> `predict(...)`,
exactly as documented in the model cards. It loads each model once and reuses
it across every task, and prints both a per-task summary and a ready-to-paste
markdown results table. Useful flags:

```bash
python3 model_testing/test_models.py --list                    # what would run
python3 model_testing/test_models.py --datasets beans dtd      # a subset
python3 model_testing/test_models.py --models pictsure/pictsure-clip
./model_testing/download_datasets.sh beans dtd                 # download a subset
```

All models used here are public, so no Hugging Face token is required; if you
hit anonymous rate limits, set the `HF_TOKEN` environment variable or run
`huggingface-cli login` first.

Datasets are fetched at the smallest useful size: the per-class subsets the
few-shot tasks actually need (a few hundred KB per dataset), pulled through
the Hugging Face dataset viewer API rather than by downloading whole datasets.

Models tested:

- `pictsure/pictsure-vit`
- `pictsure/pictsure-resnet`
- `pictsure/pictsure-dinov2`
- `pictsure/pictsure-dinov2-large`
- `pictsure/pictsure-clip`

Datasets tested:

| Dataset | Classes | Source | What it tests |
|---|---|---|---|
| `catsdogs` | cat, dog | [pictsure-library/Examples](https://github.com/PictSure/pictsure-library/tree/main/Examples/CatsDogs) | the exact 2-shot example from the model cards |
| `braintumor` | 4 MRI tumor types | [pictsure-library/Examples](https://github.com/PictSure/pictsure-library/tree/main/Examples/BrainTumor_preprocessed) | medical imaging, widest shot sweep (1/3/5/10) |
| `chestxray` | NORMAL, PNEUMONIA | [hf-vision/chest-xray-pneumonia](https://huggingface.co/datasets/hf-vision/chest-xray-pneumonia) | a second medical domain: binary, grayscale radiographs |
| `plantdoc` | 8 tomato leaf diseases | [PlantDoc-Dataset](https://github.com/pratikkayal/PlantDoc-Dataset) | hardest fine-grained task; classes differ only in lesion texture |
| `beans` | 3 bean leaf conditions | [AI-Lab-Makerere/beans](https://huggingface.co/datasets/AI-Lab-Makerere/beans) | the same plant-disease idea, but clean and only 3-way |
| `swedishflowers` | 5 wildflower species | [swedish-flowers-dataset](https://huggingface.co/datasets/renukadevichappidi/swedish-flowers-dataset) | easy control: visually distinct natural classes |
| `caltech101` | 8 classic objects | [dpdl-benchmark/caltech101](https://huggingface.co/datasets/dpdl-benchmark/caltech101) | centered, well-lit objects - closest to pretraining data |
| `oxfordpets` | 8 cat/dog breeds | [timm/oxford-iiit-pet](https://huggingface.co/datasets/timm/oxford-iiit-pet) | fine-grained version of the CatsDogs example |
| `food101` | 8 dish categories | [ethz/food101](https://huggingface.co/datasets/ethz/food101) | fine-grained natural photos, heavy intra-class variation |
| `stanfordcars` | 6 car models | [tanganke/stanford_cars](https://huggingface.co/datasets/tanganke/stanford_cars) | fine-grained ceiling: shared silhouette, badge-level detail |
| `dtd` | 8 texture categories | [tanganke/dtd](https://huggingface.co/datasets/tanganke/dtd) | no object to latch onto - texture statistics only |
| `eurosat` | 10 land-cover classes | [blanchon/EuroSAT_RGB](https://huggingface.co/datasets/blanchon/EuroSAT_RGB) | overhead satellite viewpoint, 64x64 |
| `resisc45` | 8 remote-sensing scenes | [timm/resisc45](https://huggingface.co/datasets/timm/resisc45) | overhead again, but higher-res and object-like |
| `gtsrb` | 8 traffic signs | [tanganke/gtsrb](https://huggingface.co/datasets/tanganke/gtsrb) | 3 speed-limit signs differ only in digits - an OCR probe |
| `cifar10` | 10 coarse objects | [uoft-cs/cifar10](https://huggingface.co/datasets/uoft-cs/cifar10) | 32x32: far below the encoders' native resolution |
| `fashionmnist` | 10 clothing classes | [fashion_mnist](https://huggingface.co/datasets/zalando-datasets/fashion_mnist) | most out-of-domain input: 28x28 grayscale on black |

Shot counts and query counts are set per dataset (in that dataset's
`test_*.py`) so every class always keeps at least one held-out query image,
and so the wide 10-class tasks stay affordable.

## Results

All five encoders ran all 45 tasks end-to-end with no errors - 225 task runs,
zero failures - confirming the documented usage pattern works for every repo,
not just `pictsure-vit`.

Accuracy is shown as a percentage with the exact correct/total count behind it;
the best model per row is **bold** (ties all bolded). Aggregated over all 45
tasks:

| Model | Micro accuracy | Macro accuracy | Best or tied on |
|---|---|---|---|
| `pictsure-vit` | 53.0% | 57.6% | 5/45 tasks |
| `pictsure-resnet` | 28.1% | 33.3% | 1/45 tasks |
| `pictsure-dinov2` | 74.9% | 77.0% | 16/45 tasks |
| `pictsure-dinov2-large` | 75.8% | 76.9% | 22/45 tasks |
| `pictsure-clip` | **77.6%** | **80.2%** | **23/45 tasks** |

Per task:

| Task | vit | resnet | dinov2 | dinov2-large | clip |
|---|---|---|---|---|---|
| Beans (1-shot) | 53% (8/15) | 40% (6/15) | 60% (9/15) | 67% (10/15) | **73% (11/15)** |
| Beans (3-shot) | 47% (7/15) | 40% (6/15) | 47% (7/15) | 33% (5/15) | **87% (13/15)** |
| Beans (5-shot) | 60% (9/15) | 40% (6/15) | **87% (13/15)** | 60% (9/15) | 80% (12/15) |
| Beans (10-shot) | 50% (3/6) | 67% (4/6) | 67% (4/6) | 67% (4/6) | **100% (6/6)** |
| BrainTumor (1-shot) | 40% (8/20) | 50% (10/20) | 45% (9/20) | 40% (8/20) | **60% (12/20)** |
| BrainTumor (3-shot) | 70% (14/20) | 70% (14/20) | 45% (9/20) | 50% (10/20) | **75% (15/20)** |
| BrainTumor (5-shot) | 65% (13/20) | 50% (10/20) | 45% (9/20) | 55% (11/20) | **85% (17/20)** |
| BrainTumor (10-shot) | 70% (14/20) | 75% (15/20) | **80% (16/20)** | **80% (16/20)** | **80% (16/20)** |
| Caltech101 (1-shot) | 58% (14/24) | 17% (4/24) | 88% (21/24) | **92% (22/24)** | 79% (19/24) |
| Caltech101 (3-shot) | 83% (20/24) | 33% (8/24) | **100% (24/24)** | **100% (24/24)** | **100% (24/24)** |
| Caltech101 (5-shot) | 75% (18/24) | 21% (5/24) | **100% (24/24)** | **100% (24/24)** | **100% (24/24)** |
| CatsDogs (README example, 2-shot) | **100% (1/1)** | **100% (1/1)** | **100% (1/1)** | **100% (1/1)** | **100% (1/1)** |
| ChestXray (1-shot) | **90% (9/10)** | 50% (5/10) | **90% (9/10)** | 80% (8/10) | **90% (9/10)** |
| ChestXray (3-shot) | **100% (10/10)** | 60% (6/10) | 90% (9/10) | 90% (9/10) | 90% (9/10) |
| ChestXray (5-shot) | **100% (10/10)** | 80% (8/10) | 90% (9/10) | 70% (7/10) | 80% (8/10) |
| ChestXray (10-shot) | **100% (4/4)** | 75% (3/4) | **100% (4/4)** | **100% (4/4)** | 50% (2/4) |
| CIFAR10 (1-shot) | 13% (4/30) | 13% (4/30) | 47% (14/30) | **60% (18/30)** | 47% (14/30) |
| CIFAR10 (5-shot) | 53% (16/30) | 17% (5/30) | 77% (23/30) | **90% (27/30)** | 77% (23/30) |
| DTD (1-shot) | 50% (12/24) | 12% (3/24) | **88% (21/24)** | 79% (19/24) | 83% (20/24) |
| DTD (3-shot) | 62% (15/24) | 21% (5/24) | **100% (24/24)** | **100% (24/24)** | 88% (21/24) |
| DTD (5-shot) | 75% (18/24) | 4% (1/24) | **96% (23/24)** | 92% (22/24) | 92% (22/24) |
| EuroSAT (1-shot) | 40% (12/30) | 10% (3/30) | 33% (10/30) | **47% (14/30)** | 30% (9/30) |
| EuroSAT (5-shot) | 67% (20/30) | 33% (10/30) | 67% (20/30) | **77% (23/30)** | **77% (23/30)** |
| FashionMNIST (1-shot) | 37% (11/30) | 40% (12/30) | 70% (21/30) | **73% (22/30)** | 50% (15/30) |
| FashionMNIST (5-shot) | 43% (13/30) | 73% (22/30) | 73% (22/30) | **77% (23/30)** | 67% (20/30) |
| Food101 (1-shot) | 29% (7/24) | 8% (2/24) | 75% (18/24) | **88% (21/24)** | 75% (18/24) |
| Food101 (3-shot) | 33% (8/24) | 21% (5/24) | 83% (20/24) | **92% (22/24)** | 83% (20/24) |
| Food101 (5-shot) | 54% (13/24) | 8% (2/24) | **92% (22/24)** | **92% (22/24)** | 88% (21/24) |
| GTSRB (1-shot) | 58% (14/24) | 17% (4/24) | 50% (12/24) | 62% (15/24) | **75% (18/24)** |
| GTSRB (3-shot) | 54% (13/24) | 29% (7/24) | 67% (16/24) | 75% (18/24) | **88% (21/24)** |
| GTSRB (5-shot) | 62% (15/24) | 25% (6/24) | 88% (21/24) | 92% (22/24) | **96% (23/24)** |
| OxfordPets (1-shot) | 25% (6/24) | 8% (2/24) | **92% (22/24)** | **92% (22/24)** | 75% (18/24) |
| OxfordPets (3-shot) | 21% (5/24) | 17% (4/24) | **100% (24/24)** | **100% (24/24)** | 88% (21/24) |
| OxfordPets (5-shot) | 50% (12/24) | 21% (5/24) | **100% (24/24)** | **100% (24/24)** | **100% (24/24)** |
| PlantDoc (1-shot) | 21% (8/38) | 18% (7/38) | **39% (15/38)** | 24% (9/38) | 26% (10/38) |
| PlantDoc (3-shot) | 21% (7/33) | 9% (3/33) | 30% (10/33) | 30% (10/33) | **39% (13/33)** |
| RESISC45 (1-shot) | 71% (17/24) | 25% (6/24) | 79% (19/24) | 79% (19/24) | **92% (22/24)** |
| RESISC45 (3-shot) | 75% (18/24) | 46% (11/24) | 92% (22/24) | 88% (21/24) | **96% (23/24)** |
| RESISC45 (5-shot) | 92% (22/24) | 38% (9/24) | 92% (22/24) | 92% (22/24) | **100% (24/24)** |
| StanfordCars (1-shot) | 39% (7/18) | 22% (4/18) | 39% (7/18) | 33% (6/18) | **100% (18/18)** |
| StanfordCars (3-shot) | 44% (8/18) | 6% (1/18) | 83% (15/18) | 61% (11/18) | **94% (17/18)** |
| StanfordCars (5-shot) | 61% (11/18) | 22% (4/18) | 89% (16/18) | 83% (15/18) | **94% (17/18)** |
| SwedishFlowers (1-shot) | 52% (13/25) | 24% (6/25) | **100% (25/25)** | **100% (25/25)** | 84% (21/25) |
| SwedishFlowers (3-shot) | 60% (15/25) | 24% (6/25) | 96% (24/25) | **100% (25/25)** | 88% (22/25) |
| SwedishFlowers (5-shot) | 67% (16/24) | 21% (5/24) | 96% (23/24) | **100% (24/24)** | 92% (22/24) |

Takeaways:

- **The encoder matters far more than the shot count.** This is the paper's own
  claim, and it shows up bluntly here: `pictsure-resnet` averages 28% micro
  accuracy against CLIP's 78%, and adding shots does not close that gap. On DTD
  the ResNet variant actually gets *worse* with more context (12% -> 4%).
- **CLIP and DINOv2-large split the wins along domain lines.** CLIP leads where
  the class distinction is semantic or textual - traffic signs (96% at 5-shot,
  including three speed-limit signs that differ only in their digits), remote
  sensing, car models, brain MRI, bean disease. DINOv2/DINOv2-large lead on
  object-centric natural images and on the degraded inputs (CIFAR-10,
  Fashion-MNIST), where CLIP's text-aligned features have less to grip.
- **Out-of-domain inputs cost less than expected.** Fashion-MNIST at 28x28
  grayscale still reaches 77% (DINOv2-large, 5-shot) over ten classes, and
  CIFAR-10 at 32x32 reaches 90%. Resolution mismatch degrades these encoders
  gracefully rather than breaking them.
- **Fine-grained difficulty is not one axis.** OxfordPets (breed-level) is
  solved - 100% for both DINOv2 variants at 3-shot - while PlantDoc (lesion
  texture) stays at 21-39% for everything. The distinguishing feature has to be
  *represented* in the embedding; more shots cannot invent it.
- **StanfordCars is the one number to distrust.** CLIP scores 100% (18/18) at
  1-shot on six car models, above its own 3- and 5-shot scores. With 18 queries
  and probable overlap with CLIP's pretraining data, read this as a signal about
  the pretraining corpus, not about few-shot ability.
- **More shots usually help, but not monotonically.** BrainTumor, GTSRB,
  RESISC45, Food101 and CIFAR-10 all trend up with shots; Beans and ChestXray
  wobble. Single-query and few-query tasks (CatsDogs, the 10-shot rows) are
  sanity checks, not measurements.

Reproducibility notes:

- The four datasets carried over from the previous version of this harness
  (CatsDogs, BrainTumor, PlantDoc, SwedishFlowers) reproduce their earlier
  numbers exactly, apart from one SwedishFlowers ResNet query. The harness now
  calls `Image.open(...).convert("RGB")` before every prediction, which is
  required for the grayscale datasets (ChestXray, Fashion-MNIST) and changes
  the input for the few non-RGB images elsewhere.
- Run on Apple Silicon via the MPS backend, which `PictSure` selects
  automatically. The full sweep takes roughly 15 minutes; the dataset download
  takes considerably longer than the tests do.

## License

[MIT](LICENSE)
