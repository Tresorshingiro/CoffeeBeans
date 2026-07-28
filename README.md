# Coffee Bean Grading — MLOps Pipeline

Grades green Arabica coffee beans into four quality classes from a single
photograph: **defect**, **longberry**, **peaberry**, **premium**.

Coffee is one of Rwanda's largest exports and grading is still done by eye at
most cooperatives. This is an end-to-end pipeline — training, serving,
monitoring, and retraining on new data — around a model that does it
automatically.

- **Live app:** _TODO: paste your Hugging Face Space URL_
- **API docs:** _TODO: `<space-url>/docs`_
- **Video demo:** _TODO: paste your YouTube link_

## Model

MobileNetV2 transfer learning on the USK-Coffee dataset (Febriana et al.,
2022): 8,000 images, 256×256, balanced across four classes.

Training ran in two phases — a frozen ImageNet base, then fine-tuning the top
30 layers at 1e-5 with BatchNorm held frozen. Optimizations used: transfer
learning, dropout regularization, early stopping on validation loss,
`ReduceLROnPlateau` scheduling, and checkpointing that only overwrites on
improvement.

### Model artifacts

| Path | Format | Role |
|---|---|---|
| `models/coffee_model.keras` | Keras v3 native, 21MB | **The live artifact** — what the API loads and what retraining starts from |
| `models/best.weights.h5` | HDF5 weights, 21MB | The best checkpoint the notebook saved, across both training phases |
| `models/coffee_model_tf/` | TensorFlow SavedModel, 20MB | Portable export for tooling that expects SavedModel |

All three carry the **same weights**. The notebook restores the best checkpoint
(`model.load_weights("models/best.weights.h5")`) before evaluating and before
saving the `.keras` file, so the deployed model is the best model produced.
Verified directly — scored on the same 400-image slice, `coffee_model.keras`
and `best.weights.h5` both give **accuracy 0.8675, loss 0.3873**, and their
weight tensors are element-wise identical.

To load the `.h5` you need the architecture, since it is weights-only:

```python
import tensorflow as tf
model = tf.keras.models.load_model("models/coffee_model.keras")  # or src.model.build_model()
model.load_weights("models/best.weights.h5")
```

A *full-model* `.h5` save is not possible here, and that is a consequence of the
design rather than an oversight: augmentation and `preprocess_input` live
*inside* the model graph so training, the notebook, and the API cannot drift in
how they preprocess. The legacy HDF5 serializer cannot represent those layers
and fails with `TypeError: cannot pickle 'module' object`. `.keras` is the
supported successor to `.h5` and handles them correctly.

Regenerate the SavedModel export with:

```python
import tensorflow as tf
tf.keras.models.load_model("models/coffee_model.keras").export("models/coffee_model_tf")
```

### Test set results (1,600 held-out images)

| Metric | Value |
|---|---|
| Accuracy | 0.865 |
| Loss | 0.402 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| defect | 0.906 | 0.767 | 0.831 | 400 |
| longberry | 0.905 | 0.930 | 0.917 | 400 |
| peaberry | 0.869 | 0.895 | 0.882 | 400 |
| premium | 0.792 | 0.868 | 0.828 | 400 |

The deployed service re-scores the same model on a 400-image stratified slice
at startup and gets **0.8675**, confirming the serving path and the notebook
agree.

## What the data says

**1. Class balance.** Exactly 1,600 training images per class. Plain accuracy
is therefore an honest headline metric and no resampling or class weighting was
needed.

**2. Bean colour.** Averaged over the *bean pixels only* — masking out the
white backdrop, which otherwise dominates the frame and washes every class out
to the same value — longberry is clearly the lightest class (brightness 137
against 124–129 for the rest). The revealing part is the bottom of the chart:
defect (124.9) and peaberry (124.5) are almost indistinguishable by colour.

**3. Bean area.** The share of the frame the bean occupies supplies what colour
could not. Peaberry is the most compact class (0.205) and the most consistent
(std 0.047); premium is the largest (0.268). So peaberry, which colour could
not separate from defect, is cleanly separable by size.

**The story:** read together, these predicted the model's weakness before
training ran. Every other class owns a distinctive corner of the feature space
— longberry the lightest, peaberry the smallest and most consistent, premium
the largest. Defect owns nothing: mid-range in colour, mid-range in area, wide
variance in both, overlapping all three neighbours. That is exactly what the
confusion matrix went on to show — defect had the lowest recall of any class at
0.767, against longberry's 0.930.

## Architecture

```
Browser ──► nginx ──┬──► Streamlit UI (:8501)
                    └──► FastAPI (:8000) ──► TensorFlow model
                                         └──► SQLite
Locust  ──► nginx ──► FastAPI replicas
```

## Retraining pipeline

1. **Upload** — a ZIP with class-named folders. Each file is validated against
   the four known classes, checked for decodability, guarded against
   decompression bombs, deduped by SHA-256, staged to `data/pending/<class>/`,
   and recorded in SQLite.
2. **Trigger** — retraining unlocks automatically once 50 images are staged;
   a force option overrides the threshold manually for demos.
3. **Preprocess** — staged images are decoded, resized to 224×224, and mixed
   with a 400-image stratified replay sample from the original training set so
   the model cannot forget classes absent from the upload.
4. **Retrain** — the **current champion `.keras` model is loaded as the
   starting point**, BatchNorm stays frozen, and it fine-tunes for 3 epochs at
   1e-5.
5. **Gate** — the candidate is scored on a fixed, seeded held-out test slice
   and is promoted **only if it beats the champion**. On promotion the staged
   images join the training set and the live model hot-swaps with no restart.
   On rejection the champion and the staged images are both left untouched, so
   more data can be added and the run retried.

Labels are assigned explicitly from a single ordered `CLASS_NAMES` constant
rather than inferred from directory contents. Letting Keras infer them means an
upload containing only `defect/` and `premium/` maps them to indices 0 and 1 —
and index 1 means `longberry` to the model, so retraining would silently learn
wrong labels.

## Setup

### Local

```bash
git clone <repo-url> && cd CoffeeBeans
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt

# Optional: fetch the full dataset (the repo ships a 1,200-image sample)
HF_DATA_REPO=<your-dataset-repo> .venv/bin/python scripts/download_data.py

.venv/bin/python scripts/build_insights.py
.venv/bin/python scripts/register_baseline.py   # required: arms the promotion gate
.venv/bin/pytest tests/ -v

.venv/bin/uvicorn api.main:app --port 8000 &
.venv/bin/streamlit run ui/app.py --server.port 8501
```

UI at http://localhost:8501, API docs at http://localhost:8000/docs.

### Docker

```bash
docker-compose up --build
```

Everything is served through http://localhost:8090 — UI at `/`, API at `/api/`,
Swagger at `/docs`.

**Two environment notes.** Port 8090 is used rather than the usual 8080, which
is often already taken (Tomcat and other dev servers) — the stack still starts
in that case, but every API call returns the *other* server's 404 page, which
is a confusing failure. And `docker-compose` v1.29.2 raises
`KeyError: 'ContainerConfig'` against Docker Engine 29.x when recreating
containers that have volumes; remove them first rather than recreating in place:

```bash
docker rm -f $(docker ps -aq --filter name=coffeebeans_)
docker-compose up -d
```

### Cloud (Hugging Face Spaces)

Build from `docker/Dockerfile.space`, which runs nginx, FastAPI, and Streamlit
behind the single port a Space exposes. `README_HF.md` carries the required
Space frontmatter. Set `HF_TOKEN` and `HF_MODEL_REPO` as Space secrets so
retrained models survive restarts.

## Flood request simulation

Locust, identical profile per run: **50 users, spawn rate 5/s, 2 minutes**,
90% `POST /api/predict` and 10% `GET /api/status`, all through nginx.

**Host:** Intel Core i5-7300HQ @ 2.50GHz, 4 cores, 7.6GB RAM, Docker 29.1.3,
docker-compose 1.29.2.

Each replica is pinned to **1.0 CPU** with `TF_NUM_INTRAOP_THREADS=1`. Without
this, TensorFlow claims every core in a single container, one replica saturates
the host, and adding more shows no gain — the pinning is what makes replica
count a genuine independent variable rather than a flat table.

| Containers | RPS | Median (ms) | p95 (ms) | p99 (ms) | Requests | Failures |
|---|---|---|---|---|---|---|
| 1 | 8.19 | 5,000 | 12,000 | 17,000 | 975 | 0 |
| 2 | **14.13** | 2,600 | 6,200 | 12,000 | 1,672 | 0 |
| 4 | 10.71 | 1,900 | 15,000 | 20,000 | 1,280 | 0 |

### Findings

**Scaling from 1 to 2 replicas works as expected.** Throughput rose 1.73×
(8.19 → 14.13 RPS) and median latency almost halved (5,000 → 2,600 ms). A
second container is a second unit of inference capacity, and the load splits
across both.

**Scaling to 4 replicas made throughput worse, not better** — 10.71 RPS, below
the 2-replica result. This is a host-capacity ceiling rather than a flaw in the
service. Four replicas pinned to 1.0 CPU each demand all 4 physical cores, but
they are not the only things running: nginx, the Streamlit UI, unrelated
containers, and **Locust itself** all compete for the same 4 cores. Past two
replicas the machine is oversubscribed and time spent context-switching
outweighs the added parallelism. The tail latencies show it plainly — p95 more
than doubled (6,200 → 15,000 ms) and p99 reached 20 seconds, even as the median
kept falling to 1,900 ms. Requests that get a worker promptly are served
faster; requests that queue behind a descheduled worker wait much longer.

**No request failed at any replica count** (3,927 requests total), so the
degradation is latency, never errors.

**Methodological caveat:** the load generator shares a host with the service.
On a 4-core machine that materially affects the 4-replica figure. A cleaner
experiment would drive load from a separate machine, or cap replicas at
`cores - 1`. On this hardware, **2 replicas is the optimum**.

Reproduce with `./locust/run_load_tests.sh`.

## Repository structure

```
├── README.md
├── notebook/coffee_bean_classification.ipynb
├── src/{config,preprocessing,model,prediction,database}.py
├── api/main.py
├── ui/app.py
├── data/{train,test}/          1,200-image sample; full set via scripts/
├── models/coffee_model.keras   + coffee_model_tf/ (SavedModel export)
├── scripts/{make_sample,download_data,build_insights,register_baseline}.py
├── locust/
├── docker/
└── tests/                      72 tests
```

## Testing

```bash
.venv/bin/pytest tests/ -v
```

72 tests covering upload validation (including zip-slip and decompression-bomb
rejection), the label-ordering regression, dataset determinism, the promotion
gate in both directions, and every API endpoint's success and error paths.

## Known limitations

- **Retraining runs in the API process**, so prediction latency degrades while
  a job is in flight. The fix is a separate worker process; it was out of scope
  here. Load tests were run with no retrain active.
- **Space storage is ephemeral, and persistence is only half-implemented.**
  When `HF_TOKEN` and `HF_MODEL_REPO` are set, a promoted model is *uploaded*
  to a Hugging Face Model repo, so it is never lost. But there is no restore
  path: on restart the container has no `data/app.db`, `scripts/register_baseline.py`
  re-registers the original `models/coffee_model.keras`, and the Space serves
  the baseline again. The retrained model has to be fetched from the Model repo
  manually. Uploaded images are not backed up at all, so `data/pending/` and any
  images migrated into `data/train/` reset on restart as well.
  Closing this needs a restore step at startup that pulls the newest
  `coffee_model_*.keras` from the Model repo, scores it, and registers it as
  champion before the API starts.
- **The decompression-bomb guard trusts the ZIP header's declared
  `file_size`**, which a crafted archive could understate. A fully robust guard
  would read through a capped stream instead.
- **Predictions are only trustworthy on single-bean photos on a plain
  background**, matching the training data. Multi-bean or cluttered images are
  out of distribution.
- **The load test runs locally**, not on the Space — a free Space is a single
  container with no replica control.
