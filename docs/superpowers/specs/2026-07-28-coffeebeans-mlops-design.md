# Coffee Bean Grading — MLOps Pipeline Design

**Date:** 2026-07-28
**Status:** Approved, ready for implementation planning

## 1. Context

An end-to-end ML pipeline that grades green Arabica coffee beans into four classes —
`defect`, `longberry`, `peaberry`, `premium` — from a single photograph. Dataset is
USK-Coffee (Febriana et al., 2022): 8,000 images at 256×256, evenly balanced.

The model-building half already exists in `notebook/coffee_bean_classification.ipynb`:
MobileNetV2 transfer learning, two-phase training (frozen base, then fine-tuning the top
30 layers with BatchNorm held frozen), reaching **0.865 test accuracy** with per-class F1
of longberry 0.917, peaberry 0.882, defect 0.831, premium 0.828.

This spec covers the deployment half: `src/` modules, a FastAPI service, a Streamlit UI,
a retraining pipeline with a promotion gate, Docker packaging, a Locust load-test
methodology, and the README.

### Decisions already made

| Decision | Choice |
|---|---|
| Cloud platform | Hugging Face Spaces (Docker SDK, free tier: 2 vCPU / 16GB RAM) |
| UI architecture | Streamlit UI calling a separate FastAPI service |
| Bulk upload format | ZIP archive with class-named subfolders |
| Retrain strategy | Pending data + replay slice, gated on test accuracy |
| Repo data | Sampled subset committed, full dataset fetched by script |
| Retrain execution | FastAPI background task + status polling |

## 2. Repository structure

```
CoffeeBeans/
├── README.md
├── requirements.txt
├── docker-compose.yml
├── Dockerfile                       API image (scaled locally)
├── .gitignore
├── notebook/
│   └── coffee_bean_classification.ipynb
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── prediction.py
│   └── database.py
├── api/
│   └── main.py
├── ui/
│   └── app.py
├── data/
│   ├── train/<class>/               200 images per class (committed)
│   ├── test/<class>/                100 images per class (committed)
│   └── pending/<class>/             staged uploads (gitignored)
├── models/
│   ├── coffee_model.keras           champion
│   └── best.weights.h5
├── locust/
│   └── locustfile.py
├── scripts/
│   ├── download_data.py
│   ├── make_sample.py
│   └── build_insights.py
└── docker/
    ├── nginx.conf
    ├── nginx.space.conf
    └── Dockerfile.space
```

The three rubric-mandated paths — `src/preprocessing.py`, `src/model.py`,
`src/prediction.py` — stay exactly where the assignment specifies. Everything else sits
alongside.

### Housekeeping carried into implementation

- Rename `models/coffee_model (1).keras` to `models/coffee_model.keras`. The space and
  `(1)` are a browser-download artifact and will break path handling.
- Delete the stray `data/{train,test}/` directory, an unescaped brace-expansion artifact.
  It is empty and would confuse any directory-walking code.
- Add `.gitignore` before the first commit. The working tree currently holds 124MB of
  images and model binaries across untracked `data/` and `models/`; `git add .` on an
  empty repo would commit all of it.

## 3. Data strategy

**Committed to Git:** a stratified sample of 200 training and 100 test images per class
(1,200 images, roughly 12MB). Generated once by `scripts/make_sample.py` with a fixed
seed so it is reproducible.

**Full dataset:** `scripts/download_data.py` pulls all 8,000 images from a Hugging Face
Dataset repo that the user creates. HF Datasets is preferred over Kaggle because it needs
no API credentials baked into the Docker build, and it is the same infrastructure the
Space runs on.

**At Space build time** the Dockerfile runs `download_data.py`. If it succeeds, the
replay pool and promotion gate use the full dataset and the gate's accuracy figure is
directly comparable to the notebook's 0.865. If it fails, the service falls back to the
committed sample and logs a warning at startup. The service is functional either way; the
fallback only makes the gate's estimate noisier.

### Which test data each consumer uses

Two consumers evaluate against test data, with different speed requirements:

- **The promotion gate (§6)** always uses a fixed stratified slice of `EVAL_SLICE_SIZE`
  (400) images, seeded so every run scores against identical data and accuracy figures are
  comparable across runs. On a 2-vCPU Space this takes roughly 25 seconds, keeping the
  retrain demo near two minutes rather than four. When only the committed sample is
  present, the slice is the whole sample.
- **`/api/metrics` (§5)** evaluates against the full available test set — 1,600 images
  when the download succeeded, 400 otherwise — because it is called on demand and its
  numbers are meant to be compared directly against the notebook's 0.865.

## 4. Module design

### `src/config.py`

Central constants. Load-bearing despite its size.

```python
CLASS_NAMES = ["defect", "longberry", "peaberry", "premium"]  # explicit, ordered
IMG_SIZE = (224, 224)
BATCH = 32
RETRAIN_THRESHOLD = 50
REPLAY_SAMPLES = 400        # drawn from data/train/ during retrain
EVAL_SLICE_SIZE = 400       # stratified test images used by the promotion gate
RETRAIN_EPOCHS = 3
RETRAIN_LR = 1e-5
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
```

`CLASS_NAMES` must be passed to every `image_dataset_from_directory` call. Without it,
Keras derives labels alphabetically from whichever subfolders happen to exist — so an
upload containing only `defect/` and `premium/` would map them to indices 0 and 1, where
index 1 means `longberry` to the model. Retraining would then actively teach the model
wrong labels. This is a live bug in the notebook's current `retrain()` and the explicit
constant is the fix.

### `src/preprocessing.py`

| Function | Behaviour |
|---|---|
| `load_dataset(directory, **kw)` | Wraps `image_dataset_from_directory`, always with `class_names=CLASS_NAMES` |
| `build_augmentation()` | The `Sequential` block from the notebook: RandomFlip, RandomRotation(0.1), RandomZoom(0.1), RandomContrast(0.1) |
| `stage_upload(zip_bytes, batch_id)` | Validates archive, rejects unknown class folders and undecodable files, dedupes by SHA-256, writes to `data/pending/<class>/`, returns per-class accepted/rejected counts with reasons |
| `sample_replay(n)` | Stratified random draw of `n` images from `data/train/` |
| `decode_image_bytes(b)` | `tf.io.decode_image(..., expand_animations=False)`, resize to `IMG_SIZE` |

`decode_image_bytes` deliberately uses `decode_image` rather than the notebook's
`decode_jpeg`. User uploads will include PNGs and `decode_jpeg` raises on them.

### `src/model.py`

| Function | Behaviour |
|---|---|
| `build_model()` | MobileNetV2 base + GAP + Dropout(0.3) + Dense(4, softmax); augmentation and `preprocess_input` inside the graph |
| `train_frozen(...)` | Phase 1, mirrors the notebook |
| `finetune(...)` | Phase 2, top 30 layers unfrozen with BatchNorm held frozen |
| `evaluate(model, ds)` | Returns accuracy, loss, per-class precision/recall/F1, confusion matrix |
| `retrain(pending_dir, replay_n, epochs, progress_cb)` | Full pipeline, see §6 |
| `promote(candidate, metrics)` | Timestamped save, registry insert, pending migration, HF push |

Augmentation and `preprocess_input` stay inside the model graph, as in the notebook, so
training, the notebook, and the API cannot drift in how they preprocess.

### `src/prediction.py`

Loads the champion model once at process startup and holds it in a module-level
reference. Loading per request would dominate latency and invalidate the Locust results.

```python
predict_image(image_bytes) -> {
    "class": str,
    "confidence": float,
    "probabilities": {class_name: float, ...},
    "latency_ms": float,
    "model_version": str,
}
```

A module-level `threading.Lock` guards the model reference so promotion can swap it
atomically without breaking in-flight predictions.

### `src/database.py`

SQLite at `data/app.db` via stdlib `sqlite3`. No ORM. Connections are per-request with
`check_same_thread=False` and WAL mode enabled, since the background retrain thread writes
concurrently with request handlers.

```sql
uploads(id, batch_id, filename, class_label, sha256, uploaded_at, used_in_run)
retrain_runs(id, started_at, finished_at, status, n_pending, n_replay, epochs,
             candidate_accuracy, champion_accuracy, promoted, model_path, log)
predictions(id, ts, predicted_class, confidence, latency_ms, model_version)
model_registry(id, version, path, accuracy, promoted_at, is_champion)
```

`uploads` makes the rubric's "Data file Uploading + Saving to Database" literally true.
`predictions` backs the monitoring charts. `model_registry` holds the champion pointer
read at startup.

## 5. API surface

FastAPI on `:8000`, exposed at `/api/*` through nginx. Swagger at `/docs`.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Liveness without touching the model; used by nginx and the HF healthcheck |
| `GET` | `/api/status` | Uptime, champion version and accuracy, pending counts per class, `retrain_ready`, predictions served, mean latency |
| `POST` | `/api/predict` | Multipart image → class, confidence, full probability vector, latency, model version |
| `POST` | `/api/upload` | Multipart ZIP → `batch_id`, per-class accepted, rejected with reasons, total pending, threshold, `retrain_ready` |
| `POST` | `/api/retrain` | Body `{epochs?, replay_n?, force?}` → `{job_id, status}` |
| `GET` | `/api/retrain/{job_id}` | `{status, log[], metrics, promoted}` |
| `GET` | `/api/retrain/history` | Past runs for the history table and accuracy-over-time chart |
| `GET` | `/api/insights` | Precomputed dataset visualization payload |
| `GET` | `/api/metrics` | Champion's accuracy, loss, per-class P/R/F1, confusion matrix on the test set |

### Retrain triggering

The assignment asks for two distinct things: an automatic trigger "when the need arises",
and a button a user presses. Both are implemented and both are visible in the UI.

- **Automatic:** `/api/status` compares `pending_count` against `RETRAIN_THRESHOLD` (50)
  and returns `retrain_ready`. The UI renders the retrain button as locked until it flips.
- **Manual override:** `/api/retrain` accepts `force: true`, which bypasses the threshold.
  This exists so a live demo cannot get stuck below the threshold.

### `/api/metrics` and production evaluation

This endpoint evaluates the *deployed* champion against the test set and returns the same
four metric families the notebook reports. It is what satisfies "demonstrate the
evaluation process of the model in production" — the notebook's confusion matrix and the
live one can be shown side by side.

### Error handling

| Code | Condition |
|---|---|
| `400` | Malformed ZIP, no recognized class folders, zero valid images |
| `409` | Retrain already in progress |
| `413` | Upload exceeds `MAX_UPLOAD_BYTES` |
| `415` | Not a ZIP, or undecodable image on `/predict` |
| `422` | Pending below threshold and `force` not set |
| `503` | Model failed to load at startup |

Every code has a corresponding human-readable Streamlit message; raw tracebacks are never
surfaced to the UI.

The `409` is not cosmetic. Without it, two clicks of the retrain button start two
TensorFlow training sessions inside one process on a 2-vCPU box, which will exhaust memory
and take the Space down.

## 6. Retrain pipeline

```
UPLOAD
  zip ─► validate ─► SHA-256 dedupe ─┬─ rejected: reason recorded per file
                                     └─ accepted ─► data/pending/<class>/
                                                    INSERT uploads

TRIGGER
  /api/status: pending 51 ≥ 50 ─► retrain_ready ─► button unlocks
                                                   (or force:true)

JOB  (threadpool, lock held)
  1  preprocess pending      class_names=CLASS_NAMES
  2  + replay 400            stratified from data/train/
  3  concatenate, shuffle, batch 32
  4  load models/coffee_model.keras     ← champion is the pretrained start
  5  BatchNorm frozen, Adam 1e-5, 3 epochs, progress callback
  6  evaluate on fixed EVAL_SLICE_SIZE stratified test slice
  7  GATE: candidate_accuracy > champion_accuracy ?

     PROMOTE                            REJECT
       save coffee_model_<stamp>.keras    discard candidate
       INSERT model_registry              champion untouched
       pending ─► data/train/<class>/     pending retained
       hot-swap live model reference
       push to HF Model repo (if token)
                    │
                    └─► UPDATE retrain_runs (metrics, log, promoted)
```

### Design rationale

**The champion is the starting point.** Step 4 loads the previously trained `.keras` file
rather than building a fresh model. This is what the rubric means by "uses a custom model
created as a pre-trained model."

**Replay prevents forgetting.** Fine-tuning on uploaded images alone pulls the model
toward whichever classes appear in the batch. Mixing in 400 stratified samples from the
original training set holds the other classes in place.

**The gate makes retraining safe.** A candidate is only promoted if it beats the current
champion on the test set. A bad upload cannot degrade the deployed model.

**Promoted data joins the replay pool.** On promotion, staged images move into
`data/train/<class>/` so subsequent retrains replay against them. This is what makes the
system a pipeline rather than a one-shot script.

**Rejection is non-destructive.** A failed gate leaves both champion and pending images
untouched, so more data can be uploaded and the run retried.

**The job function must be `def`, not `async def`.** Starlette dispatches synchronous
background tasks to a threadpool. An `async def` performing TensorFlow work would block
the event loop and freeze every concurrent request — including the status polling that
draws the progress log.

**Hot-swap under lock.** On promotion the in-memory model reference is replaced
atomically, so `/api/predict` serves the new model immediately without a restart.

**HF persistence.** Spaces have an ephemeral filesystem; uploads and retrained models are
lost on restart or rebuild. If `HF_TOKEN` is present, promoted models are pushed to a HF
Model repo, which restores the retrained champion after a restart. Absent the token this
step is skipped and the service still works.

### Known limitation

Retraining runs inside the API process, so prediction latency degrades while a job is in
flight. The fix is a separate worker process; it is out of scope here. This is stated
explicitly in the README. It does not affect the load-test results, which are collected
with no retrain active.

## 7. UI design

Four Streamlit pages, each mapped to a required capability.

| Page | Contents |
|---|---|
| **Predict** | Single-image upload, predicted class, confidence, probability bars for all four classes, image preview |
| **Insights** | The four dataset visualizations with written interpretations |
| **Data & Retrain** | ZIP upload, per-file staging results, pending-vs-threshold gauge, retrain button, live polled log, run history table, accuracy-over-time chart |
| **Monitoring** | Uptime, champion version, requests served, mean and p95 latency, live prediction class distribution, confusion matrix from `/api/metrics` |

### Visualizations

Computed once by `scripts/build_insights.py` into a JSON payload served by
`/api/insights`. Recomputing per page load on a 2-vCPU Space would be unusably slow.

1. **Class balance.** Dead even at 1,600 per class. Interpretation: plain accuracy is an
   honest headline metric and no resampling or class weighting is needed.
2. **Mean RGB intensity per class.** Defect runs darker across all three channels.
   Interpretation: black and insect-damaged beans drag the average down, so colour carries
   real discriminative signal.
3. **Mean image per class.** Longberry's average is stretched vertically, peaberry's is
   compact, premium sits between, defect smears out. Interpretation: shape carries signal,
   and defect's visual variance is directly visible.
4. **Bean area distribution.** Threshold the bean against the white background and plot
   foreground-pixel ratio per class. Interpretation: quantifies the silhouette differences
   that visualization 3 shows only qualitatively.

The narrative that ties these together: **the EDA predicted the model's weakness before
training ran.** The smeared defect mean image indicated defect would be hardest to
classify, and defect came back with the lowest recall (0.767) in the confusion matrix.
Closing that loop explicitly is the substance of the "what story does it tell?"
requirement.

## 8. Docker and deployment

### Images

- `Dockerfile` — API only. This is the image scaled during the load test.
- `docker/Dockerfile.space` — all-in-one for HF: nginx, FastAPI, and Streamlit under one
  process manager on port 7860, since a Space exposes exactly one port.

Both are based on `python:3.11-slim` with `tensorflow-cpu` (~1.5GB). The GPU build would
add over 1.5GB for no benefit, since Spaces run CPU-only.

TensorFlow version must be pinned to 2.20 to match the version that wrote the `.keras`
file. Older TensorFlow cannot load it.

### Local compose topology

```
docker compose up --scale api=4

  locust ──► nginx :80 ──┬──► api_1 :8000
                         ├──► api_2 :8000
                         ├──► api_3 :8000
                         └──► api_4 :8000
             ui :8501 ───────┘
```

nginx resolves upstream hostnames once at startup. With a static
`proxy_pass http://api:8000`, every request would reach the same replica regardless of
scale, and all three load-test runs would return identical numbers. The config must use
Docker's embedded DNS with a variable upstream so it re-resolves:

```nginx
resolver 127.0.0.11 valid=10s;
set $upstream http://api:8000;
proxy_pass $upstream;
```

### Space routing

`/` → Streamlit, `/api/*` → FastAPI, `/docs` → Swagger. Single public URL.

## 9. Load-test methodology

`locust/locustfile.py` defines an `HttpUser` with a task mix of roughly 90%
`POST /api/predict` (random image drawn from the test set, multipart) and 10%
`GET /api/status`.

**Protocol:** identical load profile across all runs — 50 users, spawn rate 5/s, 2-minute
duration, headless with `--csv`. Runs at 1, 2, and 4 API containers.

**CPU pinning is required for the experiment to mean anything.** TensorFlow by default
claims every available core, so a single container saturates the host and additional
replicas produce no measurable gain — yielding a flat results table that appears to show
work but demonstrates nothing. Each replica is therefore constrained in compose:

```yaml
cpus: "1.0"
environment:
  TF_NUM_INTRAOP_THREADS: "1"
  TF_NUM_INTEROP_THREADS: "1"
```

With this in place, replica count is a genuine independent variable. The README states
this explicitly.

**Recorded per run:** requests/sec, median, p95, p99, mean latency, failure rate.

| Containers | RPS | Median | p95 | p99 | Failures |
|---|---|---|---|---|---|
| 1 | | | | | |
| 2 | | | | | |
| 4 | | | | | |

One additional low-concurrency run against the live HF URL provides a production latency
data point. It is kept modest because free Spaces run on shared 2-vCPU hardware.

**Expected findings to report rather than hide:**

- Throughput flattens once replica count exceeds the host's physical cores. Host specs are
  recorded in the README and this is reported as a finding.
- p99 substantially exceeds median due to TensorFlow's first-inference warmup. A warmup
  inference at container startup mitigates this and is included.

## 10. README contents

- Project description and problem framing
- Public Space URL
- YouTube demo link
- Repository structure
- Local setup (venv, requirements, dataset download)
- Docker setup (compose, scaling)
- Cloud deployment steps (HF Space)
- Notebook evaluation results: 0.865 test accuracy plus the per-class precision/recall/F1
  table
- Data insights with the four interpretations
- Flood-test results table, host specifications, and CPU-pinning note
- Known limitations

## 11. Video demo requirements

Approximately six minutes, **camera on** — this is an explicit rubric line and camera-off
caps that criterion.

1. Problem, dataset, motivation
2. Notebook walkthrough: preprocessing, optimization techniques (transfer learning, early
   stopping, LR scheduling, two-phase fine-tuning), all four evaluation metrics,
   confusion matrix
3. Live URL, Insights page, walking the interpretations
4. Predict a single bean, correct class displayed
5. Upload a ZIP, staging table, pending counter crossing the threshold
6. Trigger retraining, live log, gate decision, promotion
7. Predict again, showing the new model version in the response
8. Load-test results

**Demo image selection.** The prediction criterion awards full marks only if the displayed
class is correct. Demo images should be drawn from `longberry` (F1 0.917) or `peaberry`
(F1 0.882). `defect` should be avoided — its 0.767 recall means roughly a one-in-four
chance of a visible misclassification. Verify the exact demo images against the deployed
model before recording.

## 12. Testing strategy

Unit tests with pytest, run locally, not deployed:

- `stage_upload` — valid ZIP, unknown class folder rejected, non-image rejected, duplicate
  SHA-256 deduped, empty archive rejected
- `decode_image_bytes` — JPEG and PNG both decode; corrupt bytes raise cleanly
- `load_dataset` — label ordering matches `CLASS_NAMES` when only a subset of class
  folders is present (the regression test for the label-ordering bug)
- `sample_replay` — returns the requested count, stratified across classes
- Promotion gate — promotes when candidate beats champion, rejects when it does not, and
  leaves the champion untouched on rejection
- API smoke tests via `TestClient` — each endpoint's success path plus the documented
  error codes, with the model stubbed

Full training is not unit-tested; the retrain path is verified end to end by running it
once against a small staged batch during implementation.

## 13. Out of scope

- Separate worker process for retraining
- Authentication or rate limiting
- Model explainability (Grad-CAM and similar)
- Multi-bean detection in a single photograph
- Automatic retraining on a schedule; the trigger is threshold-based plus manual
