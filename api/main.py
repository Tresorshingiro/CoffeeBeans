"""FastAPI service. All model work happens here; the UI is a pure client."""
import json
import threading
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src import config, database, model as model_module, prediction, preprocessing

START_TIME = time.time()

_JOBS: dict = {}
_retrain_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    database.init_db()
    try:
        prediction.load_champion()
        prediction.warmup()
    except Exception as exc:  # noqa: BLE001 — service must start to report this
        print(f"WARNING: could not load champion model: {exc}")
    yield


app = FastAPI(
    title="Coffee Bean Grading API",
    description="Grades green Arabica coffee beans into four classes.",
    version="1.0.0",
    lifespan=lifespan,
)


class RetrainRequest(BaseModel):
    epochs: int | None = None
    replay_n: int | None = None
    force: bool = False


@app.get("/api/health")
def health():
    """Liveness only. Deliberately does not touch the model."""
    return {"status": "ok"}


@app.get("/api/status")
def status():
    counts = preprocessing.pending_counts()
    pending_total = sum(counts.values())
    stats = database.prediction_stats()
    champion = database.get_champion()
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "model_loaded": prediction.is_ready(),
        "model_version": prediction.get_version(),
        "model_accuracy": champion["accuracy"] if champion else None,
        "classes": config.CLASS_NAMES,
        "pending_counts": counts,
        "pending_total": pending_total,
        "retrain_threshold": config.RETRAIN_THRESHOLD,
        "retrain_ready": pending_total >= config.RETRAIN_THRESHOLD,
        "predictions_served": stats["total"],
        "mean_latency_ms": stats["mean_latency_ms"],
        "p95_latency_ms": stats["p95_latency_ms"],
        "class_counts": stats["class_counts"],
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if not prediction.is_ready():
        raise HTTPException(503, "Model is not loaded")
    payload = await file.read()
    try:
        return prediction.predict_image(payload)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(415, "Could not decode image") from exc


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    payload = await file.read()
    if len(payload) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Upload exceeds the maximum size")
    try:
        result = preprocessing.stage_upload(payload, str(uuid.uuid4())[:8])
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    result["retrain_threshold"] = config.RETRAIN_THRESHOLD
    result["retrain_ready"] = result["pending_total"] >= config.RETRAIN_THRESHOLD
    return result


def _run_retrain_job(run_id: int, epochs, replay_n) -> None:
    """Synchronous by design.

    Starlette dispatches sync background tasks to a threadpool. An async def
    here would block the event loop and freeze the status polling that draws
    the progress log.
    """
    job = _JOBS[run_id]
    try:
        result = model_module.retrain(
            progress_cb=job["log"].append, epochs=epochs, replay_n=replay_n)
        job.update(status="completed", promoted=result["promoted"],
                   metrics=result["metrics"],
                   candidate_accuracy=result["candidate_accuracy"],
                   champion_accuracy=result["champion_accuracy"])
        database.finish_retrain_run(
            run_id, "completed", result["candidate_accuracy"],
            result["champion_accuracy"], result["promoted"],
            result["model_path"], "\n".join(job["log"]))
    except Exception as exc:  # noqa: BLE001
        job.update(status="failed", error=str(exc))
        job["log"].append(f"FAILED: {exc}")
        database.finish_retrain_run(
            run_id, "failed", None, None, False, None, "\n".join(job["log"]))
    finally:
        if _retrain_lock.locked():
            _retrain_lock.release()


@app.post("/api/retrain", status_code=202)
def trigger_retrain(request: RetrainRequest, background: BackgroundTasks):
    counts = preprocessing.pending_counts()
    pending_total = sum(counts.values())
    if pending_total == 0:
        raise HTTPException(422, "No pending images to retrain on")
    if pending_total < config.RETRAIN_THRESHOLD and not request.force:
        raise HTTPException(
            422,
            f"Only {pending_total} pending images; threshold is "
            f"{config.RETRAIN_THRESHOLD}. Upload more or use force.")
    if not _retrain_lock.acquire(blocking=False):
        raise HTTPException(409, "A retraining run is already in progress")

    epochs = request.epochs or config.RETRAIN_EPOCHS
    replay_n = request.replay_n or config.REPLAY_SAMPLES
    run_id = database.create_retrain_run(pending_total, replay_n, epochs)
    _JOBS[run_id] = {"status": "running", "log": [], "promoted": None,
                     "metrics": None, "error": None}
    background.add_task(_run_retrain_job, run_id, epochs, replay_n)
    return {"job_id": run_id, "status": "running",
            "n_pending": pending_total, "n_replay": replay_n}


# Declared before /api/retrain/{job_id} so "history" is not parsed as an int.
@app.get("/api/retrain/history")
def retrain_history():
    return {"runs": database.list_retrain_runs()}


@app.get("/api/retrain/{job_id}")
def retrain_status(job_id: int):
    job = _JOBS.get(job_id)
    if job is None:
        record = database.get_retrain_run(job_id)
        if record is None:
            raise HTTPException(404, f"No retraining job with id {job_id}")
        return {"job_id": job_id, "status": record["status"],
                "log": (record["log"] or "").splitlines(),
                "promoted": bool(record["promoted"]),
                "candidate_accuracy": record["candidate_accuracy"],
                "champion_accuracy": record["champion_accuracy"]}
    return {"job_id": job_id, **job}


@app.get("/api/insights")
def insights():
    if not config.INSIGHTS_PATH.exists():
        raise HTTPException(
            404, "Insights not generated. Run scripts/build_insights.py")
    return JSONResponse(json.loads(config.INSIGHTS_PATH.read_text()))


@app.get("/api/metrics")
def metrics():
    """Evaluate the deployed champion against the full test set.

    This is the production-evaluation surface: its numbers should line up
    with the notebook's.
    """
    if not prediction.is_ready():
        raise HTTPException(503, "Model is not loaded")
    ds = preprocessing.load_dataset(config.test_dir(), shuffle=False)
    result = model_module.evaluate(prediction.get_model(), ds)
    result["model_version"] = prediction.get_version()
    return result
