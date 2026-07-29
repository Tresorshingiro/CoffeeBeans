"""Streamlit UI. A pure HTTP client of the API — no model code lives here."""
import os
import time

import pandas as pd
import requests
import streamlit as st

API_BASE = os.environ.get("API_BASE", "http://localhost:8000")

st.set_page_config(page_title="Coffee Bean Grading", page_icon="☕",
                   layout="wide")


def api_get(path, timeout=120, **kwargs):
    return requests.get(f"{API_BASE}{path}", timeout=timeout, **kwargs)


def api_post(path, **kwargs):
    return requests.post(f"{API_BASE}{path}", timeout=600, **kwargs)


def show_error(response):
    try:
        detail = response.json().get("detail", response.text)
    except Exception:
        detail = response.text
    st.error(f"{response.status_code}: {detail}")


st.sidebar.title("☕ Coffee Bean Grading")
page = st.sidebar.radio(
    "Page", ["Predict", "Insights", "Data & Retrain", "Monitoring"])

try:
    status = api_get("/api/status").json()
    st.sidebar.success(f"API up · {status['uptime_seconds']:.0f}s")
    st.sidebar.caption(f"Model: {status['model_version']}")
except Exception:
    status = None
    st.sidebar.error("API unreachable")


# ── Predict
if page == "Predict":
    st.title("Predict a bean grade")
    st.caption("Upload a single close-up photo of one green coffee bean on a "
               "plain background.")

    uploaded = st.file_uploader("Bean image", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        left, right = st.columns([1, 1])
        with left:
            st.image(uploaded, caption=uploaded.name, use_container_width=True)
        if st.button("Predict", type="primary"):
            response = api_post(
                "/api/predict",
                files={"file": (uploaded.name, uploaded.getvalue(),
                                uploaded.type)})
            if response.status_code != 200:
                show_error(response)
            else:
                body = response.json()
                with right:
                    st.metric("Predicted grade", body["class"],
                              f"{body['confidence'] * 100:.1f}% confidence")
                    st.caption(f"Served by {body['model_version']} in "
                               f"{body['latency_ms']:.0f} ms")
                    st.bar_chart(pd.Series(body["probabilities"],
                                           name="probability"))


# ── Insights 
elif page == "Insights":
    st.title("What the data says")
    response = api_get("/api/insights")
    if response.status_code != 200:
        show_error(response)
        st.info("Run `python scripts/build_insights.py` to generate these.")
    else:
        data = response.json()
        notes = data["interpretations"]

        st.subheader("1. Class balance")
        st.bar_chart(pd.Series(data["class_counts"], name="training images"))
        st.info(notes["class_balance"])

        st.subheader("2. Bean colour — averaged over bean pixels only")
        channels = pd.DataFrame(data["channel_means"]).T
        st.bar_chart(channels[["r", "g", "b"]].rename(
            columns={"r": "Red", "g": "Green", "b": "Blue"}))
        st.caption("Overall brightness per class")
        st.bar_chart(channels[["brightness"]])
        st.info(notes["channel_means"])

        st.subheader("3. Bean area — share of the frame the bean fills")
        areas = pd.DataFrame(data["area_ratios"]).T
        st.bar_chart(areas[["mean"]])
        st.dataframe(areas, use_container_width=True)
        st.info(notes["area_ratios"])

        st.subheader("The story these tell together")
        st.success(notes["story"])


# ── Data & Retrain 
elif page == "Data & Retrain":
    st.title("Upload data and retrain")

    st.subheader("1. Upload new labelled beans")
    st.caption("A .zip containing folders named defect, longberry, peaberry, "
               "or premium.")
    archive = st.file_uploader("ZIP archive", type=["zip"])
    if archive is not None and st.button("Upload", type="primary"):
        response = api_post(
            "/api/upload",
            files={"file": (archive.name, archive.getvalue(),
                            "application/zip")})
        if response.status_code != 200:
            show_error(response)
        else:
            body = response.json()
            st.success(f"Staged {body['total_accepted']} images")
            st.dataframe(
                pd.DataFrame(
                    [{"class": k, "accepted": v}
                     for k, v in body["accepted"].items()]),
                use_container_width=True)
            if body["rejected"]:
                with st.expander(f"{len(body['rejected'])} files rejected"):
                    st.dataframe(pd.DataFrame(body["rejected"]),
                                 use_container_width=True)

    st.divider()
    st.subheader("2. Retrain")
    if status:
        pending = status["pending_total"]
        threshold = status["retrain_threshold"]
        st.progress(min(pending / threshold, 1.0),
                    text=f"{pending} of {threshold} images staged")
        if status["pending_counts"]:
            st.dataframe(
                pd.DataFrame([{"class": k, "pending": v}
                              for k, v in status["pending_counts"].items()]),
                use_container_width=True)

        ready = status["retrain_ready"]
        if ready:
            st.success("Threshold reached — retraining is unlocked.")
        else:
            st.warning(f"{threshold - pending} more images needed to unlock "
                       "retraining automatically.")
        force = st.checkbox("Force retrain below threshold", value=not ready)

        if st.button("Start retraining", type="primary",
                     disabled=pending == 0):
            response = api_post("/api/retrain", json={"force": force})
            if response.status_code != 202:
                show_error(response)
            else:
                job_id = response.json()["job_id"]
                st.session_state["job_id"] = job_id

    job_id = st.session_state.get("job_id")
    if job_id:
        st.caption(f"Job {job_id}")
        log_box = st.empty()
        status_box = st.empty()
        job = None
        for _ in range(400):
            job = api_get(f"/api/retrain/{job_id}").json()
            log_box.code("\n".join(job.get("log", [])) or "starting…")
            if job["status"] != "running":
                break
            status_box.info("Retraining in progress…")
            time.sleep(2)

        if job and job["status"] == "completed":
            if job.get("promoted"):
                status_box.success(
                    f"Promoted — new model beat the champion "
                    f"({job['candidate_accuracy']:.4f} vs "
                    f"{job['champion_accuracy']:.4f})")
            else:
                status_box.warning(
                    f"Rejected — candidate did not beat the champion "
                    f"({job['candidate_accuracy']:.4f} vs "
                    f"{job['champion_accuracy']:.4f}). Champion kept.")
        elif job and job["status"] == "failed":
            status_box.error(f"Retraining failed: {job.get('error')}")

    st.divider()
    st.subheader("3. Retraining history")
    history = api_get("/api/retrain/history")
    if history.status_code == 200:
        runs = history.json()["runs"]
        if runs:
            frame = pd.DataFrame(runs)[
                ["id", "started_at", "status", "n_pending", "n_replay",
                 "candidate_accuracy", "champion_accuracy", "promoted"]]
            st.dataframe(frame, use_container_width=True)
            scored = frame.dropna(subset=["candidate_accuracy"])
            if not scored.empty:
                st.line_chart(
                    scored.set_index("id")[["candidate_accuracy",
                                            "champion_accuracy"]])
        else:
            st.caption("No retraining runs yet.")


# ── Monitoring 
elif page == "Monitoring":
    st.title("Service and model monitoring")
    if not status:
        st.error("API unreachable.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Uptime", f"{status['uptime_seconds'] / 60:.1f} min")
        c2.metric("Model", status["model_version"])
        c3.metric("Predictions served", status["predictions_served"])
        c4.metric("Mean latency", f"{status['mean_latency_ms']:.0f} ms")

        if status["class_counts"]:
            st.subheader("What the model has been predicting")
            st.bar_chart(pd.Series(status["class_counts"], name="predictions"))

        st.subheader("Evaluation in production")
        st.caption("The deployed model scored against the held-out test set.")
        if st.button("Run evaluation"):
            # Scores the whole test set on one pinned CPU: ~104s for the 1,600
            # images a local bind-mounted data/full exposes, so the default
            # 120s read timeout leaves no headroom once the sidebar's status
            # polling competes for the same worker. nginx already allows 900s.
            with st.spinner("Evaluating the deployed model on the full test "
                            "set — this takes a minute or two…"):
                response = api_get("/api/metrics", timeout=600)
            if response.status_code != 200:
                show_error(response)
            else:
                body = response.json()
                a, b, c = st.columns(3)
                a.metric("Accuracy", f"{body['accuracy']:.3f}")
                b.metric("Loss", f"{body['loss']:.3f}")
                c.metric("Samples", body["n_samples"])
                st.dataframe(pd.DataFrame(body["per_class"]).T,
                             use_container_width=True)
                st.subheader("Confusion matrix")
                st.dataframe(
                    pd.DataFrame(body["confusion_matrix"],
                                 index=[f"actual {c}" for c in status["classes"]],
                                 columns=[f"pred {c}" for c in status["classes"]]),
                    use_container_width=True)
