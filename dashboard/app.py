"""Step 18 - Streamlit dashboard (worker-friendly version).
Start the API first (uvicorn api.main:app --reload), then:  streamlit run dashboard/app.py
"""
import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
 
GREEN, RED, AMBER = "#3DDC84", "#FF5A5F", "#FFB020"
 
st.set_page_config(page_title="Manufacturing Quality Intelligence", page_icon="🏭", layout="wide")
 
# ---------------- styling: cream background, black text, red/green signals ----------------
st.markdown(f"""
<style>
.stApp {{ background:#0F1923; }}
header[data-testid="stHeader"] {{ background:#0F1923; }}
.main p, .main li, .main label, .main span, .main h1, .main h2, .main h3, .main h4,
.main div[data-testid="stMarkdownContainer"], .main [data-testid="stCaptionContainer"] {{ color:#FFFFFF; }}
 
/* sidebar: same look as the reference photo */
[data-testid="stSidebar"] {{ background:#1B2A38; }}
[data-testid="stSidebar"] * {{ color:#FFFFFF; }}
[data-testid="stSidebar"] .sb-caption {{ color:#9FB0BF; font-size:0.85rem; }}
[data-testid="stSidebar"] hr {{ border-color:#FFFFFF; opacity:0.7; margin:0.2rem 0 0.8rem 0; }}
[data-testid="stSidebar"] [data-baseweb="input"],
[data-testid="stSidebar"] [data-baseweb="base-input"],
[data-testid="stSidebar"] input {{ background:#0D1117 !important; color:#FFFFFF !important; }}
[data-testid="stSidebar"] button {{ background:#0D1117; }}
[data-testid="stSidebar"] [data-baseweb="select"] > div {{ background:#0D1117 !important; }}
 
/* KPI cards */
.kpi {{ background:#1B2A38; border:1px solid #3A5166; border-radius:14px; padding:16px 18px;
        border-left:10px solid var(--c); }}
.kpi .lbl {{ font-size:0.95rem; color:#FFFFFF; }}
.kpi .val {{ font-size:2.1rem; font-weight:800; color:var(--c); line-height:1.2; }}
.kpi .sub {{ font-size:0.85rem; color:#B8C7D4; }}
 
/* status banner */
.banner {{ border-radius:14px; padding:16px 20px; font-size:1.4rem; font-weight:800;
           color:#0D1117 !important; margin:8px 0 14px 0; }}
.stTabs [data-baseweb="tab"] {{ font-weight:700; font-size:1.05rem; color:#FFFFFF; }}
.stButton>button {{ background:#FFFFFF; color:#0D1117; font-weight:700; border-radius:10px; padding:0.6rem 1.4rem; }}
.stButton>button:hover {{ background:{GREEN}; color:#0D1117; }}
</style>
""", unsafe_allow_html=True)
 
st.title("🏭 Manufacturing Quality Intelligence")
 
flash = st.session_state.pop("flash", None)      # result of the last button press
if flash:
    {"success": st.success, "warning": st.warning, "error": st.error}[flash[0]](flash[1])
 
 
# ---------------- helpers ----------------
def _error_text(r):
    """Readable error text even when the server answers with plain text (HTTP 500)."""
    try:
        detail = r.json().get("detail", r.text)
    except ValueError:
        return r.text
    return detail if isinstance(detail, str) else str(detail)
 
 
def call_api(endpoint, **kwargs):
    """POST to the API and return JSON (or None after showing a friendly error)."""
    try:
        r = requests.post(f"{API_URL}{endpoint}", timeout=30, **kwargs)
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach the API. Start it with:  uvicorn api.main:app --reload")
        return None
    except requests.exceptions.Timeout:
        st.error("The API took too long to answer.")
        return None
    if r.status_code != 200:
        st.error(f"API error {r.status_code}: {_error_text(r)}")
        return None
    return r.json()
 
 
def get_api(endpoint):
    try:
        r = requests.get(f"{API_URL}{endpoint}", timeout=30)
    except requests.exceptions.RequestException:
        st.error("Cannot reach the API. Start it with:  uvicorn api.main:app --reload")
        return None
    if r.status_code != 200:
        st.error(f"API error {r.status_code}: {_error_text(r)}")
        return None
    return r.json()
 
 
def act(steps, success, timeout=180):
    """Run one or more API calls in order, remember the result message, then redraw.
    steps   = [(endpoint, payload), ...]   (stops at the first error)
    success = a text, or a function(last_response) -> (kind, text)
    The message is shown after the redraw, so errors can no longer disappear."""
    resp = None
    for endpoint, payload in steps:
        try:
            r = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=timeout)
        except requests.exceptions.RequestException as e:
            st.session_state["flash"] = ("error", f"Cannot reach the API: {e}")
            st.rerun()
        if r.status_code != 200:
            st.session_state["flash"] = ("error", f"API error {r.status_code}: {_error_text(r)}")
            st.rerun()
        try:
            resp = r.json()
        except ValueError:
            resp = None
    st.session_state["flash"] = success(resp) if callable(success) else ("success", success)
    st.rerun()
 
 
def produced_message(r):
    """Turn the answer of /produce into a message for the manager."""
    if not isinstance(r, dict) or "produced" not in r:
        return ("success", "Done.")
    if r["status"] == "PAUSED":
        return ("warning", f"The line was STOPPED by an alert after {r['inspected']} products. "
                           "Read the alert below, fix the cause, write a note and resume.")
    if r["status"] in ("COMPLETED", "ON_HOLD", "RELEASED", "SCRAPPED"):
        return ("success", f"All {r['inspected']} products produced. Batch status: {r['status']}. "
                           "Open the 'Batch Report' tab to read the report.")
    return ("success", f"{r['produced']} products produced ({r['inspected']} so far).")
 
 
def api_online():
    try:
        return requests.get(f"{API_URL}/health", timeout=3).status_code == 200
    except requests.exceptions.RequestException:
        return False
 
 
def kpi(col, label, value, color="#FFFFFF", sub=""):
    col.markdown(
        f'<div class="kpi" style="--c:{color}"><div class="lbl">{label}</div>'
        f'<div class="val">{value}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True)
 
 
def banner(text, good):
    st.markdown(f'<div class="banner" style="background:{GREEN if good else RED}">'
                f'{"✅" if good else "⚠"} {text}</div>', unsafe_allow_html=True)
 
 
def risk_color(level):
    level = str(level).lower()
    return GREEN if "low" in level else RED if "high" in level else AMBER
 
 
# ---------------- sidebar: process inputs shared by all tabs ----------------
FEATURE_ORDER = ["temperature", "pressure", "machine_speed", "vibration", "humidity",
                 "material_thickness", "cycle_time", "tool_wear"]
LIMITS = {  # min, max, step
    "temperature": (0.0, 200.0, 0.5), "pressure": (0.0, 20.0, 0.1),
    "machine_speed": (0.0, 5000.0, 10.0), "vibration": (0.0, 20.0, 0.1),
    "humidity": (0.0, 100.0, 1.0), "material_thickness": (0.0, 20.0, 0.1),
    "cycle_time": (0.0, 200.0, 1.0), "tool_wear": (0.0, 200.0, 1.0)}
DEFAULTS = {"temperature": 72.0, "pressure": 5.2, "machine_speed": 1450.0, "vibration": 2.4,
            "humidity": 55.0, "material_thickness": 4.0, "cycle_time": 48.0, "tool_wear": 35.0}
MANUAL, AVG = "Manual entry", "Batch average"
for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(f"in_{_k}", _v)
 
 
def quiet_get(endpoint):
    """GET without error boxes (used by the sidebar)."""
    try:
        r = requests.get(f"{API_URL}{endpoint}", timeout=5)
        return r.json() if r.status_code == 200 else None
    except (requests.exceptions.RequestException, ValueError):
        return None
 
 
def load_inputs():
    """Copy the readings of the chosen batch / product into the input boxes."""
    batch = st.session_state.get("sel_batch", MANUAL)
    if batch == MANUAL:
        return
    prods = quiet_get(f"/batches/{batch}/products") or []
    if not prods:
        return
    choice = st.session_state.get("sel_product", AVG)
    if choice == AVG:
        vals = {k: sum(p[k] for p in prods) / len(prods) for k in FEATURE_ORDER}
    else:
        pos = int(choice[1:3])
        p = next((p for p in prods if p["position"] == pos), None)
        if p is None:
            return
        vals = {k: p[k] for k in FEATURE_ORDER}
    for k, v in vals.items():
        lo, hi, _ = LIMITS[k]
        st.session_state[f"in_{k}"] = float(min(max(round(v, 2), lo), hi))
    for key in ("process_result", "drift_result", "report"):   # old results no longer match
        st.session_state.pop(key, None)
 
 
def on_batch_change():
    st.session_state["sel_product"] = AVG
    load_inputs()
 
 
def num(label, key, help_text):
    lo, hi, step = LIMITS[key]
    return sb.number_input(label, min_value=lo, max_value=hi, step=step, format="%.2f",
                           key=f"in_{key}", help=help_text)
 
 
def section(title):
    sb.markdown(f"### {title}")
    sb.markdown("---")
 
 
sb = st.sidebar
sb.markdown("## ⚙ Production Parameters")
sb.markdown('<div class="sb-caption">Enter the current machine readings.</div>', unsafe_allow_html=True)
 
section("📦 Select Batch")
batches = quiet_get("/batches") or []
usable = [b["batch_id"] for b in batches
          if b["status"] in ("RUNNING", "PAUSED")
          or b["ok_count"] + b["defective_count"] + b["review_count"] > 0]
options = [MANUAL] + usable
if st.session_state.get("sel_batch") not in options:
    st.session_state["sel_batch"] = MANUAL
sb.selectbox("Batch", options, key="sel_batch", on_change=on_batch_change,
             help="Pick a processed batch to load its readings, or use Manual entry.")
selected_batch = st.session_state["sel_batch"]
if selected_batch != MANUAL:
    _prods = quiet_get(f"/batches/{selected_batch}/products") or []
    labels = [AVG] + [f"P{p['position']:02d} · {p['final_status']}" for p in _prods]
    if st.session_state.get("sel_product") not in labels:
        st.session_state["sel_product"] = AVG
    sb.selectbox("Product", labels, key="sel_product", on_change=load_inputs,
                 help="Batch average = the mean reading of all products in the batch.")
    sb.markdown(f'<div class="sb-caption">Showing: {selected_batch} / '
                f'{st.session_state["sel_product"]}</div>', unsafe_allow_html=True)
 
section("🌡 Operating Conditions")
temperature = num("Temperature", "temperature", "Process temperature in °C.")
pressure = num("Pressure", "pressure", "Line pressure in bar.")
humidity = num("Humidity", "humidity", "Room humidity in %.")
 
section("⚙ Machine Conditions")
machine_speed = num("Machine speed", "machine_speed", "Spindle speed in RPM.")
vibration = num("Vibration", "vibration", "Machine vibration level.")
material_thickness = num("Material thickness", "material_thickness", "Thickness in mm.")
cycle_time = num("Cycle time", "cycle_time", "Seconds per cycle.")
tool_wear = num("Tool wear", "tool_wear", "Tool wear in minutes of use.")
 
process = {
    "temperature": temperature, "pressure": pressure, "machine_speed": machine_speed,
    "vibration": vibration, "humidity": humidity, "material_thickness": material_thickness,
    "cycle_time": cycle_time, "tool_wear": tool_wear,
}
sb.markdown("---")
sb.write("API status:", "🟢 **online**" if api_online() else "🔴 **offline**")
 
tabs = st.tabs(["📊 Overview", "📷 Image Inspection", "⚙ Process Prediction",
                "🔍 Explainability", "⚠ Drift Detection", "🧾 Quality Report",
                "🏭 Live Batch", "📦 Batch History", "📑 Batch Report"])
 
# ---------------- Overview: KPI cards ----------------
with tabs[0]:
    summary_file = REPORTS / "overview_summary.json"
    if summary_file.exists():
        s = json.loads(summary_file.read_text())
        total = max(s["total_products"], 1)
        rate = s["defective_products"] / total * 100
        c1, c2, c3, c4, c5 = st.columns(5)
        kpi(c1, "Total products", f"{s['total_products']:,}", "#FFFFFF", "All inspected")
        kpi(c2, "OK products", f"{s['ok_products']:,}", GREEN, f"{100 - rate:.1f}% of total")
        kpi(c3, "Defective products", f"{s['defective_products']:,}", RED, f"{rate:.1f}% of total")
        kpi(c4, "Defect rate", f"{rate:.1f}%", GREEN if rate < 5 else RED, "Target: under 5%")
        kpi(c5, "Drift alerts", f"{s['drift_alerts']:,}",
            GREEN if s["drift_alerts"] == 0 else RED,
            "Process is stable" if s["drift_alerts"] == 0 else "Check the process")
    else:
        st.warning("Run  python -m src.drift_detection.train_drift  to create the overview summary.")
 
    st.markdown("#### Model performance")
    cols = st.columns(2)
    for col, (name, caption) in zip(cols, [("image_confusion_matrix.png", "Image model - confusion matrix"),
                                           ("xgb_confusion_matrix.png", "Process model - confusion matrix")]):
        if (REPORTS / name).exists():
            col.image(str(REPORTS / name), caption=caption, width=380)
 
# ---------------- Image inspection ----------------
with tabs[1]:
    st.subheader("Upload a product image")
    up = st.file_uploader("Drag a JPG / PNG here or click Browse", type=["jpg", "jpeg", "png"])
    if up is not None:
        left, right = st.columns([1, 1])
        left.image(up, width=320)
        with right:
            if st.button("🔎 Inspect image"):
                with st.spinner("Inspecting..."):
                    res = call_api("/predict-image", files={"file": (up.name, up.getvalue(), up.type)})
                if res:
                    st.session_state["image_label"] = res["label"]
                    st.session_state["image_result"] = res
            res = st.session_state.get("image_result")
            if res:
                banner(f"{res['label']} ({res['confidence']}% confidence)", res["label"] != "DEFECTIVE")
                st.markdown("**Probabilities**")
                for lab, p in res["probabilities"].items():
                    st.progress(int(p), text=f"{lab}: {p}%")
 
# ---------------- Process prediction ----------------
with tabs[2]:
    st.subheader("Defect prediction from process parameters")
    st.caption("Change the values in the left panel, then press the button.")
    if st.button("🔮 Predict defect risk"):
        with st.spinner("Predicting..."):
            res = call_api("/predict-process", json=process)
        if res:
            st.session_state["process_result"] = res
    res = st.session_state.get("process_result")
    if res:
        is_ok = res["label"].upper() not in ("DEFECTIVE", "DEFECT")
        banner(f"Prediction: {res['label']}", is_ok)
        c1, c2, c3 = st.columns(3)
        kpi(c1, "Prediction", res["label"], GREEN if is_ok else RED)
        kpi(c2, "Defect probability", f"{res['defect_probability']}%",
            GREEN if res["defect_probability"] < 30 else RED)
        kpi(c3, "Risk level", res["risk_level"], risk_color(res["risk_level"]))
 
# ---------------- Explainability ----------------
with tabs[3]:
    st.subheader("Why this prediction?")
    res = st.session_state.get("process_result")
    if not res:
        st.info("Run a prediction in the 'Process Prediction' tab first.")
    else:
        st.caption("🔺 red = pushes toward a defect   🔻 green = pushes toward OK")
        df = pd.DataFrame(res["top_factors"])
        for _, r in df.iterrows():
            bad = r["shap_value"] > 0
            icon, color = ("🔺", RED) if bad else ("🔻", GREEN)
            st.markdown(
                f"{icon} **{r['feature']}** = {r['value']} → "
                f"<span style='color:{color};font-weight:700'>{r['effect']}</span> "
                f"(SHAP {r['shap_value']:+})", unsafe_allow_html=True)
        st.bar_chart(df.set_index("feature")["shap_value"], color=RED)
    if (REPORTS / "shap_summary.png").exists():
        st.image(str(REPORTS / "shap_summary.png"), caption="Global SHAP summary", width=600)
 
# ---------------- Drift detection ----------------
with tabs[4]:
    st.subheader("Is the process behaving normally?")
    if st.button("📡 Check for drift"):
        with st.spinner("Checking..."):
            res = call_api("/detect-drift", json=process)
        if res:
            st.session_state["drift_result"] = res
    res = st.session_state.get("drift_result")
    if res:
        banner(res["status"], not res["is_drift"])
        c1, c2 = st.columns(2)
        kpi(c1, "Anomaly score", res["anomaly_score"], RED if res["is_drift"] else GREEN,
            "Negative = unusual")
        kpi(c2, "Process state", "DRIFT" if res["is_drift"] else "NORMAL",
            RED if res["is_drift"] else GREEN)
 
    if selected_batch != MANUAL:
        st.markdown(f"#### Drift across the whole batch {selected_batch}")
        prods = get_api(f"/batches/{selected_batch}/products")
        if prods:
            dfd = pd.DataFrame(prods)
            n_drift = int(dfd["drift"].sum())
            c1, c2, c3 = st.columns(3)
            kpi(c1, "Products checked", len(dfd))
            kpi(c2, "Unusual readings", n_drift, RED if n_drift else GREEN)
            kpi(c3, "Lowest anomaly score", round(float(dfd["anomaly_score"].min()), 3),
                RED if n_drift else GREEN, "Negative = unusual")
            st.caption("Anomaly score by product position (below 0 = unusual reading)")
            st.line_chart(dfd.set_index("position")["anomaly_score"], color=RED)
            flagged = dfd[dfd["drift"] == 1][["product_id", "anomaly_score", "temperature",
                                              "pressure", "machine_speed", "vibration",
                                              "humidity", "material_thickness", "cycle_time",
                                              "tool_wear"]]
            if len(flagged):
                st.markdown("**Products with unusual readings**")
                st.dataframe(flagged, width="stretch", hide_index=True)
            else:
                st.success("No unusual readings in this batch.")
    else:
        st.caption("Select a batch in the left panel to see drift across all its products.")
 
# ---------------- Quality report ----------------
with tabs[5]:
    st.subheader("Combined quality intelligence")
    img_label = st.session_state.get("image_label")
    st.markdown(f"**Image result used:** {img_label or 'none yet (upload an image in the Image tab)'}")
    if st.button("🧾 Generate quality report"):
        with st.spinner("Building report..."):
            rep = call_api("/quality-report", json={"process": process, "image_label": img_label})
        if rep:
            st.session_state["report"] = rep
    rep = st.session_state.get("report")
    if rep:
        status = str(rep["overall_status"])
        banner(status, not any(w in status.upper() for w in ("DEFECT", "RISK", "ALERT", "DRIFT", "FAIL")))
        c1, c2, c3 = st.columns(3)
        img_res = str(rep["image_result"])
        kpi(c1, "Image result", img_res, RED if "DEFECT" in img_res.upper() else GREEN)
        p = rep["process_result"]
        kpi(c2, "Process result", p["label"], RED if p["label"].upper().startswith("DEFECT") else GREEN,
            f"{p['defect_probability']}% defect probability")
        ps = rep["process_status"]["status"]
        kpi(c3, "Process status", ps, RED if "drift" in ps.lower() or "unusual" in ps.lower() else GREEN)
        st.markdown("**Major factors**")
        for f in rep["major_factors"]:
            st.markdown(f"• **{f['feature']}** = {f['value']} → {f['effect']}")
 
 
# ---------------- Live batch monitor ----------------
with tabs[6]:
    st.subheader("Live batch monitor")
    user = st.text_input("Your name (saved in the history)", value="Shift manager",
                         key="user_name").strip() or "unknown"
    if st.button("🔄 Refresh", key="refresh_live"):
        st.rerun()
    cur = get_api("/batches/current")
    if cur is None:
        st.warning("Could not read the current batch (see the error above).")
    elif not cur:
        st.info("No batch is open on this line. Starting a batch picks 50 random products "
                "from the production data and inspects them one by one.")
        if st.button("▶ Start a new batch (produce 50 products)", key="start_batch"):
            with st.spinner("Producing and inspecting 50 products..."):
                act([("/batches/start-and-produce", {"user": user})], produced_message)
    else:
        bid, n = cur["batch_id"], cur["counts"]
        running = cur["status"] == "RUNNING"
        banner(f"Batch {bid}  |  {cur['line_id']}  |  {cur['status']}", running)
        c1, c2, c3, c4, c5 = st.columns(5)
        kpi(c1, "Inspected", f"{n['n']} / {cur['planned_size']}")
        kpi(c2, "OK", n["ok"], GREEN)
        kpi(c3, "Defective", n["defective"], RED if n["defective"] else GREEN)
        kpi(c4, "To review", n["review"], AMBER if n["review"] else GREEN)
        kpi(c5, "Unusual readings", n["drift"], RED if n["drift"] else GREEN)
 
        st.markdown("#### Open alerts")
        alerts = get_api("/alerts") or []
        if not alerts:
            st.success("No open alerts.")
        for a in alerts:
            (st.error if a["level"] == "STOP" else st.warning)(f"**{a['level']}**  {a['message']}")
            if st.button("Acknowledge", key=f"ack{a['alert_id']}"):
                act([(f"/alerts/{a['alert_id']}/ack", {"user": user})], "Alert acknowledged.")
 
        st.markdown("#### Actions")
        note = st.text_input("Note (required to resume)", key="live_note")
        b1, b2, _ = st.columns(3)
        if running:
            if n["n"] < cur["planned_size"] and b1.button("⏩ Continue production", key="go_on"):
                with st.spinner("Producing..."):
                    act([(f"/batches/{bid}/produce", {})], produced_message)
            if b1.button("⛔ Stop the line", key="stop_line"):
                act([(f"/batches/{bid}/stop", {"user": user, "note": note})],
                    ("warning", "Line stopped. The batch is PAUSED."))
        elif cur["status"] == "PAUSED":
            if b1.button("▶ Resume and continue production", key="resume_line"):
                if not note.strip():
                    st.warning("Write a note first: what was checked or fixed?")
                else:
                    with st.spinner("Producing..."):
                        act([(f"/batches/{bid}/resume", {"user": user, "note": note}),
                             (f"/batches/{bid}/produce", {})], produced_message)
        if cur["status"] in ("RUNNING", "PAUSED"):
            if b2.button("🏁 Close batch now", key="close_batch"):
                act([(f"/batches/{bid}/close", {"user": user})],
                    "Batch closed. Open the 'Batch Report' tab to read the report.")
        if cur["status"] == "ON_HOLD":
            st.error("This batch is ON HOLD. Open the 'Batch Report' tab to release or scrap it. "
                     "No new batch can start until then.")
 
# ---------------- Batch history ----------------
with tabs[7]:
    st.subheader("All batches")
    hist = get_api("/batches")
    if hist:
        dfh = pd.DataFrame(hist)[["batch_id", "line_id", "started_at", "status", "ok_count",
                                  "defective_count", "review_count", "drift_count", "decision"]]
        st.dataframe(dfh, width="stretch", hide_index=True)
        st.download_button("⬇ Download history (CSV)", dfh.to_csv(index=False),
                           "batch_history.csv", "text/csv")
    else:
        st.info("No batches yet. Start one in the 'Live Batch' tab.")
 
# ---------------- Batch report ----------------
with tabs[8]:
    st.subheader("Batch report")
    hist = get_api("/batches") or []
    finished = [b for b in hist if b["status"] not in ("RUNNING", "PAUSED")]
    if not finished:
        st.info("No finished batches yet. Close a batch in the 'Live Batch' tab first.")
    else:
        bid = st.selectbox("Choose a batch", [b["batch_id"] for b in finished])
        rep = get_api(f"/batches/{bid}/report")
        if rep:
            status = next(b["status"] for b in finished if b["batch_id"] == bid)
            banner(f"{rep['batch_id']}: {status}", status in ("COMPLETED", "RELEASED"))
            c1, c2, c3, c4 = st.columns(4)
            kpi(c1, "Products", rep["total"])
            kpi(c2, "OK", rep["ok"], GREEN)
            kpi(c3, "Defective", rep["defective"], RED if rep["defective"] else GREEN,
                f"{rep['defect_rate']}% of batch")
            kpi(c4, "To review", rep["review"], AMBER if rep["review"] else GREEN)
            st.markdown(f"**Recommendation:** {rep['recommendation']}")
            st.markdown("#### Defective products and why")
            if not rep["defective_products"]:
                st.success("No defective products in this batch.")
            for p in rep["defective_products"]:
                st.markdown(f"**{p['product_id']}**")
                for reason in p["reasons"]:
                    st.write("•", reason)
            if rep["review_products"]:
                with st.expander(f"{len(rep['review_products'])} products to review"):
                    for p in rep["review_products"]:
                        st.markdown(f"**{p['product_id']}**: " + " ".join(p["reasons"]))
            if rep["process_averages"]:
                st.markdown("#### Process averages for this batch")
                st.dataframe(pd.DataFrame([rep["process_averages"]]), hide_index=True)
            prods = get_api(f"/batches/{bid}/products")
            if prods:
                dfp = pd.DataFrame(prods).drop(columns=["reasons", "top_factors"])
                st.download_button("⬇ Download all products (CSV)", dfp.to_csv(index=False),
                                   f"{bid}_products.csv", "text/csv")
            if status == "ON_HOLD":
                st.markdown("#### Manager decision")
                who = st.text_input("Your name", value="Shift manager", key="decide_user")
                why = st.text_input("Reason (required)", key="decide_note")
                d1, d2 = st.columns(2)
                for col, label, decision, msg in [(d1, "✅ Release batch", "RELEASE", "Batch released."),
                                                  (d2, "🗑 Scrap batch", "SCRAP", "Batch scrapped.")]:
                    if col.button(label, key=f"dec_{decision}"):
                        if not why.strip():
                            st.warning("Write the reason first.")
                        else:
                            act([(f"/batches/{bid}/decision",
                                  {"user": who.strip() or "unknown", "note": why,
                                   "decision": decision})], msg)
