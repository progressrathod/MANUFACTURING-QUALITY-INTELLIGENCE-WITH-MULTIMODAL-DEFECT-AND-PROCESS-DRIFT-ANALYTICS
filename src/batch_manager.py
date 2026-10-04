"""Step 22 - Batch manager: IDs, product inspection, alert rules, reports, history."""
import json
from collections import Counter
from datetime import datetime
from typing import Optional
 
from src.config import (FEATURES, LINE_ID, BATCH_SIZE, WARN_DEFECTS, STOP_DEFECTS,
                        STREAK_WARN, WARN_DRIFT, STOP_DRIFT, SPEC_LIMITS)
from src.database import get_conn, init_db
from src.process_prediction.predict import predict_process
from src.process_prediction.explain_shap import explain
from src.drift_detection.predict import detect_drift
 
BLOCKING = ("RUNNING", "PAUSED", "ON_HOLD")     # a line with such a batch cannot start a new one
 
 
def _now():
    return datetime.now().isoformat(timespec="seconds")
 
 
def _log(c, batch_id, actor, action, note=""):
    c.execute("INSERT INTO events(batch_id, at, actor, action, note) VALUES (?,?,?,?,?)",
              (batch_id, _now(), actor, action, note))
 
 
def _batch(c, batch_id):
    row = c.execute("SELECT * FROM batches WHERE batch_id=?", (batch_id,)).fetchone()
    if row is None:
        raise KeyError(f"Unknown batch {batch_id}")
    return row

def _add_column(c, table, col, ddl):
    cols = [
        r["name"]
        for r in c.execute(f"PRAGMA table_info({table})")
    ]

    if col not in cols:
        c.execute(
            f"ALTER TABLE {table} ADD COLUMN {col} {ddl}"
        )
 
def _counts(c, batch_id):
    r = c.execute("""SELECT COUNT(*) AS n,
                            SUM(final_status='OK') AS ok,
                            SUM(final_status='DEFECTIVE') AS defective,
                            SUM(final_status='REVIEW') AS review,
                            SUM(drift) AS drift
                     FROM products WHERE batch_id=?""", (batch_id,)).fetchone()
    return {k: (r[k] or 0) for k in ("n", "ok", "defective", "review", "drift")}
 
 
# ------------------------------------------------------------------ start / stop / resume
def start_batch(line_id=LINE_ID, size=BATCH_SIZE, operator="system"):
    """Create a new batch and return its ID. Refused while the line has an open batch."""
    init_db()
    with get_conn() as c:
        marks = ",".join("?" * len(BLOCKING))
        blocked = c.execute(f"SELECT batch_id, status FROM batches "
                            f"WHERE line_id=? AND status IN ({marks})",
                            (line_id, *BLOCKING)).fetchone()
        if blocked:
            raise RuntimeError(f"Cannot start a new batch: {blocked['batch_id']} is "
                               f"{blocked['status']}. Resolve it first.")
        day = datetime.now().strftime("%Y%m%d")
        n = c.execute("SELECT COUNT(*) FROM batches WHERE batch_id LIKE ?",
                      (f"B-{day}-%",)).fetchone()[0] + 1
        batch_id = f"B-{day}-{n:03d}"
        c.execute("INSERT INTO batches(batch_id, line_id, started_at, status, planned_size) "
                  "VALUES (?,?,?,?,?)", (batch_id, line_id, _now(), "RUNNING", size))
        _log(c, batch_id, operator, "BATCH_STARTED", f"line {line_id}, size {size}")
    return batch_id
 
 
def stop_batch(batch_id, user, note=""):
    with get_conn() as c:
        b = _batch(c, batch_id)
        if b["status"] != "RUNNING":
            raise RuntimeError(f"Batch {batch_id} is {b['status']}, not RUNNING.")
        c.execute("UPDATE batches SET status='PAUSED' WHERE batch_id=?", (batch_id,))
        _log(c, batch_id, user, "STOPPED_BY_MANAGER", note)
 
 
def resume_batch(batch_id, user, note):
    if not note.strip():
        raise ValueError("A note is required to resume: what was checked or fixed?")
    with get_conn() as c:
        b = _batch(c, batch_id)
        if b["status"] != "PAUSED":
            raise RuntimeError(f"Batch {batch_id} is {b['status']}, not PAUSED.")
        c.execute("UPDATE batches SET status='RUNNING' WHERE batch_id=?", (batch_id,))
        _log(c, batch_id, user, "RESUMED", note)
 
 
# ------------------------------------------------------------------ inspect one product
def inspect_product(batch_id, process_row: dict, image_label: Optional[str] = None,
                    image_confidence: Optional[float] = None, image_file: Optional[str] = None):
    """Give the next product in the batch its ID, judge it, store it, check the alert rules."""
    with get_conn() as c:
        b = _batch(c, batch_id)
        if b["status"] != "RUNNING":
            raise RuntimeError(f"Batch {batch_id} is {b['status']} - cannot inspect products.")
        position = c.execute("SELECT COUNT(*) FROM products WHERE batch_id=?",
                             (batch_id,)).fetchone()[0] + 1
        product_id = f"{batch_id}-P{position:02d}"
 
        proc = predict_process(process_row)
        drift = detect_drift(process_row)
        factors = explain(process_row, top_n=3)
        out_of_spec = [f"{k} = {process_row[k]} is outside the allowed range {lo} to {hi}."
                       for k, (lo, hi) in SPEC_LIMITS.items() if not lo <= process_row[k] <= hi]
 
        # DEFECTIVE only when the image (strong evidence) says so.
        # The weak process signals alone send the product to REVIEW (a person checks it).
        image_bad = image_label == "DEFECTIVE"
        if image_bad:
            status = "DEFECTIVE"
        elif proc["label"] == "DEFECTIVE" or drift["is_drift"] or out_of_spec:
            status = "REVIEW"
        else:
            status = "OK"
 
        reasons = []
        if image_bad:
            reasons.append(f"Visual inspection found a defect ({image_confidence}% confidence).")
        if proc["label"] == "DEFECTIVE":
            reasons.append(f"Process model: {proc['defect_probability']}% defect probability "
                           f"({proc['risk_level']} risk).")
        if drift["is_drift"]:
            reasons.append(f"Unusual machine behaviour (anomaly score {drift['anomaly_score']}).")
        reasons += out_of_spec
        if status != "OK":
            reasons += [f"Main factor: {f['feature']} = {f['value']} {f['effect']}."
                        for f in factors if f["shap_value"] > 0]
 
        c.execute("""INSERT INTO products(product_id, batch_id, position, inspected_at,
                       temperature, pressure, machine_speed, vibration, humidity,
                       material_thickness, cycle_time, tool_wear, image_file, image_label,
                       image_confidence, process_label, defect_probability, risk_level,
                       drift, anomaly_score, final_status, reasons, top_factors)
                     VALUES (?,?,?,?, ?,?,?,?,?,?,?,?, ?,?,?, ?,?,?, ?,?, ?,?,?)""",
                  (product_id, batch_id, position, _now(),
                   *[process_row[k] for k in FEATURES],
                   image_file, image_label, image_confidence,
                   proc["label"], proc["defect_probability"], proc["risk_level"],
                   int(drift["is_drift"]), drift["anomaly_score"],
                   status, json.dumps(reasons), json.dumps(factors)))
 
        _check_rules(c, batch_id, product_id)
        if position >= b["planned_size"]:
            _close(c, batch_id)
    return {"product_id": product_id, "position": position, "final_status": status,
            "reasons": reasons}
 
 
# ------------------------------------------------------------------ alert rules
def _alert(c, batch_id, product_id, code, level, message):
    if c.execute("SELECT 1 FROM alerts WHERE batch_id=? AND code=?", (batch_id, code)).fetchone():
        return                                   # each alert is raised once per batch
    c.execute("INSERT INTO alerts(batch_id, product_id, created_at, level, code, message) "
              "VALUES (?,?,?,?,?,?)", (batch_id, product_id, _now(), level, code, message))
    if level == "STOP":                          # the line is paused until a manager resumes it
        c.execute("UPDATE batches SET status='PAUSED' WHERE batch_id=? AND status='RUNNING'",
                  (batch_id,))
        _log(c, batch_id, "system", "AUTO_PAUSED", message)
 
 
def _check_rules(c, batch_id, product_id):
    s = _counts(c, batch_id)
    last = [r["final_status"] for r in c.execute(
        "SELECT final_status FROM products WHERE batch_id=? ORDER BY position DESC LIMIT ?",
        (batch_id, STREAK_WARN))]
    tag = f"Batch {batch_id}, product {product_id}: "
    if s["defective"] >= STOP_DEFECTS:
        _alert(c, batch_id, product_id, "DEFECT_STOP", "STOP",
               tag + f"{s['defective']} defective products. STOP the line and inspect the machine.")
    elif s["defective"] >= WARN_DEFECTS:
        _alert(c, batch_id, product_id, "DEFECT_WARN", "WARNING",
               tag + f"{s['defective']} defective products so far. Watch the line closely.")
    if len(last) == STREAK_WARN and all(x == "DEFECTIVE" for x in last):
        _alert(c, batch_id, product_id, "STREAK", "WARNING",
               tag + f"{STREAK_WARN} defective products in a row. Check the tool and material.")
    if s["drift"] >= STOP_DRIFT:
        _alert(c, batch_id, product_id, "DRIFT_STOP", "STOP",
               tag + f"{s['drift']} unusual readings. STOP the line and check the machine.")
    elif s["drift"] >= WARN_DRIFT:
        _alert(c, batch_id, product_id, "DRIFT_WARN", "WARNING",
               tag + f"{s['drift']} unusual readings. Machine behaviour is changing.")
 
 
# ------------------------------------------------------------------ close + report
def _close(c, batch_id):
    s = _counts(c, batch_id)
    stops = c.execute("SELECT COUNT(*) FROM alerts WHERE batch_id=? AND level='STOP'",
                      (batch_id,)).fetchone()[0]
    status = "ON_HOLD" if (s["defective"] >= WARN_DEFECTS or stops) else "COMPLETED"
    c.execute("""UPDATE batches SET status=?, ended_at=?, ok_count=?, defective_count=?,
                 review_count=?, drift_count=? WHERE batch_id=?""",
              (status, _now(), s["ok"], s["defective"], s["review"], s["drift"], batch_id))
    _log(c, batch_id, "system", "BATCH_CLOSED", status)
    report = _build_report(c, batch_id)
    c.execute("INSERT INTO batch_reports(batch_id, generated_at, report_json) VALUES (?,?,?)",
              (batch_id, _now(), json.dumps(report)))
    return report
 
 
def close_batch(batch_id, user="system"):
    """Close a batch early (e.g. after a stop) and generate its report."""
    with get_conn() as c:
        b = _batch(c, batch_id)
        if b["status"] not in ("RUNNING", "PAUSED"):
            raise RuntimeError(f"Batch {batch_id} is already {b['status']}.")
        _log(c, batch_id, user, "CLOSED_BY_USER")
        return _close(c, batch_id)
    if defective >= WARN_DEFECTS or had_stop or (REQUIRE_REVIEW_BEFORE_NEXT and review_count > 0):
        status = "ON_HOLD"
 
def _build_report(c, batch_id):
    b, s = dict(_batch(c, batch_id)), _counts(c, batch_id)
    prods = [dict(r) for r in c.execute(
        "SELECT * FROM products WHERE batch_id=? ORDER BY position", (batch_id,))]
    for p in prods:
        p["reasons"] = json.loads(p["reasons"])
        p["top_factors"] = json.loads(p["top_factors"])
    bad = [p for p in prods if p["final_status"] == "DEFECTIVE"]
    review = [p for p in prods if p["final_status"] == "REVIEW"]
    causes = Counter(f["feature"] for p in bad + review
                     for f in p["top_factors"] if f["shap_value"] > 0)
    alerts = [dict(r) for r in c.execute(
        "SELECT * FROM alerts WHERE batch_id=? ORDER BY alert_id", (batch_id,))]
    avg = {k: round(sum(p[k] for p in prods) / len(prods), 2) for k in FEATURES} if prods else {}
    short = lambda lst: [{"product_id": p["product_id"], "position": p["position"],
                          "reasons": p["reasons"]} for p in lst]
    top = causes.most_common(1)[0][0] if causes else None
    if b["status"] == "ON_HOLD":
        rec = (f"HOLD this batch. Remove the {s['defective']} defective units, re-inspect the "
               f"{s['review']} REVIEW units and check the machine"
               + (f" (main suspect: {top})" if top else "")
               + " before the next batch starts.")
    else:
        rec = (f"Batch can be released after removing the {s['defective']} defective units "
               f"and checking the {s['review']} REVIEW units.")
    return {
        "batch_id": batch_id, "line_id": b["line_id"], "started_at": b["started_at"],
        "ended_at": b["ended_at"], "status": b["status"], "planned_size": b["planned_size"],
        "total": s["n"], "ok": s["ok"], "defective": s["defective"], "review": s["review"],
        "drift_alerts": s["drift"],
        "defect_rate": round(s["defective"] / max(s["n"], 1) * 100, 1),
        "process_averages": avg, "top_causes": causes.most_common(3),
        "defective_products": short(bad), "review_products": short(review),
        "alerts": alerts, "recommendation": rec,
    }
# ------------------------------------------------------------------ manager decision
def decide_batch(batch_id, decision, user, note):
    if decision not in ("RELEASE", "SCRAP"):
        raise ValueError("decision must be RELEASE or SCRAP")

    if not note.strip():
        raise ValueError("A note is required: why release or scrap this batch?")

    with get_conn() as c:
        b = _batch(c, batch_id)

        if b["status"] not in ("ON_HOLD", "COMPLETED"):
            raise RuntimeError(
                f"Batch {batch_id} is {b['status']}; it cannot be decided yet."
            )

        c.execute(
            """UPDATE batches
               SET status=?, decision=?, decision_by=?, decision_note=?
               WHERE batch_id=?""",
            (
                "RELEASED" if decision == "RELEASE" else "SCRAPPED",
                decision,
                user,
                note,
                batch_id
            )
        )

        _log(c, batch_id, user, decision, note)
 
 
# ------------------------------------------------------------------ reading data (history)
def list_batches(limit=200):
    with get_conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT * FROM batches ORDER BY started_at DESC, batch_id DESC LIMIT ?", (limit,))]
 
 
def current_batch(line_id=LINE_ID):
    """The batch that needs attention on this line (running, paused or on hold), if any."""
    with get_conn() as c:
        marks = ",".join("?" * len(BLOCKING))
        row = c.execute(f"SELECT * FROM batches WHERE line_id=? AND status IN ({marks}) "
                        f"ORDER BY started_at DESC LIMIT 1", (line_id, *BLOCKING)).fetchone()
        if row is None:
            return None
        return {**dict(row), "counts": _counts(c, row["batch_id"])}
 
 
def list_products(batch_id):
    with get_conn() as c:
        _batch(c, batch_id)
        rows = [dict(r) for r in c.execute(
            "SELECT * FROM products WHERE batch_id=? ORDER BY position", (batch_id,))]
    for p in rows:
        p["reasons"] = json.loads(p["reasons"])
        p["top_factors"] = json.loads(p["top_factors"])
    return rows
 
 
def get_report(batch_id):
    with get_conn() as c:
        _batch(c, batch_id)
        row = c.execute("SELECT report_json FROM batch_reports WHERE batch_id=? "
                        "ORDER BY report_id DESC LIMIT 1", (batch_id,)).fetchone()
        if row is None:
            raise RuntimeError(f"Batch {batch_id} is not finished - no report yet.")
        return json.loads(row["report_json"])
 
 
def open_alerts():
    with get_conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT * FROM alerts WHERE acknowledged_at IS NULL ORDER BY alert_id DESC")]
 
 
def ack_alert(alert_id, user):
    with get_conn() as c:
        cur = c.execute("UPDATE alerts SET acknowledged_by=?, acknowledged_at=? "
                        "WHERE alert_id=? AND acknowledged_at IS NULL", (user, _now(), alert_id))
        if cur.rowcount == 0:
            raise KeyError(f"No open alert {alert_id}")


def pending_reviews(batch_id):
    with get_conn() as c:
        return c.execute(
            "SELECT COUNT(*) FROM products WHERE batch_id=? "
            "AND final_status='REVIEW' AND review_result IS NULL",
            (batch_id,)).fetchone()[0]

def review_product(product_id, result, user, note):
    if result not in ("PASS", "REJECT"):
        raise ValueError("Result must be PASS or REJECT.")
    if not note.strip():
        raise ValueError("A written note is required.")
    with get_conn() as c:
        p = c.execute("SELECT batch_id FROM products WHERE product_id=? "
                      "AND final_status='REVIEW'", (product_id,)).fetchone()
        if not p:
            raise LookupError("No REVIEW product with that ID.")
        now = datetime.now().isoformat(timespec="seconds")
        c.execute("UPDATE products SET review_result=?, reviewed_by=?, "
                  "reviewed_at=?, review_note=? WHERE product_id=?",
                  (result, user, now, note, product_id))
        c.execute("INSERT INTO events (batch_id, at, actor, action, note) "
                  "VALUES (?,?,?,?,?)",
                  (p["batch_id"], now, user, f"REVIEW_{result}", f"{product_id}: {note}"))
    return {"product_id": product_id, "result": result}