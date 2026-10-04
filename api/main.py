"""Step 17 - FastAPI backend.   Run:  uvicorn api.main:app --reload"""
import io
from typing import Literal, Optional
 
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
 
from src.image_classification.predict import predict_image
from src.process_prediction.predict import predict_process
from src.process_prediction.explain_shap import explain
from src.drift_detection.predict import detect_drift
from src.quality_logic import quality_report
 
app = FastAPI(title="Manufacturing Quality Intelligence API", version="1.0")
 
 
class ProcessInput(BaseModel):
    """Input validation - bad values return HTTP 422 automatically."""
    temperature: float = Field(..., gt=0, lt=500, examples=[72.0])
    pressure: float = Field(..., gt=0, lt=100, examples=[5.2])
    machine_speed: float = Field(..., gt=0, lt=10000, examples=[1450])
    vibration: float = Field(..., ge=0, lt=100, examples=[2.4])
    humidity: float = Field(..., ge=0, le=100, examples=[55])
    material_thickness: float = Field(..., gt=0, lt=100, examples=[4.0])
    cycle_time: float = Field(..., gt=0, lt=1000, examples=[48])
    tool_wear: float = Field(..., ge=0, lt=1000, examples=[35])
 
 
class QualityRequest(BaseModel):
    process: ProcessInput
    image_label: Optional[Literal["OK", "DEFECTIVE"]] = None
 
 
@app.get("/health")
def health():
    return {"status": "ok"}
 
 
@app.post("/predict-image")
async def predict_image_endpoint(file: UploadFile = File(...)):
    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
    return predict_image(img)
 
 
@app.post("/predict-process")
def predict_process_endpoint(item: ProcessInput):
    row = item.model_dump()
    result = predict_process(row)
    result["top_factors"] = explain(row, top_n=5)
    return result
 
 
@app.post("/detect-drift")
def detect_drift_endpoint(item: ProcessInput):
    return detect_drift(item.model_dump())
 
 
@app.post("/quality-report")
def quality_report_endpoint(req: QualityRequest):
    return quality_report(req.process.model_dump(), req.image_label)

# ---- Step 24 - add to api/main.py ----
# 1) add this import near the other imports:
from src import batch_manager as bm
 
 
# 2) add these classes and endpoints at the END of the file:
class InspectRequest(BaseModel):
    process: ProcessInput
    image_label: Optional[Literal["OK", "DEFECTIVE"]] = None
    image_confidence: Optional[float] = None
    image_file: Optional[str] = None
 
 
class ActionRequest(BaseModel):
    user: str = Field(..., min_length=1, examples=["Priya (shift manager)"])
    note: str = ""
 
 
class DecisionRequest(ActionRequest):
    decision: Literal["RELEASE", "SCRAP"]
 
 
def _run(fn, *args, **kwargs):
    """Turn batch-manager errors into clear HTTP errors."""
    try:
        return fn(*args, **kwargs)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e).strip("'\""))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
 
 
@app.post("/batches/start")
def batches_start(req: ActionRequest):
    return {"batch_id": _run(bm.start_batch, operator=req.user)}
 
 
@app.get("/batches/current")
def batches_current():
    return bm.current_batch() or {}
 
 
@app.get("/batches")
def batches_history():
    return bm.list_batches()
 
 
@app.post("/batches/{batch_id}/inspect")
def batches_inspect(batch_id: str, req: InspectRequest):
    return _run(bm.inspect_product, batch_id, req.process.model_dump(),
                req.image_label, req.image_confidence, req.image_file)
 
 
@app.post("/batches/{batch_id}/stop")
def batches_stop(batch_id: str, req: ActionRequest):
    _run(bm.stop_batch, batch_id, req.user, req.note)
    return {"status": "PAUSED"}
 
 
@app.post("/batches/{batch_id}/resume")
def batches_resume(batch_id: str, req: ActionRequest):
    _run(bm.resume_batch, batch_id, req.user, req.note)
    return {"status": "RUNNING"}
 
 
@app.post("/batches/{batch_id}/close")
def batches_close(batch_id: str, req: ActionRequest):
    return _run(bm.close_batch, batch_id, req.user)
 
 
@app.post("/batches/{batch_id}/decision")
def batches_decision(batch_id: str, req: DecisionRequest):
    _run(bm.decide_batch, batch_id, req.decision, req.user, req.note)
    return {"decision": req.decision}
 
 
@app.get("/batches/{batch_id}/report")
def batches_report(batch_id: str):
    return _run(bm.get_report, batch_id)
 
 
@app.get("/batches/{batch_id}/products")
def batches_products(batch_id: str):
    return _run(bm.list_products, batch_id)
 
 
@app.get("/alerts")
def alerts_open():
    return bm.open_alerts()
 
 
@app.post("/alerts/{alert_id}/ack")
def alerts_ack(alert_id: int, req: ActionRequest):
    _run(bm.ack_alert, alert_id, req.user)
    return {"acknowledged": alert_id}

# ---- Step 29 - add to api/main.py ----
# 1) add this import near the other imports:
from src import production
 
 
# 2) add these two endpoints at the END of the file:
@app.post("/batches/start-and-produce")
def batches_start_and_produce(req: ActionRequest):
    """Start a new batch, pick its 50 random products and inspect them one by one."""
    return _run(production.start_and_produce, req.user)
 
 
@app.post("/batches/{batch_id}/produce")
def batches_produce(batch_id: str):
    """Produce the remaining products (use after the manager resumed a paused batch)."""
    return _run(production.run_production, batch_id)
