"""Step 16 - Combine image result + process result + drift status."""
from typing import Optional
 
from src.process_prediction.predict import predict_process
from src.process_prediction.explain_shap import explain
from src.drift_detection.predict import detect_drift
 
 
def overall_status(image_label: Optional[str], process_label: str, drift: bool) -> str:
    """Simple, editable rules. Change them to suit your project."""
    image_bad = image_label == "DEFECTIVE"
    process_bad = process_label == "DEFECTIVE"
 
    if image_bad and process_bad:
        return "CRITICAL - defect confirmed by image and process"
    if (image_bad or process_bad) and drift:
        return "HIGH - defect signal with unusual process behaviour"
    if image_bad:
        return "DEFECT - product failed visual inspection"
    if process_bad:
        return "WARNING - process conditions look defect-prone"
    if drift:
        return "WATCH - process behaviour is unusual"
    return "OK - no issues detected"
 
 
def quality_report(process_row: dict, image_label: Optional[str] = None) -> dict:
    proc = predict_process(process_row)
    drift = detect_drift(process_row)
    return {
        "image_result": image_label or "NOT PROVIDED",
        "process_result": proc,
        "process_status": drift,
        "major_factors": explain(process_row, top_n=3),
        "overall_status": overall_status(image_label, proc["label"], drift["is_drift"]),
    }
 
 
if __name__ == "__main__":
    import json
    sample = {"temperature": 98, "pressure": 7.4, "machine_speed": 1990, "vibration": 5.0,
              "humidity": 88, "material_thickness": 5.1, "cycle_time": 68, "tool_wear": 90}
    print(json.dumps(quality_report(sample, "DEFECTIVE"), indent=2))
