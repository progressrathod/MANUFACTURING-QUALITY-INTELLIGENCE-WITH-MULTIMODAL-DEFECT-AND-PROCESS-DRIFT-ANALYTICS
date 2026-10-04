"""Step 26 - tests for batch tracking. Run: pytest -q
They use a temporary database, so your real history is never touched."""
import pytest
 
from src import batch_manager as bm
from src.config import BATCH_SIZE, STOP_DEFECTS
 
ROW = {"temperature": 72, "pressure": 5.2, "machine_speed": 1450, "vibration": 2.4,
       "humidity": 55, "material_thickness": 4.0, "cycle_time": 48, "tool_wear": 35}
 
 
@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr("src.database.DB_PATH", tmp_path / "test.db")
 
 
def test_product_ids_are_unique_and_follow_the_batch_id():
    b = bm.start_batch()
    ids = [bm.inspect_product(b, ROW, "OK", 99.0)["product_id"] for _ in range(5)]
    assert len(set(ids)) == 5
    assert ids[0] == f"{b}-P01" and ids[4] == f"{b}-P05"
 
 
def test_batch_closes_after_50_products_and_report_counts_add_up():
    b = bm.start_batch()
    for i in range(BATCH_SIZE):
        label = "DEFECTIVE" if i in (3, 10, 20, 30, 40, 41) else "OK"   # 6 defective
        bm.inspect_product(b, ROW, label, 99.0)
    rep = bm.get_report(b)
    assert rep["total"] == BATCH_SIZE
    assert rep["defective"] == 6
    assert rep["ok"] + rep["defective"] + rep["review"] == BATCH_SIZE
    assert len(rep["defective_products"]) == 6
    assert all(p["reasons"] for p in rep["defective_products"])   # every defect has a reason
    with pytest.raises(RuntimeError):                              # batch is closed now
        bm.inspect_product(b, ROW, "OK", 99.0)
 
 
def test_line_stops_at_the_stop_rule_and_next_batch_is_blocked():
    b = bm.start_batch()
    for _ in range(STOP_DEFECTS):
        bm.inspect_product(b, ROW, "DEFECTIVE", 99.0)
    assert bm.current_batch()["status"] == "PAUSED"
    with pytest.raises(RuntimeError):
        bm.inspect_product(b, ROW, "OK", 99.0)       # cannot add products while stopped
    with pytest.raises(RuntimeError):
        bm.start_batch()                              # cannot start another batch either
    with pytest.raises(ValueError):
        bm.resume_batch(b, "manager", "")             # a note is required
    bm.resume_batch(b, "manager", "Replaced the worn tool")
    assert bm.current_batch()["status"] == "RUNNING"
 
 
def test_on_hold_batch_needs_a_decision_before_the_next_batch():
    b = bm.start_batch()
    for _ in range(STOP_DEFECTS):
        bm.inspect_product(b, ROW, "DEFECTIVE", 99.0)
    bm.close_batch(b)
    assert bm.current_batch()["status"] == "ON_HOLD"
    with pytest.raises(RuntimeError):
        bm.start_batch()
    bm.decide_batch(b, "SCRAP", "manager", "Tool failure, whole batch rejected")
    assert bm.start_batch() != b                      # now the next batch can start
