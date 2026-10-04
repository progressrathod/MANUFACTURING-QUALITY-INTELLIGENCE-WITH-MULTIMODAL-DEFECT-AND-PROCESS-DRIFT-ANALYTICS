"""Tests for the random production generator (images + number of defective products).
They use a temporary database and fake image folders, so your real data is never touched.
Run: pytest -q"""
import pytest
 
from src import production, batch_manager as bm
from src.database import init_db
from src.config import MIN_DEFECTIVE_PER_BATCH, MAX_DEFECTIVE_PER_BATCH
 
 
def _make_images(root, n_def=30, n_ok=30):
    for folder, prefix, n in (("def_front", "cast_def", n_def), ("ok_front", "cast_ok", n_ok)):
        (root / folder).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (root / folder / f"{prefix}_{i}.jpeg").write_bytes(b"x")     # content is not opened here
 
 
@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr("src.database.DB_PATH", tmp_path / "test.db")
 
 
def test_plan_uses_both_folders_and_respects_the_defective_limit(tmp_path, monkeypatch):
    _make_images(tmp_path / "images")
    monkeypatch.setattr(production, "IMG_RAW", tmp_path / "images")
    monkeypatch.setattr(production, "IMG_SPLIT", tmp_path / "nothing")
    for _ in range(20):                                   # random, so repeat
        b = bm.start_batch()
        n_bad = production.plan_batch(b)
        assert MIN_DEFECTIVE_PER_BATCH <= n_bad <= MAX_DEFECTIVE_PER_BATCH
        with production.get_conn() as c:
            rows = c.execute("SELECT truth, image_file FROM batch_plan WHERE batch_id=?", (b,)).fetchall()
        assert len(rows) == 50
        assert sum(r["truth"] for r in rows) == n_bad
        assert all(r["image_file"].startswith("cast_def") for r in rows if r["truth"] == 1)
        assert all(r["image_file"].startswith("cast_ok") for r in rows if r["truth"] == 0)
        bm.close_batch(b)
 
 
def test_falls_back_to_images_split_when_raw_folder_is_missing(tmp_path, monkeypatch):
    _make_images(tmp_path / "split" / "train")
    monkeypatch.setattr(production, "IMG_RAW", tmp_path / "missing")
    monkeypatch.setattr(production, "IMG_SPLIT", tmp_path / "split")
    production.require_images()                                          # must not raise
    assert production._find_image("def_front", "cast_def_3.jpeg") is not None
 
 
def test_missing_images_give_a_clear_error_and_no_stuck_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(production, "IMG_RAW", tmp_path / "missing")
    monkeypatch.setattr(production, "IMG_SPLIT", tmp_path / "missing_too")
    with pytest.raises(RuntimeError, match="No images found"):
        production.start_and_produce()
    init_db()
    assert bm.current_batch() is None                      # no RUNNING batch was left behind