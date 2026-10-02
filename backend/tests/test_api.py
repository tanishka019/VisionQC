"""API tests with the ML layer mocked out (fast; no torch/anomalib calls)."""
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import db
import main
from ml import model as ml
from conftest import make_png


@pytest.fixture(autouse=True)
def clean_state():
    db.clear_history()
    db.set_config("threshold", "0.5")
    db.set_config("model_trained", "false")
    main._train_state.update({"status": "idle", "progress": 0, "message": ""})
    yield


@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.fixture
def trained(monkeypatch):
    """Pretend a model is trained; inspection returns a score derived from image brightness."""
    db.set_config("model_trained", "true")
    monkeypatch.setattr(ml, "is_trained", lambda: True)

    def fake_inspect(image_path: Path, threshold: float):
        img = Image.open(image_path).convert("L")
        score = round(1 - sum(img.getdata()) / (255 * img.width * img.height), 4)
        heatmap = ml.RESULTS_DIR / f"heatmap_{image_path.stem}.jpg"
        img.save(heatmap)
        result = "FAIL" if score >= threshold else "PASS"
        conf = round((score if result == "FAIL" else 1 - score) * 100, 1)
        return {"score": score, "raw_score": score, "confidence": conf,
                "result": result, "heatmap_path": str(heatmap)}

    monkeypatch.setattr(ml, "inspect", fake_inspect)


def upload(name="part.png", data=None):
    return ("file", (name, data if data is not None else make_png(), "image/png"))


# ─── Health / config ──────────────────────────────────────────────────────────
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["model_trained"] is False


def test_trained_requires_checkpoint(client, monkeypatch):
    db.set_config("model_trained", "true")
    monkeypatch.setattr(ml, "is_trained", lambda: False)
    assert client.get("/model/status").json()["trained"] is False


def test_threshold_roundtrip_and_validation(client):
    assert client.post("/threshold", json={"threshold": 0.7}).json()["saved"] is True
    assert client.get("/threshold").json()["threshold"] == 0.7
    assert client.post("/threshold", json={"threshold": 1.5}).status_code == 400


# ─── Inspect ──────────────────────────────────────────────────────────────────
def test_inspect_requires_trained_model(client):
    r = client.post("/inspect", files=[upload()])
    assert r.status_code == 400


def test_inspect_pass_and_fail(client, trained):
    bright = client.post("/inspect", files=[upload(data=make_png((250, 250, 250)))]).json()
    dark   = client.post("/inspect", files=[upload(data=make_png((5, 5, 5)))]).json()
    assert bright["result"] == "PASS" and dark["result"] == "FAIL"
    assert client.get(bright["heatmap_url"]).status_code == 200
    assert client.get(bright["image_url"]).status_code == 200


def test_inspect_rejects_bad_files(client, trained):
    assert client.post("/inspect", files=[upload("notes.txt", b"hello")]).status_code == 400
    assert client.post("/inspect", files=[upload("fake.png", b"not an image")]).status_code == 400


def test_inspect_rejects_oversized(client, trained, monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_MB", 0.0001)
    assert client.post("/inspect", files=[upload()]).status_code == 400


def test_batch_inspect_summary(client, trained):
    files = [
        ("files", ("a.png", make_png((250, 250, 250)), "image/png")),
        ("files", ("b.png", make_png((5, 5, 5)), "image/png")),
        ("files", ("c.txt", b"x", "text/plain")),
    ]
    body = client.post("/inspect/batch", files=files).json()
    assert body["summary"] == {"total": 3, "passed": 1, "failed": 1}
    assert body["results"][2]["result"] == "ERROR"


def test_websocket_receives_inspection(client, trained):
    with client.websocket_connect("/ws/live") as ws:
        client.post("/inspect", files=[upload()])
        msg = ws.receive_json()
    assert msg["type"] == "inspection" and msg["result"] in ("PASS", "FAIL")


# ─── History / stats ──────────────────────────────────────────────────────────
def test_history_filter_and_delete_removes_files(client, trained):
    client.post("/inspect", files=[upload(data=make_png((250, 250, 250)))])
    fail = client.post("/inspect", files=[upload(data=make_png((5, 5, 5)))]).json()

    h = client.get("/history", params={"filter": "FAIL"}).json()
    assert h["total"] == 1 and h["inspections"][0]["id"] == fail["id"]

    image_file = main.UPLOADS_DIR / Path(fail["image_url"]).name
    assert image_file.exists()
    assert client.delete(f"/history/{fail['id']}").json()["deleted"] is True
    assert not image_file.exists()
    assert client.delete(f"/history/{fail['id']}").status_code == 404

    assert client.delete("/history").json()["cleared"] == 1
    assert client.get("/history").json()["total"] == 0


def test_stats(client, trained):
    client.post("/inspect", files=[upload(data=make_png((5, 5, 5)))])
    today = client.get("/stats").json()["today"]
    assert today["total"] == 1 and today["failed"] == 1 and today["rejection_rate"] == 100.0


def test_prune_old_files(tmp_path):
    for i in range(5):
        (tmp_path / f"{i}.jpg").write_bytes(b"x")
        time.sleep(0.01)
    main.prune_old_files(tmp_path, keep=2)
    assert sorted(f.name for f in tmp_path.iterdir()) == ["3.jpg", "4.jpg"]


# ─── Training ─────────────────────────────────────────────────────────────────
def test_train_needs_enough_valid_images(client):
    files = [("files", (f"{i}.png", make_png(), "image/png")) for i in range(3)]
    assert client.post("/train", files=files).status_code == 400
    files = [("files", (f"{i}.txt", b"x", "text/plain")) for i in range(6)]
    assert client.post("/train", files=files).status_code == 400


def test_train_flow_updates_state(client, monkeypatch):
    def fake_train(folder, product_name, progress=None):
        assert len(list(Path(folder).iterdir())) == 6
        progress(50, "halfway")
        return {"status": "trained", "image_count": 6}

    monkeypatch.setattr(ml, "train", fake_train)
    monkeypatch.setattr(ml, "is_trained", lambda: True)
    files = [("files", (f"{i}.png", make_png(), "image/png")) for i in range(6)]
    with client:  # keeps the event loop alive for the background task
        r = client.post("/train", files=files, data={"product_name": "Screw"})
        assert r.json()["status"] == "started"
        for _ in range(100):
            if main._train_state["status"] != "running":
                break
            time.sleep(0.05)
    assert main._train_state["status"] == "done", main._train_state
    status = client.get("/model/status").json()
    assert status["trained"] is True and status["product_name"] == "Screw"


# ─── Auth ─────────────────────────────────────────────────────────────────────
def test_api_key_protects_writes(client, monkeypatch):
    monkeypatch.setattr(main, "API_KEY", "secret")
    assert client.get("/threshold").status_code == 200
    assert client.post("/threshold", json={"threshold": 0.6}).status_code == 401
    ok = client.post("/threshold", json={"threshold": 0.6}, headers={"X-API-Key": "secret"})
    assert ok.status_code == 200


def test_api_key_rejection_has_cors_headers(client, monkeypatch):
    monkeypatch.setattr(main, "API_KEY", "secret")
    r = client.post("/threshold", json={"threshold": 0.6},
                    headers={"Origin": "http://localhost:5173"})
    assert r.status_code == 401
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"
