import daily_report


def test_pdf_builds_with_and_without_data():
    stats = {"total": 3, "passed": 2, "failed": 1, "rejection_rate": 33.3, "avg_confidence": 88.0}
    rows = [{"id": 1, "timestamp": "2026-10-03 06:25:12", "anomaly_score": 0.2, "confidence": 88.0,
             "result": "PASS", "threshold": 0.5, "heatmap_path": "", "filename": "a<b>.png"}]
    full = daily_report.build_pdf(stats, [{"hour": "06", "passed": 2, "failed": 1}], [], rows, {"Product": "screw"})
    empty = daily_report.build_pdf({"total": 0}, [], [], [], {})
    assert full.startswith(b"%PDF") and empty.startswith(b"%PDF")


def test_report_endpoint_returns_a_pdf():
    from fastapi.testclient import TestClient
    import main
    with TestClient(main.app) as client:
        r = client.get("/report/today.pdf")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content.startswith(b"%PDF")
    assert "attachment" in r.headers["content-disposition"]
