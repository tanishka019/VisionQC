"""Today's overview as a downloadable PDF (summary, hourly activity, last 7 days, every inspection, FAIL heatmaps)."""
import io
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

INK, MUTED, LINE = colors.HexColor("#0e0e0e"), colors.HexColor("#6b6a66"), colors.HexColor("#d9d8d2")
PASS_C, FAIL_C = colors.HexColor("#4f7d12"), colors.HexColor("#e8472b")
MAX_HEATMAPS = 8
MAX_ROWS = 300


def _table(data, widths, result_col=None, header=True):
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    style = [
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        style += [("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("TEXTCOLOR", (0, 0), (-1, 0), MUTED)]
    if result_col is not None:
        for i, row in enumerate(data[1:], 1):
            style.append(("TEXTCOLOR", (result_col, i), (result_col, i), FAIL_C if row[result_col] == "FAIL" else PASS_C))
            style.append(("FONTNAME", (result_col, i), (result_col, i), "Helvetica-Bold"))
    t.setStyle(TableStyle(style))
    return t


def build_pdf(stats: dict, hourly: list, weekly: list, rows: list, model_info: dict, now: datetime = None) -> bytes:
    now = now or datetime.now()
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=22, alignment=0, textColor=INK, spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=12, textColor=INK, spaceBefore=14, spaceAfter=6, keepWithNext=1)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9, textColor=MUTED)

    out = io.BytesIO()
    doc = SimpleDocTemplate(out, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
                            title=f"VisionQC daily report {now:%Y-%m-%d}", author="VisionQC")
    el = [Paragraph("VisionQC daily report", h1),
          Paragraph(f"{now:%A, %d %B %Y} &nbsp;·&nbsp; generated {now:%H:%M}", body), Spacer(1, 6)]

    total = stats.get("total", 0)
    pass_rate = f"{stats.get('passed', 0) / total * 100:.1f}%" if total else "-"
    el.append(Paragraph("Summary", h2))
    el.append(_table([
        ["Inspected", "Passed", "Failed", "Pass rate", "Rejection rate", "Avg confidence"],
        [total, stats.get("passed", 0), stats.get("failed", 0), pass_rate, f"{stats.get('rejection_rate', 0)}%",
         f"{stats.get('avg_confidence', 0)}%" if total else "-"],
    ], [28 * mm] * 6))

    if model_info:
        el.append(Paragraph("Model", h2))
        el.append(_table([[k, str(v)] for k, v in model_info.items()], [45 * mm, 120 * mm], header=False))

    el.append(Paragraph("Activity by hour", h2))
    if hourly:
        el.append(_table([["Hour", "Passed", "Failed", "Total"]] +
                         [[f"{int(r['hour']):02d}:00", r["passed"] or 0, r["failed"] or 0, (r["passed"] or 0) + (r["failed"] or 0)] for r in hourly],
                         [30 * mm] * 4))
    else:
        el.append(Paragraph("No inspections yet today.", body))

    el.append(Paragraph("Last 7 days", h2))
    if weekly:
        el.append(_table([["Day", "Passed", "Failed", "Total"]] +
                         [[r["day"], r["passed"] or 0, r["failed"] or 0, r["total"] or 0] for r in weekly], [30 * mm] * 4))
    else:
        el.append(Paragraph("No data.", body))

    el.append(Paragraph(f"Today's inspections ({len(rows)})", h2))
    if rows:
        shown = rows[-MAX_ROWS:]
        el.append(_table([["#", "Time", "File", "Result", "Score", "Confidence"]] +
                         [[r["id"], str(r["timestamp"])[11:19], escape(str(r.get("filename") or "-"))[:38], r["result"],
                           f"{r['anomaly_score'] * 100:.1f}%", f"{r['confidence']}%"] for r in shown],
                         [14 * mm, 22 * mm, 62 * mm, 20 * mm, 22 * mm, 25 * mm], result_col=3))
        if len(rows) > MAX_ROWS:
            el.append(Paragraph(f"Showing the latest {MAX_ROWS} of {len(rows)}.", body))
    else:
        el.append(Paragraph("No inspections yet today.", body))

    fails = [r for r in rows if r["result"] == "FAIL" and r.get("heatmap_path") and Path(r["heatmap_path"]).exists()][-MAX_HEATMAPS:]
    if fails:
        el.append(Paragraph("Failed parts: heatmaps", h2))
        cells = []
        for r in fails:
            w = 80 * mm
            try:
                from PIL import Image as PILImage
                with PILImage.open(r["heatmap_path"]) as im:
                    ratio = im.height / im.width
            except Exception:
                continue
            cells.append([Image(r["heatmap_path"], width=w, height=w * min(ratio, 1.0)),
                          Paragraph(f"#{r['id']} · {escape(str(r.get('filename') or ''))[:30]} · score {r['anomaly_score'] * 100:.0f}%", body)])
        for i in range(0, len(cells), 2):
            pair = cells[i:i + 2]
            row = [[c[0], c[1]] for c in pair] + [""] * (2 - len(pair))
            el.append(Table([row], colWidths=[85 * mm, 85 * mm]))
            el.append(Spacer(1, 6))

    doc.build(el)
    return out.getvalue()
