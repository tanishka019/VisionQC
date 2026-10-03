import base64
import html
import io
import os
import sys
import shutil
import tempfile
import zipfile
from pathlib import Path
from datetime import datetime, timedelta
import streamlit as st
from PIL import Image, ImageDraw

# Ensure backend directory is in Python path
ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml import model
import db

# ----------------- PAGE CONFIG -----------------
def _page_icon() -> Image.Image:
    """Brand mark (lime viewfinder on a dark tile), drawn at runtime so it works on any Streamlit version."""
    size, pad, arm, w = 128, 30, 26, 8
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=30, fill=(20, 20, 20, 255))
    lime = (203, 245, 92, 255)
    for x0, x1 in ((pad, pad + arm), (size - pad - arm, size - pad)):
        for y in (pad, size - pad - w):
            d.rectangle([x0, y, x1, y + w - 1], fill=lime)
    for x in (pad, size - pad - w):
        for y0, y1 in ((pad, pad + arm), (size - pad - arm, size - pad)):
            d.rectangle([x, y0, x + w - 1, y1], fill=lime)
    c = size // 2
    d.ellipse([c - 8, c - 8, c + 8, c + 8], fill=lime)
    return img


st.set_page_config(
    page_title="VisionQC",
    page_icon=_page_icon(),
    layout="wide",
    initial_sidebar_state="auto",
)

# Design tokens mirror the React UI: warm paper, near-black ink, a lime accent that doubles as "pass",
# vermilion for "fail". Fonts are served from ./static/fonts (Streamlit static serving).
STYLE = """
<style>
    @font-face { font-family: 'Inter'; font-weight: 400; font-display: swap; src: url('app/static/fonts/inter-400.woff2') format('woff2'); }
    @font-face { font-family: 'Inter'; font-weight: 500; font-display: swap; src: url('app/static/fonts/inter-500.woff2') format('woff2'); }
    @font-face { font-family: 'Inter'; font-weight: 600; font-display: swap; src: url('app/static/fonts/inter-600.woff2') format('woff2'); }
    @font-face { font-family: 'JetBrains Mono'; font-weight: 400; font-display: swap; src: url('app/static/fonts/jetbrains-mono-400.woff2') format('woff2'); }
    @font-face { font-family: 'JetBrains Mono'; font-weight: 500; font-display: swap; src: url('app/static/fonts/jetbrains-mono-500.woff2') format('woff2'); }
    @font-face { font-family: 'Instrument Serif'; font-weight: 400; font-display: swap; src: url('app/static/fonts/instrument-serif-400.woff2') format('woff2'); }

    :root {
        --paper: #ecebe6; --card: #f7f6f2;
        --ink: #0e0e0e; --ink-2: #4b4b47; --ink-3: #85847f;
        --line: #d9d8d1; --line-strong: #c4c3bb;
        --dark: #141414; --dark-2: #1d1d1d; --dark-line: #2c2c2b;
        --on-dark: #f3f2ee; --on-dark-2: #a2a19b; --on-dark-3: #6c6b66;
        --lime: #cbf55c; --pass: #14733a; --fail: #e8472b;
        --serif: 'Instrument Serif', Georgia, 'Times New Roman', serif;
        --font: 'Inter', ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
        --mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }

    html, body, .stApp, p, label, li, button, input, textarea,
    [data-testid="stMarkdownContainer"], [data-baseweb="tab"], [data-testid="stCaptionContainer"] {
        font-family: var(--font);
    }

    /* Chrome we don't need */
    #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { visibility: hidden; height: 0; }
    [data-testid="stHeader"] { background: transparent; }

    .stApp { background: var(--paper); }
    .block-container { max-width: 1200px; padding: 2.25rem 2rem 5rem; }

    /* Type */
    h1 { font-family: var(--serif) !important; font-weight: 400 !important; font-size: 3.4rem !important; letter-spacing: -0.025em !important; line-height: 1 !important; padding: 0 !important; }
    h2, h3 { font-weight: 600 !important; letter-spacing: -0.01em !important; }
    h3 { font-size: 1.02rem !important; }
    .page-sub { color: var(--ink-3); margin: 0.6rem 0 2rem; font-size: .95rem; }
    .label { font-family: var(--mono); font-size: .68rem; font-weight: 500; letter-spacing: .1em; text-transform: uppercase; color: var(--ink-3); }
    .mono { font-family: var(--mono); }

    /* Sidebar (dark) */
    [data-testid="stSidebar"] { background: var(--dark); border-right: 0; }
    [data-testid="stSidebar"] * { color: var(--on-dark); }
    [data-testid="stSidebar"] hr { border-color: var(--dark-line) !important; }
    [data-testid="stSidebar"] .stSlider label p { font-family: var(--mono); font-size: .68rem; letter-spacing: .1em; text-transform: uppercase; color: var(--on-dark-3); }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: var(--on-dark-3) !important; }
    .brand { display: flex; align-items: center; gap: 11px; margin: .25rem 0 1.75rem; }
    .brand-mark { width: 32px; height: 32px; border-radius: 10px; background: var(--dark-2); color: var(--lime); display: grid; place-items: center; }
    .brand-name { font-family: var(--serif); font-size: 1.65rem; letter-spacing: -0.01em; line-height: 1; color: var(--on-dark); }
    .model-chip { display: flex; align-items: center; gap: 9px; background: var(--dark-2); padding: 9px 14px; border-radius: 999px; font-family: var(--mono); font-size: .74rem; color: var(--on-dark) !important; margin: .25rem 0 1.25rem; }
    .model-chip .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--on-dark-3); flex-shrink: 0; }
    .model-chip .dot.on { background: var(--lime); box-shadow: 0 0 0 3px rgba(203,245,92,.18); }

    /* Radios as pills: sidebar navigation + inline filters */
    [role="radiogroup"] label > div:first-child { display: none; }
    [data-testid="stSidebar"] [role="radiogroup"] { gap: 4px; }
    [data-testid="stSidebar"] [role="radiogroup"] label { width: 100%; padding: 9px 16px; border-radius: 999px; margin: 0; cursor: pointer; }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover { background: var(--dark-2); }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) { background: rgba(203,245,92,.14); }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p { color: var(--lime); }
    .stMain [role="radiogroup"] { gap: 8px; }
    .stMain [role="radiogroup"] label { border: 1px solid var(--line-strong); border-radius: 999px; padding: 6px 18px; margin: 0; cursor: pointer; background: transparent; }
    .stMain [role="radiogroup"] label:has(input:checked) { background: var(--ink); border-color: var(--ink); }
    .stMain [role="radiogroup"] label:has(input:checked) p { color: var(--on-dark); }

    /* Tabs as a pill group (react-aria roles in new Streamlit, BaseWeb attributes in older ones) */
    [role="tablist"], [data-baseweb="tab-list"] { background: var(--card) !important; border: 1px solid var(--line) !important; border-radius: 999px !important; padding: 4px !important; gap: 2px !important; width: fit-content !important; }
    [role="tab"], [data-baseweb="tab"] { border-radius: 999px !important; padding: 6px 20px !important; height: auto !important; font-weight: 500 !important; background: transparent !important; }
    [role="tab"][aria-selected="true"], [data-baseweb="tab"][aria-selected="true"] { background: var(--ink) !important; }
    [role="tab"][aria-selected="true"] p, [data-baseweb="tab"][aria-selected="true"] p { color: var(--on-dark) !important; }
    [data-baseweb="tab-highlight"], [data-baseweb="tab-border"], [data-testid="stTabHighlight"], [data-testid="stTabBorder"] { display: none !important; }
    [role="tablist"]::after, [role="tablist"]::before, [role="tab"]::after, [role="tab"]::before { display: none !important; }
    [role="tablist"] > *:not([role="tab"]) { display: none !important; }
    [role="tab"] > div:last-child:not(:first-child) { display: none !important; }

    /* Controls */
    .stButton > button, .stDownloadButton > button { border-radius: 999px; font-weight: 500; height: 2.75rem; padding: 0 1.7rem; background: transparent; border: 1px solid var(--line-strong); color: var(--ink); }
    .stButton > button:hover { border-color: var(--ink); color: var(--ink); background: transparent; }
    .stButton > button[kind="primary"] { background: var(--ink); border-color: var(--ink); color: var(--on-dark); }
    .stButton > button[kind="primary"]:hover { background: #2a2a2a; color: var(--on-dark); }
    [data-baseweb="input"], [data-baseweb="base-input"] { border-radius: 999px !important; }
    [data-baseweb="select"] > div { border-radius: 14px; }
    [data-testid="stFileUploaderDropzone"] { border: 1px dashed var(--line-strong); border-radius: 14px; background: var(--card); }
    [data-testid="stAlert"] { border-radius: 14px; border: 1px solid var(--line); background: var(--card) !important; }
    [data-testid="stAlert"] * { color: var(--ink) !important; }
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--ink-3) !important; opacity: 1 !important; }
    code { color: var(--ink) !important; background: var(--card) !important; border: 1px solid var(--line); border-radius: 6px; font-family: var(--mono) !important; font-size: .82em !important; }
    hr { border-color: var(--line) !important; }
    [data-testid="stImage"] img { border-radius: 0; }

    /* Panels */
    .panel { background: var(--dark); color: var(--on-dark); border-radius: 20px; padding: 26px; }
    .panel .label { color: var(--on-dark-3); }
    .panel-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 18px; }
    .panel-head b { font-size: .95rem; font-weight: 600; letter-spacing: -0.01em; }
    .panel-head span { font-size: .82rem; color: var(--on-dark-2); }
    .card { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 4px 22px; }

    /* Hero */
    .hero { padding: 30px 34px 26px; }
    .hero-top { display: flex; justify-content: space-between; margin-bottom: 18px; }
    .hero-grid { display: grid; grid-template-columns: 1.15fr 1fr; gap: 44px; align-items: end; }
    .hero-big .big { font-family: var(--serif); font-size: 11rem; line-height: .8; letter-spacing: -.04em; margin-top: 14px; }
    .hero-big p { margin: 16px 0 0; color: var(--on-dark-2); font-size: .9rem; }
    .hero-side { display: grid; grid-template-columns: repeat(3, 1fr); border-left: 1px solid var(--dark-line); }
    .hero-stat { padding: 4px 0 4px 24px; }
    .hero-stat .figure { font-family: var(--serif); font-size: 3.6rem; line-height: 1; letter-spacing: -.03em; margin: 12px 0 6px; }
    .hero-stat .figure.pass { color: var(--lime); }
    .hero-stat .figure.fail { color: #ff6a4d; }
    .hero-stat small { color: var(--on-dark-3); font-size: .78rem; }
    .strip { margin-top: 30px; padding-top: 22px; border-top: 1px solid var(--dark-line); }
    .strip-bars { display: flex; align-items: flex-end; gap: 4px; height: 84px; }
    .strip-col { flex: 1; height: 100%; display: flex; flex-direction: column; justify-content: flex-end; gap: 2px; min-width: 0; }
    .strip-col i { display: block; border-radius: 2px; }
    .strip-col .p { background: var(--lime); }
    .strip-col .f { background: #ff6a4d; }
    .strip-col .e { background: var(--dark-line); height: 3px; }
    .strip-ticks { display: flex; justify-content: space-between; margin-top: 10px; font-family: var(--mono); font-size: .66rem; color: var(--on-dark-3); }

    /* Week chart */
    .week { display: flex; align-items: flex-end; gap: 14px; height: 190px; padding-top: 8px; }
    .week-day { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: flex-end; height: 100%; gap: 8px; }
    .week-bars { flex: 1; width: 100%; display: flex; align-items: flex-end; justify-content: center; gap: 4px; }
    .week-bars i { display: block; width: 14px; border-radius: 4px 4px 0 0; min-height: 2px; }
    .week-bars .p { background: var(--ink); }
    .week-bars .f { background: var(--fail); }
    .week-day span { font-family: var(--mono); font-size: .64rem; color: var(--ink-3); }
    .legend { display: flex; gap: 18px; margin-top: 16px; font-size: .78rem; color: var(--ink-2); }
    .legend i { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 7px; }

    /* Stage + verdict */
    .pair { display: grid; grid-template-columns: 1fr 1fr; gap: 22px; }
    .pair figure { margin: 0; }
    .pair figcaption { font-family: var(--mono); font-size: .66rem; letter-spacing: .1em; text-transform: uppercase; color: var(--on-dark-3); margin-bottom: 12px; }
    .frame { position: relative; }
    .frame img { width: 100%; display: block; }
    .frame > i { position: absolute; width: 16px; height: 16px; border: 0 solid var(--lime); pointer-events: none; }
    .frame > i:nth-child(1) { top: -8px; left: -8px; border-top-width: 2px; border-left-width: 2px; }
    .frame > i:nth-child(2) { top: -8px; right: -8px; border-top-width: 2px; border-right-width: 2px; }
    .frame > i:nth-child(3) { bottom: -8px; left: -8px; border-bottom-width: 2px; border-left-width: 2px; }
    .frame > i:nth-child(4) { bottom: -8px; right: -8px; border-bottom-width: 2px; border-right-width: 2px; }

    .verdict { border-radius: 20px; padding: 30px 30px 26px; }
    .verdict.pass { background: var(--lime); color: var(--ink); }
    .verdict.fail { background: var(--fail); color: #fff; }
    .verdict .label { color: currentColor; opacity: .6; }
    .verdict-word { font-family: var(--serif); font-size: 8rem; line-height: .8; letter-spacing: -.04em; margin: 22px 0 18px -4px; }
    .verdict-sub { font-size: .95rem; opacity: .8; }
    .verdict-file { margin-top: 20px; font-family: var(--mono); font-size: .72rem; opacity: .55; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .gauge { margin-top: 34px; padding-top: 22px; position: relative; }
    .gauge-ticks { display: flex; align-items: flex-end; justify-content: space-between; height: 26px; }
    .gauge-ticks i { display: block; width: 2px; height: 14px; background: currentColor; opacity: .22; border-radius: 1px; }
    .gauge-ticks i.on { opacity: 1; height: 22px; }
    .gauge-ticks i.limit { opacity: 1; height: 26px; width: 3px; }
    .gauge-limit-label { position: absolute; top: 0; transform: translateX(-50%); font-family: var(--mono); font-size: .64rem; letter-spacing: .06em; white-space: nowrap; }
    .gauge-scale { display: flex; justify-content: space-between; margin-top: 8px; font-family: var(--mono); font-size: .64rem; opacity: .55; }
    .readings { display: grid; grid-template-columns: repeat(3, 1fr); margin-top: 26px; padding-top: 18px; border-top: 1px solid rgba(128,128,128,.35); }
    .verdict.pass .readings { border-top-color: rgba(14,14,14,.2); }
    .verdict.fail .readings { border-top-color: rgba(255,255,255,.3); }
    .readings b { display: block; font-family: var(--mono); font-size: 1.05rem; font-weight: 500; margin-top: 5px; }
    .readings .label { font-size: .62rem; }

    /* Key / value rows + pills */
    .rows { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 4px 22px; }
    .rows > div { display: flex; justify-content: space-between; gap: 12px; padding: 13px 0; border-bottom: 1px solid var(--line); font-size: .9rem; }
    .rows > div:last-child { border-bottom: 0; }
    .rows span:first-child { color: var(--ink-3); }
    .rows span:last-child { font-weight: 500; }
    .pill { display: inline-flex; align-items: center; gap: 7px; padding: 3px 10px 3px 9px; border-radius: 999px; font-family: var(--mono); font-size: .68rem; font-weight: 500; letter-spacing: .08em; text-transform: uppercase; }
    .pill i { width: 6px; height: 6px; border-radius: 50%; }
    .pill.pass { background: rgba(203,245,92,.5); color: #1d4d12; }
    .pill.pass i { background: var(--pass); }
    .pill.fail { background: rgba(232,71,43,.12); color: #b3321a; }
    .pill.fail i { background: var(--fail); }

    .done { text-align: center; padding: 52px 28px; }
    .check-ring { width: 54px; height: 54px; border-radius: 50%; background: var(--lime); color: var(--ink); display: grid; place-items: center; margin: 0 auto 20px; }
    .big-title { font-family: var(--serif); font-size: 3rem; letter-spacing: -.03em; line-height: 1; }
    .done p { color: var(--on-dark-2); margin: 12px 0 0; }

    @media (max-width: 900px) {
        .hero-grid { grid-template-columns: 1fr; gap: 28px; }
        .hero-side { border-left: 0; border-top: 1px solid var(--dark-line); padding-top: 22px; }
        .hero-stat:first-child { padding-left: 0; }
        .hero-big .big { font-size: 7.2rem; }
        .pair { grid-template-columns: 1fr; }
        h1 { font-size: 2.6rem !important; }
        .verdict-word { font-size: 6rem; }
        .panel-head span { display: none; }
    }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

# Initialize database
db.init_db()


# ----------------- HELPERS -----------------
MARK_SVG = (
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" '
    'stroke-linecap="square"><path d="M3 8V3h5M16 3h5v5M21 16v5h-5M8 21H3v-5"/>'
    '<circle cx="12" cy="12" r="2.4" fill="currentColor" stroke="none"/></svg>'
)
CHECK_SVG = (
    '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" '
    'stroke-linecap="round" stroke-linejoin="round"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>'
)


def render(markup: str):
    """Render an HTML block. Newlines are stripped so the markdown parser never treats it as code."""
    st.markdown(markup.replace("\n", ""), unsafe_allow_html=True)


def page_header(title: str, sub: str = ""):
    st.title(title)
    if sub:
        render(f"<p class='page-sub'>{html.escape(sub)}</p>")


def rows_html(pairs) -> str:
    """Key / value list rendered as quiet hairline rows."""
    body = "".join(f"<div><span>{html.escape(str(k))}</span><span class='mono'>{html.escape(str(v))}</span></div>" for k, v in pairs)
    return f"<div class='rows'>{body}</div>"


def pill_html(result: str) -> str:
    kind = "pass" if result == "PASS" else "fail"
    return f"<span class='pill {kind}'><i></i>{html.escape(result)}</span>"


def data_uri(img: Image.Image, max_side: int = 900) -> str:
    """Encode a PIL image as a compact JPEG data URI for inline HTML."""
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=84)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def stage_html(original: Image.Image, heatmap_path: Path) -> str:
    """Original vs heatmap on a dark panel, framed with viewfinder brackets."""
    figs = [("Original", data_uri(original))]
    if Path(heatmap_path).exists():
        figs.append(("Heatmap", data_uri(Image.open(heatmap_path))))
    items = "".join(
        f"<figure><figcaption>{name}</figcaption><div class='frame'><i></i><i></i><i></i><i></i><img src='{src}' alt='{name}'/></div></figure>"
        for name, src in figs
    )
    return (
        "<div class='panel'><div class='panel-head'><b>Deviation map</b>"
        "<span>Warm areas differ from normal</span></div>"
        f"<div class='pair'>{items}</div></div>"
    )


TICKS = 41


def verdict_html(result: dict, threshold: float, filename: str = "") -> str:
    is_pass = result["result"] == "PASS"
    kind = "pass" if is_pass else "fail"
    clamp = lambda v: min(max(v, 0.0), 1.0)
    limit_idx = round(clamp(threshold) * (TICKS - 1))
    score_idx = round(clamp(result["score"]) * (TICKS - 1))
    ticks = "".join(
        f"<i class='{'on ' if i <= score_idx else ''}{'limit' if i == limit_idx else ''}'></i>" for i in range(TICKS)
    )
    sub = "Meets the quality standard." if is_pass else "Defect detected. Reject this part."
    file_row = f"<div class='verdict-file'>{html.escape(filename)}</div>" if filename else ""
    return (
        f"<div class='verdict {kind}'>"
        "<div class='label'>Verdict</div>"
        f"<div class='verdict-word'>{result['result']}</div>"
        f"<div class='verdict-sub'>{sub}</div>"
        "<div class='gauge'>"
        f"<span class='gauge-limit-label' style='left:{limit_idx / (TICKS - 1) * 100:.1f}%'>LIMIT {threshold:.2f}</span>"
        f"<div class='gauge-ticks'>{ticks}</div>"
        "<div class='gauge-scale'><span>0</span><span>0.5</span><span>1</span></div></div>"
        "<div class='readings'>"
        f"<div><span class='label'>Score</span><b>{result['score'] * 100:.1f}%</b></div>"
        f"<div><span class='label'>Confidence</span><b>{result['confidence']}%</b></div>"
        f"<div><span class='label'>Threshold</span><b>{threshold * 100:.0f}%</b></div>"
        "</div>"
        f"{file_row}</div>"
    )


def strip_html(hourly, current_hour: int) -> str:
    by_hour = {int(r["hour"]): r for r in hourly}
    peak = max([1] + [(r["passed"] or 0) + (r["failed"] or 0) for r in by_hour.values()])
    cols = []
    for h in range(24):
        r = by_hour.get(h)
        p = (r["passed"] or 0) if r else 0
        f = (r["failed"] or 0) if r else 0
        if p + f == 0 or h > current_hour:
            cols.append("<div class='strip-col'><i class='e'></i></div>")
        else:
            fail_bar = f"<i class='f' style='height:{f / peak * 100:.1f}%'></i>" if f else ""
            pass_bar = f"<i class='p' style='height:{p / peak * 100:.1f}%'></i>" if p else ""
            cols.append(f"<div class='strip-col' title='{h:02d}:00 - {p} passed, {f} failed'>{fail_bar}{pass_bar}</div>")
    ticks = "".join(f"<span>{t}</span>" for t in ("00", "06", "12", "18", "24"))
    return f"<div class='strip-bars'>{''.join(cols)}</div><div class='strip-ticks'>{ticks}</div>"


def hero_html(stats: dict, hourly, now: datetime) -> str:
    total, passed, failed = stats["total"], stats["passed"], stats["failed"]
    pass_rate = f"{passed / total * 100:.1f}% pass rate" if total else "no data yet"
    fail_tone = "fail" if failed else ""
    return (
        "<div class='panel hero'>"
        f"<div class='hero-top'><span class='label'>Today</span><span class='label'>{(now.strftime('%a') + ', ' + str(now.day) + ' ' + now.strftime('%b')).upper()}</span></div>"
        "<div class='hero-grid'>"
        f"<div class='hero-big'><div class='label'>Inspected</div><div class='big'>{total}</div><p>units checked so far today</p></div>"
        "<div class='hero-side'>"
        f"<div class='hero-stat'><div class='label'>Passed</div><div class='figure pass'>{passed}</div><small>{pass_rate}</small></div>"
        f"<div class='hero-stat'><div class='label'>Failed</div><div class='figure {fail_tone}'>{failed}</div><small>defects found</small></div>"
        f"<div class='hero-stat'><div class='label'>Rejection</div><div class='figure'>{stats['rejection_rate']}%</div><small>of today's units</small></div>"
        "</div></div>"
        "<div class='strip'><div class='label' style='margin-bottom:14px'>Activity by hour</div>"
        f"{strip_html(hourly, now.hour)}</div></div>"
    )


def week_html(weekly, today: datetime) -> str:
    by_day = {r["day"]: r for r in weekly}
    days = [(today - timedelta(days=i)) for i in range(6, -1, -1)]
    peak = max([1] + [max(r["passed"] or 0, r["failed"] or 0) for r in by_day.values()])
    cols = []
    for d in days:
        r = by_day.get(d.strftime("%Y-%m-%d"), {"passed": 0, "failed": 0})
        p, f = r["passed"] or 0, r["failed"] or 0
        cols.append(
            "<div class='week-day'><div class='week-bars'>"
            f"<i class='p' style='height:{p / peak * 100:.1f}%' title='{p} passed'></i>"
            f"<i class='f' style='height:{f / peak * 100:.1f}%' title='{f} failed'></i>"
            f"</div><span>{d.strftime('%a')} {d.day}</span></div>"
        )
    legend = "<div class='legend'><span><i style='background:var(--ink)'></i>Passed</span><span><i style='background:var(--fail)'></i>Failed</span></div>"
    return f"<div class='week'>{''.join(cols)}</div>{legend}"


# Navigation labels (compared below)
NAV_OVERVIEW, NAV_INSPECT, NAV_TRAIN, NAV_HISTORY, NAV_SETTINGS = "Overview", "Inspect", "Train", "History", "Settings"

# ----------------- SIDEBAR -----------------
st.sidebar.markdown(
    f"<div class='brand'><span class='brand-mark'>{MARK_SVG}</span><span class='brand-name'>VisionQC</span></div>",
    unsafe_allow_html=True,
)

nav_choice = st.sidebar.radio(
    "Navigation",
    [NAV_OVERVIEW, NAV_INSPECT, NAV_TRAIN, NAV_HISTORY, NAV_SETTINGS],
    index=1,
    label_visibility="collapsed",
)

st.sidebar.divider()

product_name = db.get_config("product_name") or "Product"

# Model status (a slot, so training can flip it to READY without a page change)
is_model_ready = model.is_trained()
chip_slot = st.sidebar.empty()


def show_model_chip(ready: bool, name: str):
    if ready:
        chip_slot.markdown(
            f"<div class='model-chip'><span class='dot on'></span><span>READY &middot; {html.escape(name)}</span></div>",
            unsafe_allow_html=True,
        )
    else:
        chip_slot.markdown("<div class='model-chip'><span class='dot'></span><span>NO MODEL</span></div>", unsafe_allow_html=True)


show_model_chip(is_model_ready, product_name)

# Threshold Slider
current_threshold = float(db.get_config("threshold") or "0.5")
threshold_val = st.sidebar.slider(
    "Detection threshold",
    min_value=0.0,
    max_value=1.0,
    value=current_threshold,
    step=0.01,
    help="Higher threshold = more lenient (fewer fails). Lower = stricter (more fails).",
)

if threshold_val != current_threshold:
    db.set_config("threshold", str(threshold_val))
    st.sidebar.caption(f"Threshold updated to {threshold_val:.2f}")


# =====================================================================
# 1. OVERVIEW
# =====================================================================
if nav_choice == NAV_OVERVIEW:
    page_header("Overview", "Today's inspections at a glance.")
    now = datetime.now()
    stats = db.get_today_stats()

    render(hero_html(stats, db.get_hourly_stats(), now))
    st.write("")

    c1, c2 = st.columns([1.6, 1], gap="large")
    with c1:
        st.subheader("Last 7 days")
        render(f"<div class='card' style='padding:22px'>{week_html(db.get_weekly_stats(), now)}</div>")
    with c2:
        st.subheader("Active model")
        st.markdown(
            rows_html([
                ("Backbone", model.BACKBONE),
                ("Image resolution", f"{model.IMAGE_SIZE[0]}x{model.IMAGE_SIZE[1]}"),
                ("Product", product_name),
                ("Threshold", f"{threshold_val:.2f}"),
                ("Avg confidence", f"{stats['avg_confidence']}%" if stats["total"] else "-"),
            ]),
            unsafe_allow_html=True,
        )


# =====================================================================
# 2. INSPECT
# =====================================================================
elif nav_choice == NAV_INSPECT:
    page_header("Inspect", "Check a part against the learned normal.")

    if not is_model_ready:
        st.warning("No trained model yet. Open the Train page and add photos of good parts first.")
    else:
        sample_dir = ROOT_DIR / "images"
        sample_images = sorted(list(sample_dir.glob("*.png"))) if sample_dir.exists() else []

        tab_cam, tab_file, tab_samples = st.tabs(["Camera", "Upload", "Samples"])
        image_to_inspect = None
        source_name = "upload.jpg"

        with tab_cam:
            camera_img = st.camera_input("Point the camera at the part and capture")
            if camera_img is not None:
                image_to_inspect = Image.open(camera_img)
                source_name = "webcam_capture.jpg"

        with tab_file:
            uploaded_file = st.file_uploader("Upload a product photo", type=["jpg", "jpeg", "png", "bmp", "webp"])
            if uploaded_file is not None:
                image_to_inspect = Image.open(uploaded_file)
                source_name = uploaded_file.name

        with tab_samples:
            if sample_images:
                st.caption(f"{len(sample_images)} bundled sample screw photos.")
                selected_sample = st.selectbox(
                    "Sample photo",
                    sample_images,
                    format_func=lambda p: p.name
                )
                if selected_sample and st.button("Inspect selected sample"):
                    image_to_inspect = Image.open(selected_sample)
                    source_name = selected_sample.name
            else:
                st.info("No sample images found in the images folder.")

        if image_to_inspect is not None:
            st.write("")
            with st.spinner("Analysing image..."):
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    image_to_inspect.convert("RGB").save(tmp.name, "JPEG", quality=95)
                    temp_img_path = Path(tmp.name)

                try:
                    result = model.inspect(temp_img_path, threshold=threshold_val)

                    # Save inspection record to SQLite
                    db.log_inspection(
                        image_path=str(temp_img_path),
                        heatmap_path=result["heatmap_path"],
                        score=result["score"],
                        confidence=result["confidence"],
                        result=result["result"],
                        threshold=threshold_val,
                        filename=source_name,
                    )
                except Exception as e:
                    result = None
                    st.error(f"Inference error: {e}")

            if result is not None:
                col_main, col_side = st.columns([1.7, 1], gap="large")
                with col_main:
                    render(stage_html(image_to_inspect, Path(result["heatmap_path"])))
                with col_side:
                    render(verdict_html(result, threshold_val, source_name))


# =====================================================================
# 3. TRAIN
# =====================================================================
elif nav_choice == NAV_TRAIN:
    page_header("Train", "Show VisionQC what a good part looks like. No defect photos needed.")
    st.markdown("""
    PatchCore learns what a **normal, defect-free** product looks like from **20–30 good photos**.
    Anything that deviates from it is flagged with a heatmap.
    """)

    product_input = st.text_input("Product name", value=product_name)

    sample_dir = ROOT_DIR / "images"
    has_sample_dir = sample_dir.exists() and len(list(sample_dir.glob("*.png"))) >= 5

    MODE_SERVER = "Sample folder (instant)"
    MODE_ZIP    = "Upload a .zip folder"
    MODE_DROP   = "Upload individual photos"

    train_mode = st.radio(
        "Training photos",
        [MODE_SERVER, MODE_ZIP, MODE_DROP] if has_sample_dir else [MODE_ZIP, MODE_DROP],
        index=0,
        horizontal=True,
    )

    uploaded_zip = None
    uploaded_train_files = None
    sample_count = 20

    if train_mode == MODE_SERVER:
        st.caption("Photos are read straight from the server, so there is nothing to upload.")
        sample_count = st.slider("How many sample photos to train on", 5, min(50, len(list(sample_dir.glob("*.png")))), 20)
        st.caption(f"VisionQC will learn from {sample_count} photos in `{sample_dir.name}/`.")

    elif train_mode == MODE_ZIP:
        st.caption("Zip a folder of 15–30 good photos and drop it here. A single zip uploads much faster than many files.")
        uploaded_zip = st.file_uploader(
            "Zipped folder of good product photos",
            type=["zip"],
            key="zip_uploader"
        )
        if uploaded_zip:
            st.success(f"Loaded `{uploaded_zip.name}` ({uploaded_zip.size / 1024:.1f} KB)")

    elif train_mode == MODE_DROP:
        uploaded_train_files = st.file_uploader(
            "10–25 photos of good parts (no defects)",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            key="multi_file_uploader"
        )
        if uploaded_train_files:
            st.info(f"{len(uploaded_train_files)} photos selected.")

    if st.button("Learn normal", type="primary"):
        temp_dir = Path(tempfile.mkdtemp())
        try:
            # 1. Populate temp_dir based on chosen mode
            if train_mode == MODE_SERVER:
                samples = sorted(list(sample_dir.glob("*.png")))[:sample_count]
                for s in samples:
                    shutil.copy(s, temp_dir / s.name)

            elif train_mode == MODE_ZIP:
                if not uploaded_zip:
                    st.error("Please upload a .zip file containing your product photos first.")
                    st.stop()
                with zipfile.ZipFile(uploaded_zip, "r") as z:
                    for filename in z.namelist():
                        ext = Path(filename).suffix.lower()
                        if ext in model.IMAGE_EXTS and not Path(filename).name.startswith("."):
                            # Extract clean basename to temp_dir
                            target = temp_dir / Path(filename).name
                            with z.open(filename) as src, open(target, "wb") as dst:
                                dst.write(src.read())

            elif train_mode == MODE_DROP:
                if not uploaded_train_files or len(uploaded_train_files) < 5:
                    st.error("Please select at least 5 images (15–20 recommended) to train.")
                    st.stop()
                for uf in uploaded_train_files:
                    img = Image.open(uf)
                    img.convert("RGB").save(temp_dir / uf.name)

            progress_bar = st.progress(0)
            status_text = st.empty()

            def on_progress(pct: int, msg: str):
                progress_bar.progress(min(pct, 100))
                status_text.caption(f"{pct}%  ·  {msg}")

            with st.spinner("Learning what normal looks like..."):
                train_result = model.train(temp_dir, product_name=product_input, progress=on_progress)
                db.set_config("model_trained", "true")
                db.set_config("product_name", product_input)
                db.set_config("train_image_count", str(train_result["image_count"]))
                db.set_config("trained_at", datetime.now().isoformat())

            progress_bar.empty()
            status_text.empty()
            show_model_chip(True, product_input)
            render(
                "<div class='panel done'>"
                f"<div class='check-ring'>{CHECK_SVG}</div>"
                "<div class='big-title'>Model ready</div>"
                f"<p>Trained on {train_result['image_count']} images using {html.escape(model.BACKBONE)}. Open Inspect to try it.</p>"
                "</div>"
            )
            with st.expander("Calibration details"):
                st.json(train_result["calibration"])

        except Exception as err:
            st.error(f"Training failed: {err}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


# =====================================================================
# 4. HISTORY
# =====================================================================
elif nav_choice == NAV_HISTORY:
    page_header("History", "Every inspection, newest first.")

    filter_choice = st.radio("Filter", ["ALL", "PASS", "FAIL"], horizontal=True, label_visibility="collapsed")
    result_filter = None if filter_choice == "ALL" else filter_choice

    history = db.get_history(limit=100, result_filter=result_filter)

    if not history:
        st.info("No inspection records found.")
    else:
        st.caption(f"Showing the last {len(history)} inspections.")

        col_left, col_right = st.columns([2, 1], gap="large")

        with col_left:
            table_data = []
            for h in history:
                table_data.append({
                    "ID": h["id"],
                    "Time": h["timestamp"][:19].replace("T", " "),
                    "Result": h["result"],
                    "Score": round(h["anomaly_score"], 3),
                    "Confidence": f"{h['confidence']}%",
                    "Threshold": round(h["threshold"], 2),
                    "File": h["filename"] or "-",
                })
            st.dataframe(
                table_data,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Score": st.column_config.NumberColumn(format="%.3f"),
                    "Threshold": st.column_config.NumberColumn(format="%.2f"),
                },
            )

        with col_right:
            st.subheader("Details")
            selected_id = st.selectbox("Inspection ID", [h["id"] for h in history])
            selected_row = next((h for h in history if h["id"] == selected_id), None)
            if selected_row:
                render(
                    "<div class='rows'>"
                    f"<div><span>Result</span><span>{pill_html(selected_row['result'])}</span></div>"
                    f"<div><span>Score</span><span class='mono'>{selected_row['anomaly_score']:.4f}</span></div>"
                    f"<div><span>Confidence</span><span class='mono'>{selected_row['confidence']}%</span></div>"
                    "</div>"
                )
                if selected_row["heatmap_path"] and Path(selected_row["heatmap_path"]).exists():
                    st.write("")
                    st.image(selected_row["heatmap_path"], caption=f"Heatmap #{selected_row['id']}", use_container_width=True)

        st.divider()
        if st.button("Clear history"):
            db.clear_history()
            st.success("History cleared.")
            st.rerun()


# =====================================================================
# 5. SETTINGS
# =====================================================================
elif nav_choice == NAV_SETTINGS:
    page_header("Settings", "Model configuration and reset.")

    st.subheader("Model")
    st.markdown(
        rows_html([
            ("Backbone", model.BACKBONE),
            ("Coreset subsampling ratio", model.CORESET_RATIO),
            ("Batch size", model.BATCH_SIZE),
            ("Model trained", db.get_config("model_trained")),
            ("Product label", db.get_config("product_name")),
            ("Trained at", db.get_config("trained_at") or "Never"),
        ]),
        unsafe_allow_html=True,
    )

    st.write("")
    st.subheader("Reset model")
    st.caption("Clearing the model removes the trained memory bank. You will need to train again.")
    if st.button("Reset trained model"):
        model.reset_model()
        db.set_config("model_trained", "false")
        st.warning("Model reset. Please train the model again.")
        st.rerun()
