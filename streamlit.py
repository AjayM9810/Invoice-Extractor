import os, tempfile
import inspect
import json
import hashlib
import base64
from datetime import datetime
from pathlib import Path
import re
import streamlit as st
import pandas as pd
from ocrimg import (
    process_invoice_dir_img,
    compute_subtotal as img_compute_subtotal,
    compute_invoice_total as img_compute_invoice_total,
    compute_line_total as img_compute_line_total,
    extract_metadata as img_extract_metadata,
)
from ocrpdf import (
    process_invoice_dir_pdf,
    compute_subtotal as pdf_compute_subtotal,
    compute_invoice_total as pdf_compute_invoice_total,
    compute_line_total as pdf_compute_line_total,
    extract_metadata as pdf_extract_metadata,
)

st.set_page_config(page_title="Invoice Data Extractor", layout="wide")

st.markdown(
    """
<style>
:root {
  --bg: #0a0a0a;
  --panel: #111214;
  --text: #f2f2f2;
  --muted: #9aa0a6;
  --label: #8fd3ff;
  --label-strong: #ff9f1c;
  --value: #a9a9a9;
  --value-strong: #ffd166;
  --ok: #27c93f;
  --bad: #ff5f56;
  --warn: #ffbd2e;
  --manual: #ff1744;
  --info: #4aa3ff;
  --match: #4aa3ff;
  --table-bg: #0f1012;
  --table-border: #1f2328;
  --table-head: #15171a;
}
html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg);
  color: var(--text);
  font-family: "Space Grotesk", "Sora", "Manrope", "IBM Plex Sans", sans-serif;
}
[data-testid="stHeader"] { background: var(--bg) !important; }
[data-testid="stSidebar"] { background: #0b0c0e !important; }
[data-testid="stSidebarNav"] { background: #0b0c0e !important; }
[data-testid="stSidebar"] * {
  color: var(--text) !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] label span {
  color: #e2e8f0 !important;
}
.block-container { padding-top: 1.5rem; }

.inv-card {
  background: var(--panel);
  border: 1px solid #1a1c1f;
  border-radius: 16px;
  padding: 18px 20px;
  margin: 16px 0 22px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.35);
}
.inv-topline {
  font-size: 18px;
  font-weight: 700;
  letter-spacing: 0.2px;
}
.inv-row {
  display: flex;
  gap: 18px;
  flex-wrap: wrap;
  margin-top: 6px;
}
.inv-row .kv {
  min-width: 220px;
}
.kv .k { color: var(--label); font-weight: 600; }
.kv .v { color: var(--value); margin-left: 6px; }
.kv .k-strong { color: var(--label-strong); font-weight: 700; }
.kv .v-strong { color: var(--value-strong); margin-left: 6px; }

.section-title {
  font-size: 20px;
  font-weight: 700;
  color: var(--text);
  margin-bottom: 8px;
  text-transform: uppercase;
  letter-spacing: 0.6px;
}
.totals-grid, .status-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 6px;
}
.totals-line, .status-line {
  display: flex;
  justify-content: space-between;
  gap: 10px;
}
.totals-line .k { color: var(--label); }
.totals-line .k-strong { color: var(--label-strong); font-weight: 700; }
.totals-line .v { color: var(--value); }
.totals-line .v-strong { color: var(--value-strong); }

.badge { font-weight: 700; }
.badge.ok { color: var(--ok); }
.badge.bad { color: var(--bad); }
.badge.warn { color: var(--warn); }
.badge.manual { color: var(--manual); }
.badge.info { color: var(--info); }
.k-blue { color: var(--label); font-weight: 700; }
.k-white { color: var(--text); font-weight: 700; }

.hero {
  background: radial-gradient(1200px 400px at 10% -20%, #1a2a3a 0%, transparent 60%), #101113;
  border: 1px solid #1a1c1f;
  border-radius: 18px;
  padding: 26px 28px;
  margin: 16px 0 22px;
}
.hero-title {
  font-size: 32px;
  font-weight: 800;
  letter-spacing: 0.3px;
}
.hero-sub {
  color: var(--muted);
  margin-top: 6px;
  font-size: 15px;
}
.hero-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 10px;
  margin-top: 16px;
}
.hero-pill {
  background: #13161a;
  border: 1px solid #1f2328;
  border-radius: 12px;
  padding: 10px 12px;
  color: var(--text);
  font-size: 13px;
  text-transform: uppercase;
  letter-spacing: 0.8px;
}
.card-header {
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 1.2px;
  text-transform: uppercase;
  color: var(--label);
  margin-bottom: 10px;
}
div[data-testid="stForm"] {
  background: #0f1012;
  border: 1px solid #1f2328;
  border-radius: 14px;
  padding: 16px;
  margin-bottom: 20px;
}

/* Primary buttons */
.stButton > button,
.stFormSubmitButton > button,
button[kind="primary"],
button[kind="secondary"] {
  background: #2d7ff9 !important;
  color: #000 !important;
  border: 1px solid #1b5fcc !important;
  box-shadow: none !important;
}
.stButton > button:hover,
.stFormSubmitButton > button:hover,
button[kind="primary"]:hover,
button[kind="secondary"]:hover {
  background: #3a8bff !important;
  color: #000 !important;
}
.stButton > button:focus,
.stFormSubmitButton > button:focus,
button[kind="primary"]:focus,
button[kind="secondary"]:focus {
  outline: none !important;
  box-shadow: none !important;
}

/* Form labels */
div[data-testid="stForm"] label,
div[data-testid="stForm"] label span,
[data-testid="stTextInput"] label,
[data-testid="stTextInput"] label span,
[data-testid="stSelectbox"] label,
[data-testid="stSelectbox"] label span {
  color: #c3c9d1 !important;
}
body[data-theme="light"] div[data-testid="stForm"] label,
body[data-theme="light"] div[data-testid="stForm"] label span,
body[data-theme="light"] [data-testid="stTextInput"] label,
body[data-theme="light"] [data-testid="stTextInput"] label span,
body[data-theme="light"] [data-testid="stSelectbox"] label,
body[data-theme="light"] [data-testid="stSelectbox"] label span {
  color: #4b5563 !important;
}

.table-container table {
  background: var(--table-bg) !important;
  color: var(--text) !important;
  border: 1px solid var(--table-border) !important;
  border-collapse: collapse !important;
}
.table-container thead tr th {
  background: var(--table-head) !important;
  color: var(--text) !important;
  border-bottom: 1px solid var(--table-border) !important;
}
.table-container tbody tr td {
  border-top: 1px solid var(--table-border) !important;
  color: var(--text) !important;
}


/* File uploader visibility */
[data-testid="stFileUploader"] * {
  color: var(--text) !important;
}
[data-testid="stFileUploader"] button {
  background: #2d7ff9 !important;
  color: #000 !important;
  padding: 4px 10px !important;
  font-size: 0.85rem !important;
  border: none !important;
  outline: none !important;
  box-shadow: none !important;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
  border: 1px dashed #2a2f36 !important;
  background: #0f1012 !important;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
  background: #15171a !important;
  border: 1px solid #23262b !important;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] * {
  color: var(--text) !important;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] button {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  color: var(--text) !important;
  outline: none !important;
}

/* Dataframe toolbar + search + pagination */
[data-testid="stDataFrame"] {
  background: #ffffff !important;
  border: 1px solid #c0c0c0 !important;
  border-radius: 6px !important;
}
[data-testid="stDataFrame"] [role="grid"] {
  background: #ffffff !important;
}
[data-testid="stDataFrame"] [role="columnheader"] {
  background: #f4f4f4 !important;
  color: #000 !important;
  border-bottom: 1px solid #c0c0c0 !important;
}
[data-testid="stDataFrame"] [role="columnheader"] * {
  color: #000 !important;
}
[data-testid="stDataFrame"] [data-testid="stToolbar"] {
  background: #e6e6e6 !important;
  border: 1px solid #c0c0c0 !important;
}
[data-testid="stDataFrame"] [data-testid="stToolbar"] * {
  color: #000 !important;
}
[data-testid="stDataFrame"] button {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  color: #000 !important;
  outline: none !important;
}
[data-testid="stDataFrame"] button:focus,
[data-testid="stDataFrame"] button:active {
  outline: none !important;
  box-shadow: none !important;
}
[data-testid="stDataFrame"] button:hover {
  background: transparent !important;
  box-shadow: none !important;
}
[data-baseweb="menu"] {
  background: #f2f2f2 !important;
  color: #000 !important;
  border: 1px solid #c0c0c0 !important;
  z-index: 10000 !important;
}
[data-baseweb="menu"] * {
  color: #000 !important;
}
[data-baseweb="menu"] svg {
  fill: #000 !important;
}
[data-baseweb="popover"] {
  z-index: 10000 !important;
}
[data-baseweb="popover"] {
  background: #f2f2f2 !important;
}
[data-testid="stDataFrame"] input,
[data-testid="stDataFrame"] textarea,
[data-testid="stDataFrame"] select {
  background: #f2f2f2 !important;
  color: #000 !important;
  border: 1px solid #c0c0c0 !important;
}
[data-testid="stDataFrame"] svg {
  fill: #000 !important;
  stroke: #000 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

APP_ROOT = Path(__file__).resolve().parent
USER_DB_PATH = APP_ROOT / "users.json"
LIBRARY_ROOT = APP_ROOT / "user_library"
USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{3,20}$")
INVOICE_EXTS = {".pdf", ".png", ".jpg", ".jpeg"}


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def normalize_username(username: str) -> str:
    return (username or "").strip().lower()


def load_users() -> dict:
    if USER_DB_PATH.exists():
        try:
            with USER_DB_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
            raw_users = data.get("users", {})
            if raw_users:
                users = {}
                for name, info in raw_users.items():
                    users[normalize_username(name)] = info
                return users
        except Exception:
            st.warning("Could not read users.json. Using default credentials.")
    return {
        "admin": {
            "password_hash": hash_password("admin123"),
            "display_name": "Admin",
        }
    }

def save_users(users: dict) -> bool:
    try:
        USER_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = USER_DB_PATH.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump({"users": users}, f, indent=2)
        os.replace(tmp_path, USER_DB_PATH)
        return True
    except Exception as exc:
        st.error(f"Could not save users.json: {exc}")
        return False


def verify_login(username: str, password: str):
    users = load_users()
    username_norm = normalize_username(username)
    user = users.get(username_norm)
    if not user:
        return False, None, None
    if hash_password(password) == user.get("password_hash"):
        display = user.get("display_name") or username_norm
        return True, display, username_norm
    return False, None, None


def register_user(username: str, password: str, display_name: str):
    username_norm = normalize_username(username)
    if not USERNAME_PATTERN.match(username_norm):
        return False, "Username must be 3-20 chars and only letters, numbers, _ or -.", None, None
    if len(password or "") < 6:
        return False, "Password must be at least 6 characters.", None, None
    users = load_users()
    if username_norm in users:
        return False, "That username already exists.", None, None
    display = (display_name or "").strip() or username_norm
    users[username_norm] = {
        "password_hash": hash_password(password),
        "display_name": display,
    }
    if not save_users(users):
        return False, "Could not save user database.", None, None
    ensure_user_library(username_norm)
    return True, "Account created.", username_norm, display


def sanitize_username(username: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", normalize_username(username))
    return safe or "user"


def ensure_user_library(username: str) -> Path:
    user_dir = LIBRARY_ROOT / sanitize_username(username)
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir


def save_upload_to_library(uploaded_file, user_dir: Path, content: bytes) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_hash = hashlib.sha256(content).hexdigest()[:8]
    base = Path(uploaded_file.name).name
    dest = user_dir / f"{ts}_{short_hash}_{base}"
    counter = 1
    while dest.exists():
        dest = user_dir / f"{ts}_{short_hash}_{counter}_{base}"
        counter += 1
    with open(dest, "wb") as out:
        out.write(content)
    return dest


def extracted_paths(invoice_path: Path):
    stem = invoice_path.stem
    json_path = invoice_path.with_name(f"{stem}.extracted.json")
    csv_path = invoice_path.with_name(f"{stem}.line_items.csv")
    return json_path, csv_path


def save_extracted_data(invoice_path: Path, prepared: dict) -> None:
    json_path, csv_path = extracted_paths(invoice_path)
    try:
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(prepared, f, indent=2)
        line_items = prepared.get("line_items") or []
        if line_items:
            pd.DataFrame(line_items).to_csv(csv_path, index=False)
    except Exception as exc:
        st.warning(f"Could not save extracted data for {invoice_path.name}: {exc}")


def load_extracted_data(invoice_path: Path):
    json_path, csv_path = extracted_paths(invoice_path)
    data = None
    if json_path.exists():
        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            st.error(f"Could not read {json_path.name}: {exc}")
    return data, json_path, csv_path


def render_invoice_file(invoice_path: Path):
    data = invoice_path.read_bytes()
    ext = invoice_path.suffix.lower()
    mime = "application/pdf" if ext == ".pdf" else "image/png"
    st.download_button("Download Invoice", data, file_name=invoice_path.name, mime=mime)
    if ext == ".pdf":
        b64 = base64.b64encode(data).decode("utf-8")
        st.markdown(
            f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="720" type="application/pdf"></iframe>',
            unsafe_allow_html=True,
        )
    else:
        st.image(data, caption=invoice_path.name, use_column_width=True)


def init_session_state():
    st.session_state.setdefault("authenticated", False)
    st.session_state.setdefault("username", None)
    st.session_state.setdefault("display_name", None)
    st.session_state.setdefault("upload_hashes", set())
    st.session_state.setdefault("last_reports", [])
    st.session_state.setdefault("selected_invoice", None)
    st.session_state.setdefault("nav", "Extractor")


def do_logout():
    for key in ["authenticated", "username", "display_name", "upload_hashes", "last_reports", "selected_invoice", "nav"]:
        st.session_state.pop(key, None)
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


# Normalize raw OCR output into the fields the UI expects.
def prepare_invoice(inv: dict) -> dict:
    if not inv or "error" in inv:
        return inv

    is_pdf = "meta" in inv
    if is_pdf:
        meta = inv.get("meta") or pdf_extract_metadata(
            inv.get("raw_text", ""),
            inv.get("file", "UNKNOWN"),
            inv.get("words", []),
        )
        compute_subtotal = pdf_compute_subtotal
        compute_invoice_total = pdf_compute_invoice_total
        compute_line_total = pdf_compute_line_total
    else:
        header = inv.get("header", {})
        meta = img_extract_metadata(
            inv.get("raw_text", ""),
            inv.get("file", "UNKNOWN"),
            inv.get("words", []),
            header,
        )
        compute_subtotal = img_compute_subtotal
        compute_invoice_total = img_compute_invoice_total
        compute_line_total = img_compute_line_total

    line_items = inv.get("line_items", []) or []
    totals = inv.get("totals", {}) or {}

    subtotal_computed = compute_subtotal(line_items)
    invoice_total_computed = compute_invoice_total(totals, line_items)

    def fmt(v):
        return "N/A" if v is None or v == "" else v

    def pct_diff(a, b):
        if a is None or b is None:
            return None
        try:
            denom = max(abs(float(a)), abs(float(b)), 1e-6)
            return abs(float(a) - float(b)) / denom
        except Exception:
            return None

    def status_from_diff(d):
        if d is None:
            return "N/A"
        if d <= 0.005:
            return "PASS"
        if d <= 0.05:
            return "MINOR MISMATCH"
        return "FAIL"

    subtotal_diff = pct_diff(totals.get("subtotal"), subtotal_computed)
    invoice_diff = pct_diff(totals.get("invoice_total"), invoice_total_computed)

    subtotal_status = status_from_diff(subtotal_diff)
    invoice_status = status_from_diff(invoice_diff)
    overall_status = "PASS"
    if subtotal_status == "FAIL" or invoice_status == "FAIL":
        overall_status = "FAIL"
    elif subtotal_status == "MINOR MISMATCH" or invoice_status == "MINOR MISMATCH":
        overall_status = "MINOR MISMATCH"

    manual_review = "YES" if (invoice_diff is not None and invoice_diff > 0.05) else "NO"

    enriched_items = []
    for li in line_items:
        computed = compute_line_total(li)
        line_diff = pct_diff(li.get("total_price"), computed)
        line_status = "MATCH" if line_diff is not None and line_diff <= 0.005 else ("MISMATCH" if line_diff is not None else "N/A")
        li_out = dict(li)
        li_out["computed"] = computed
        li_out["status"] = line_status
        enriched_items.append(li_out)

    totals_view = dict(totals)
    totals_view["subtotal_computed"] = fmt(subtotal_computed)
    totals_view["invoice_computed"] = fmt(invoice_total_computed)

    header = inv.get("header", {})

    return {
        "file": inv.get("file", "UNKNOWN"),
        "company": meta.get("company", "N/A"),
        "invoice_number": meta.get("invoice_number", "N/A"),
        "invoice_date": meta.get("invoice_date", "N/A"),
        "currency": meta.get("currency", inv.get("currency", "N/A")),
        "language": meta.get("language", inv.get("language", "N/A")),
        "ship_to": meta.get("ship_to", "N/A"),
        "vat": header.get("vat", "N/A"),
        "totals": totals_view,
        "status": {
            "subtotal": subtotal_status,
            "invoice_total": invoice_status,
            "overall": overall_status,
            "manual_review": manual_review,
        },
        "line_items": enriched_items,
    }

# Renderer for structured output

# Renderer for structured output
def render_invoice_report(invoice):
    def badge_class(val: str, field: str = "") -> str:
        if not val:
            return ""
        v = str(val).upper()
        if field == "manual_review":
            if v == "YES":
                return "manual"
            if v == "NO":
                return "info"
        if v == "FAIL":
            return "warn"
        if v in {"PASS", "MATCH"}:
            return "ok"
        if v in {"MISMATCH"}:
            return "bad"
        if v in {"MINOR MISMATCH"}:
            return "warn"
        return ""

    st.markdown(
        f"""
<div class=\"inv-card\">
  <div class=\"inv-topline\">Invoice No: {invoice.get('invoice_number','N/A')}</div>
  <div class=\"inv-row\">
    <div class=\"kv\"><span class=\"k\">Company</span><span class=\"v\">: {invoice.get('company','N/A')}</span></div>
    <div class=\"kv\"><span class=\"k\">Date</span><span class=\"v\">: {invoice.get('invoice_date','N/A')}</span></div>
  </div>
  <div class=\"inv-row\">
    <div class=\"kv\"><span class=\"k\">Currency</span><span class=\"v\">: {invoice.get('currency','N/A')}</span></div>
    <div class=\"kv\"><span class=\"k\">Language</span><span class=\"v\">: {invoice.get('language','N/A')}</span></div>
  </div>
  <div class=\"inv-row\">
    <div class=\"kv\"><span class=\"k\">Ship To</span><span class=\"v\">: {invoice.get('ship_to','N/A')}</span></div>
    <div class=\"kv\"><span class=\"k\">VAT No</span><span class=\"v\">: {invoice.get('vat','N/A')}</span></div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    totals = invoice.get("totals", {})
    status = invoice.get("status", {})

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(
            f"""
<div class=\"section-title\">TOTALS</div>
<div class=\"totals-grid\">
  <div class=\"totals-line\"><span class=\"k\">Subtotal</span><span class=\"v\">{totals.get('subtotal','N/A')}</span></div>
  <div class=\"totals-line\"><span class=\"k-strong\">Subtotal(C)</span><span class=\"v-strong\">{totals.get('subtotal_computed','N/A')}</span></div>
  <div class=\"totals-line\"><span class=\"k\">Discount</span><span class=\"v\">{totals.get('discount','N/A')}</span></div>
  <div class=\"totals-line\"><span class=\"k\">Shipping</span><span class=\"v\">{totals.get('shipping','N/A')}</span></div>
  <div class=\"totals-line\"><span class=\"k\">Tax/VAT</span><span class=\"v\">{totals.get('tax','N/A')}</span></div>
  <div class=\"totals-line\"><span class=\"k\">Amount Due</span><span class=\"v\">{totals.get('amount_due','N/A')}</span></div>
  <div class=\"totals-line\"><span class=\"k\">Invoice Total</span><span class=\"v\">{totals.get('invoice_total','N/A')}</span></div>
  <div class=\"totals-line\"><span class=\"k-strong\">Invoice(C)</span><span class=\"v-strong\">{totals.get('invoice_computed','N/A')}</span></div>
</div>
""",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f"""
<div class=\"section-title\">STATUS</div>
<div class=\"status-grid\">
  <div class=\"status-line\"><span class=\"k-blue\">Subtotal</span><span class=\"badge {badge_class(status.get('subtotal'), 'subtotal')}\">{status.get('subtotal','N/A')}</span></div>
  <div class=\"status-line\"><span class=\"k-blue\">Invoice Total</span><span class=\"badge {badge_class(status.get('invoice_total'), 'invoice_total')}\">{status.get('invoice_total','N/A')}</span></div>
  <div class=\"status-line\"><span class=\"k-blue\">Overall</span><span class=\"badge {badge_class(status.get('overall'), 'overall')}\">{status.get('overall','N/A')}</span></div>
  <div class=\"status-line\"><span class=\"k-white\">Manual Review</span><span class=\"badge {badge_class(status.get('manual_review'), 'manual_review')}\">{status.get('manual_review','N/A')}</span></div>
</div>
""",
            unsafe_allow_html=True,
        )

    line_items = invoice.get("line_items", [])
    if line_items:
        st.markdown(f"#### Line Items ({len(line_items)})")
        df = pd.DataFrame(line_items)
        cols = ["description", "quantity", "unit_price", "total_price", "computed", "status"]
        df = df[[c for c in cols if c in df.columns]].reset_index(drop=True)
        df.index = range(1, len(df) + 1)
        df.index.name = "NO"
        rename_map = {c: c.replace("_", " ").upper() for c in df.columns}
        df = df.rename(columns=rename_map)

        table_key = hashlib.sha256(
            f"{invoice.get('file','unknown')}_{len(df)}".encode("utf-8")
        ).hexdigest()[:8]

        tcol1, _ = st.columns([3, 1], gap="small")
        search = tcol1.text_input(
            "Search line items",
            key=f"li_search_{table_key}",
            placeholder="Search description...",
        )
        if search:
            if "DESCRIPTION" in df.columns:
                df = df[df["DESCRIPTION"].astype(str).str.contains(search, case=False, na=False)]
            else:
                mask = df.astype(str).apply(lambda s: s.str.contains(search, case=False, na=False)).any(axis=1)
                df = df[mask]

        def style_status(val):
            v = str(val).upper()
            if v in {"PASS"}:
                return "color: #27c93f; font-weight: 700;"
            if v in {"MATCH"}:
                return "color: #27c93f; font-weight: 700;"
            if v in {"FAIL"}:
                return "color: #ffb74d; font-weight: 700;"
            if v in {"MISMATCH"}:
                return "color: #e53935; font-weight: 700;"
            if v in {"MINOR MISMATCH"}:
                return "color: #ffb74d; font-weight: 700;"
            return ""

        styler = df.style
        if "STATUS" in df.columns:
            styler = styler.applymap(style_status, subset=["STATUS"])

        st.markdown('<div class="table-container">', unsafe_allow_html=True)
        df_args = dict(use_container_width=True, hide_index=False)
        sig = inspect.signature(st.dataframe)
        if "on_select" in sig.parameters:
            st.dataframe(styler, on_select="ignore", **df_args)
        else:
            st.dataframe(styler, **df_args)
        st.markdown("</div>", unsafe_allow_html=True)

def render_homepage():
    st.markdown(
        """
<div class="hero">
  <div class="hero-title">Invoice Data Extractor</div>
  <div class="hero-sub">Securely upload invoices, extract line items, and audit totals.</div>
  <div class="hero-grid">
    <div class="hero-pill">PDF and image support</div>
    <div class="hero-pill">Totals validation</div>
    <div class="hero-pill">User-based library</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="large")
    with left:
        with st.form("login_form"):
            st.markdown('<div class="card-header">Log In</div>', unsafe_allow_html=True)
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in")

        if submitted:
            ok, display, username_norm = verify_login(username, password)
            if ok:
                st.session_state["authenticated"] = True
                st.session_state["username"] = username_norm
                st.session_state["display_name"] = display
                st.session_state["upload_hashes"] = set()
                st.session_state["last_reports"] = []
                st.session_state["selected_invoice"] = None
                st.session_state["nav"] = "Extractor"
                if hasattr(st, "rerun"):
                    st.rerun()
                else:
                    st.experimental_rerun()
            else:
                st.error("Invalid username or password.")

    with right:
        with st.form("register_form"):
            st.markdown('<div class="card-header">New User Registration</div>', unsafe_allow_html=True)
            new_username = st.text_input("New Username")
            display_name = st.text_input("Display Name (optional)")
            new_password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            created = st.form_submit_button("Create Account")

        st.caption("Username: 3-20 chars, letters/numbers/_/-. Password: 6+ chars.")

        if created:
            if new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                ok, msg, username_norm, display = register_user(
                    new_username, new_password, display_name
                )
                if ok:
                    st.success(msg)
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = username_norm
                    st.session_state["display_name"] = display
                    st.session_state["upload_hashes"] = set()
                    st.session_state["last_reports"] = []
                    st.session_state["selected_invoice"] = None
                    st.session_state["nav"] = "Extractor"
                    if hasattr(st, "rerun"):
                        st.rerun()
                    else:
                        st.experimental_rerun()
                else:
                    st.error(msg)

    if not USER_DB_PATH.exists():
        st.info("No users.json found. Default login: admin / admin123. You can register a new user here.")

    st.caption(f"User database: {USER_DB_PATH}")


def render_extractor():
    st.title("Invoice Data Extractor")
    st.write("Upload invoice files (PDFs or images) to extract and audit data.")

    user_dir = ensure_user_library(st.session_state["username"])
    st.caption(f"Library folder: {user_dir}")

    if st.button("Reset Upload Cache"):
        st.session_state["upload_hashes"] = set()
        st.success("Upload cache cleared.")

    uploaded_files = st.file_uploader(
        "Upload Files (Max = 5)",
        accept_multiple_files=True,
        type=["png", "jpg", "jpeg", "pdf"],
    )

    if uploaded_files:
        if len(uploaded_files) > 5:
            st.error("Warning: You can only upload up to 5 files.")
            return

        new_files = []
        for f in uploaded_files:
            data = f.getbuffer()
            digest = hashlib.sha256(data).hexdigest()
            if digest in st.session_state["upload_hashes"]:
                continue
            st.session_state["upload_hashes"].add(digest)
            new_files.append((f, bytes(data)))

        if not new_files:
            st.info("No new files to process. Use Reset Upload Cache to reprocess.")
            if st.session_state["last_reports"]:
                st.markdown("#### Last Results")
                for prepared in st.session_state["last_reports"]:
                    if not prepared:
                        continue
                    if "error" in prepared:
                        st.error(f"{prepared.get('file','UNKNOWN')}: {prepared.get('error')}")
                    else:
                        render_invoice_report(prepared)
            return

        with tempfile.TemporaryDirectory() as tmpdir:
            file_paths = []
            saved_map = {}
            for f, data in new_files:
                path = os.path.join(tmpdir, f.name)
                with open(path, "wb") as out:
                    out.write(data)
                file_paths.append(path)
                saved_path = save_upload_to_library(f, user_dir, data)
                saved_map.setdefault(f.name, []).append(saved_path)

            pdf_files = [p for p in file_paths if p.lower().endswith(".pdf")]
            img_files = [p for p in file_paths if p.lower().endswith((".png", ".jpg", ".jpeg"))]

            with st.spinner("Processing invoices..."):
                results = []
                if img_files:
                    results.extend(process_invoice_dir_img(tmpdir))
                if pdf_files:
                    results.extend(process_invoice_dir_pdf(tmpdir))

            prepared_reports = []
            for inv in results:
                if not inv:
                    continue
                if "error" in inv:
                    prepared_reports.append(inv)
                    st.error(f"{inv.get('file','UNKNOWN')}: {inv.get('error')}")
                    continue
                prepared = prepare_invoice(inv)
                prepared_reports.append(prepared)
                saved_paths = saved_map.get(inv.get("file", ""))
                if saved_paths:
                    save_extracted_data(saved_paths.pop(0), prepared)
                render_invoice_report(prepared)

            if prepared_reports:
                st.session_state["last_reports"] = prepared_reports
    elif st.session_state["last_reports"]:
        st.markdown("#### Last Results")
        for prepared in st.session_state["last_reports"]:
            if not prepared:
                continue
            if "error" in prepared:
                st.error(f"{prepared.get('file','UNKNOWN')}: {prepared.get('error')}")
            else:
                render_invoice_report(prepared)


def render_library():
    st.title("Your Library")
    user_dir = ensure_user_library(st.session_state["username"])
    st.caption(f"Library folder: {user_dir}")

    files = [p for p in user_dir.glob("*") if p.suffix.lower() in INVOICE_EXTS]
    files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        st.info("No files yet. Upload invoices in the Extractor tab.")
        return

    rows = []
    for p in files:
        stat = p.stat()
        json_path, csv_path = extracted_paths(p)
        rows.append(
            {
                "FILE": p.name,
                "SIZE_KB": round(stat.st_size / 1024, 1),
                "UPLOADED": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "DATA_JSON": "Yes" if json_path.exists() else "No",
                "DATA_CSV": "Yes" if csv_path.exists() else "No",
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    file_names = [p.name for p in files]
    if st.session_state.get("selected_invoice") not in file_names:
        st.session_state["selected_invoice"] = file_names[0]
    selected = st.selectbox("Select invoice", file_names, key="selected_invoice")
    selected_path = user_dir / selected

    tab_invoice, tab_data = st.tabs(["View Invoice", "View Data"])
    with tab_invoice:
        render_invoice_file(selected_path)

    with tab_data:
        data, json_path, csv_path = load_extracted_data(selected_path)
        if not data:
            st.info("No extracted data found for this invoice yet.")
        else:
            render_invoice_report(data)
            if csv_path.exists():
                st.download_button(
                    "Download Line Items CSV",
                    csv_path.read_bytes(),
                    file_name=csv_path.name,
                    mime="text/csv",
                )


init_session_state()

if not st.session_state["authenticated"]:
    render_homepage()
    st.stop()

with st.sidebar:
    display_name = st.session_state.get("display_name") or st.session_state.get("username") or "User"
    st.write(f"Signed in as {display_name}")
    nav = st.radio("Navigate", ["Extractor", "Library"], key="nav")
    if st.button("Logout"):
        do_logout()

if nav == "Library":
    render_library()
else:
    render_extractor()

