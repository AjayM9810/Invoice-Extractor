import os
import re
from PIL import Image
import pytesseract
from langdetect import detect
from pytesseract import Output
from collections import defaultdict
from deep_translator import GoogleTranslator
def process_invoice_dir_img(file_dir: str, limit: int = 10):
    results = []
    for i, fname in enumerate(os.listdir(file_dir)):
        if i >= limit:
            break
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        file_path = os.path.join(file_dir, fname)
        try:
            result = process_invoice(file_path)
            results.append(result)
        except Exception as e:
            results.append({
                "file": fname,
                "error": str(e)
            })
    return results

def safe_float(val):
    try:
        return float(val.replace(",", ""))
    except:
        return None

def safe_val(v):
    if isinstance(v, dict) and "value" in v:
        return v["value"]
    return v

def is_valid_line_item(item):
    desc = item.get("description", "").strip().lower()
    if len(desc) < 3:
        return False
    FOOTER_WORDS = [
        "thank", "please", "terms", "notes",
        "invoice within", "pay your", "are our"
    ]
    if any(w in desc for w in FOOTER_WORDS):
        return False
    numeric_count = sum(
        item.get(k) is not None
        for k in ["quantity", "unit_price", "total_price"])
    return numeric_count >= 2

def extract_words_with_layout(path):
    words = []
    img = Image.open(path)
    data = pytesseract.image_to_data(img, output_type=Output.DICT)
    for i, txt in enumerate(data["text"]):
        if txt.strip():
            words.append({
                "text": txt,
                "x": data["left"][i],
                "y": data["top"][i],
                "w": data["width"][i],
                "h": data["height"][i],
                "line": data["line_num"][i],
                "page": 0,
                "source": "image"
            })
    return words

def detect_language_safe(words):
    clean = " ".join(
        w["text"] for w in words
        if w["text"].isalpha() and len(w["text"]) > 3
    )
    try:
        return detect(clean)
    except:
        return "en"

def translate_words(words, lang):
    if lang == "en":
        return words
    text = " ".join(w["text"] for w in words)
    translated = GoogleTranslator(source=lang, target="en").translate(text)
    translated_words = translated.split()
    for i, w in enumerate(words):
        if i < len(translated_words):
            w["text"] = translated_words[i]
    return words

def reconstruct_text(words):
    lines = {}
    for w in words:
        lines.setdefault(w["line"], []).append(w["text"])
    return "\n".join(" ".join(v) for v in lines.values())

def extract_header_fields(text):
    fields = {}
    m = re.search(
        r"(?:Invoice\s*(?:No|Number)?|Invoice\s*#)\s*[:#]?\s*([A-Z0-9\-]*\d[A-Z0-9\-]*)",
        text,
        re.I,
    )
    if m:
        fields["invoice_number"] = m.group(1)
    else:
        m = re.search(r"Invoice#\s*([0-9]{3,})", text, re.I)
        if m:
            fields["invoice_number"] = m.group(1)
    m = re.search(
        r"""(
            \b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b |
            \b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b)""",
        text,
        re.IGNORECASE | re.VERBOSE,
    )
    if m:
        fields["invoice_date"] = m.group(1)
    m = re.search(r"VAT\s*(?:Number|No)?\s*[:\-]?\s*([A-Z0-9]+)", text, re.I)
    if m:
        fields["vat"] = m.group(1)
    return fields

def compute_line_total(item):
    q = safe_val(item.get("quantity"))
    u = safe_val(item.get("unit_price"))
    t = safe_val(item.get("total_price"))
    if q is None or u is None:
        return round(float(t), 2) if t is not None else None
    try:
        return round(float(q) * float(u), 2)
    except Exception:
        return None

def compute_subtotal(line_items):
    total = 0.0
    has_any = False
    for it in line_items:
        ct = compute_line_total(it)
        if ct is not None:
            total += ct
            has_any = True
    return round(total, 2) if has_any else None

def compute_invoice_total(totals, line_items):
    subtotal = compute_subtotal(line_items)
    if subtotal is None:
        return None
    discount = safe_val(totals.get("discount")) or 0.0
    shipping = safe_val(totals.get("shipping")) or 0.0
    tax = safe_val(totals.get("tax")) or 0.0
    try:
        return round(float(subtotal) - float(discount) + float(shipping) + float(tax), 2)
    except Exception:
        return None

def detect_currency(text: str) -> str:
    if not text:
        return "UNKNOWN"
    text_upper = text.upper()
    if "£" in text or "Â£" in text or "Ł" in text or "GBP" in text_upper:
        return "GBP"
    iso_codes = [
        "USD", "EUR", "GBP", "INR", "JPY", "CNY", "AUD", "CAD", "CHF",
        "NZD", "SGD", "HKD", "AED", "SAR", "ZAR", "KRW", "RUB", "TRY",
        "VND", "NGN", "PHP", "THB", "CRC", "UAH", "ILS", "LAK", "PYG",
        "GHS", "KZT", "MNT"
    ]
    for code in iso_codes:
        if code in text_upper:
            return code
    symbol_map = {
        "₹": "INR", "€": "EUR", "£": "GBP", "¥": "JPY", "₩": "KRW",
        "₽": "RUB", "₺": "TRY", "₫": "VND", "₦": "NGN", "₱": "PHP",
        "฿": "THB", "₡": "CRC", "₴": "UAH", "₪": "ILS", "₭": "LAK",
        "₲": "PYG", "₵": "GHS", "₸": "KZT", "₮": "MNT", "$": "USD"
    }
    for sym, code in symbol_map.items():
        if sym in text:
            return code
    if "A$" in text or "AU$" in text:
        return "AUD"
    if "C$" in text or "CA$" in text:
        return "CAD"
    return "UNKNOWN"

def detect_company(words, text):
    KEYWORDS = {"invoice","bill","ship","date","subtotal","total","discount",
                "balance","amount","notes","terms","order","mode","payment"}
    candidates = []
    for w in words:
        if w.get("page", 0) != 0:
            continue
        clean = (w.get("text") or "").strip()
        if not clean or len(clean) > 60:
            continue
        low = clean.lower()
        if any(k in low for k in KEYWORDS):
            continue
        if clean.isdigit():
            continue
        if re.search(r"\d|,", clean):
            continue
        candidates.append((w.get("y", 0), clean))
    if candidates:
        candidates.sort(key=lambda t: t[0])
        return candidates[0][1]
    for l in [l.strip() for l in text.splitlines() if l.strip()][:8]:
        low = l.lower()
        if any(k in low for k in KEYWORDS):
            continue
        if re.search(r"\d|,", l):
            continue
        return l
    return None

def extract_metadata(text, file_name, words, header):
    company = detect_company(words, text) or "UNKNOWN"
    invoice_number = header.get("invoice_number")
    invoice_date = header.get("invoice_date")
    ship_to = None
    m = re.search(r"(Ship To|Bill To|Invoiced To)\s*:?\\s*([^\n]+)", text, re.I)
    if m:
        ship_to = m.group(2).strip()
    else:
        line_map = {}
        for w in words:
            if w.get("page", 0) != 0:
                continue
            line_map.setdefault(w.get("line", 0), []).append(w.get("text", ""))
        lines = [" ".join(line_map[k]).strip() for k in sorted(line_map) if " ".join(line_map[k]).strip()]
        if not lines:
            lines = [l.strip() for l in text.splitlines() if l.strip()]
        vat_y = None
        header_y = None
        for w in words:
            if w.get("page", 0) != 0:
                continue
            if (w.get("text") or "").lower() == "vat":
                vat_y = w.get("y")
            low = (w.get("text") or "").lower()
            if low in {"item", "description", "unit", "cost", "quantity", "line", "total"}:
                header_y = w.get("y") if header_y is None else min(header_y, w.get("y"))
        if vat_y is not None and header_y is not None:
            line_map_y = defaultdict(list)
            for w in words:
                if w.get("page", 0) != 0:
                    continue
                y = w.get("y", 0)
                if y <= vat_y or y >= header_y:
                    continue
                y_key = round(y / 8) * 8
                line_map_y[y_key].append(w)
            candidate_lines = []
            for y in sorted(line_map_y.keys()):
                line = " ".join(w.get("text", "") for w in sorted(line_map_y[y], key=lambda x: x.get("x", 0))).strip()
                if not line:
                    continue
                if re.search(r"\b(Invoice|Date|Amount|Item|Description|Unit|Cost|Quantity|Line|Subtotal|Discount|Total|Terms|Notes)\b", line, re.I):
                    continue
                if len(re.findall(r"\d", line)) >= 8:
                    continue
                candidate_lines.append(line)
            if candidate_lines:
                ship_to = " ".join(candidate_lines[:3]).strip()
        if not ship_to:
            suffix_re = re.compile(r"\b(inc|llc|ltd|plc|corp|co|gmbh|srl|bv|oy|sa|sas|spa|pte)\b", re.I)
            header_text = " ".join(lines[:6])
            comps = []
            for m2 in re.finditer(r"[A-Za-z][A-Za-z&\- ]+\b", header_text):
                seg = m2.group(0).strip()
                if suffix_re.search(seg):
                    comps.append(seg)
            for c in comps:
                if company and company.lower() not in c.lower():
                    ship_to = c
                    break
        if not ship_to:
            m3 = re.search(r"\b[A-Z][a-z]+\\s+[A-Z][a-z]+\\b", text)
            if m3:
                ship_to = m3.group(0)
    return {
        "file": file_name,
        "company": company,
        "invoice_number": invoice_number,
        "invoice_date": invoice_date,
        "ship_to": ship_to,
    }
INVOICE_TOTAL_PRIORITY = ["grand total", "invoice total", "total amount", "total"]
AMOUNT_DUE_PRIORITY = ["amount due", "balance due", "total due", "amount payable"]

def extract_items_by_columns(words):
    """
    STRICT column-based extraction.
    No math inference.
    No guessing.
    No fallback.
    """
    HEADER_ALIASES = {
        "description": ["description", "details", "item"],
        "quantity": ["qty", "quantity", "quantlty"],
        "unit_price": ["unit", "unit cost", "rate", "price"],
        "total_price": ["line total", "amount", "total"]
    }
    for w in words:
        w["text_norm"] = w["text"].lower().strip()
    lines = defaultdict(list)
    for w in words:
        y_bucket = int(round(w["y"] / 10))
        lines[y_bucket].append(w)
    header_words = []
    header_y = None
    for y in sorted(lines.keys()):
        ws = lines[y]
        roles_found = set()
        for w in ws:
            for role, aliases in HEADER_ALIASES.items():
                if any(a in w["text_norm"] for a in aliases):
                    roles_found.add(role)
        if len(roles_found) >= 3:
            header_words = ws
            header_y = y * 10
            break
    if not header_words:
        return []
    columns = {}
    for col, aliases in HEADER_ALIASES.items():
        hits = [w for w in header_words if any(a in w["text_norm"] for a in aliases)]
        if hits:
            x1 = min(w["x"] for w in hits) - 15
            x2 = max(w["x"] + w["w"] for w in hits) + 80
            columns[col] = (x1, x2)
    if "description" not in columns:
        return []
    rows = {}
    for w in words:
        if w["y"] <= header_y + 10:
            continue
        row_key = round(w["y"] / 8) * 8
        rows.setdefault(row_key, []).append(w)
    items = []
    def is_footer_line(ws):
        line_text = " ".join(w["text"] for w in ws).lower()
        return any(k in line_text for k in ["subtotal", "discount", "total", "terms", "notes", "please pay"])
    for _, row_words in sorted(rows.items()):
        if is_footer_line(row_words):
            break
        item = {
            "description": "",
            "quantity": None,
            "unit_price": None,
            "total_price": None
        }
        for w in row_words:
            x_mid = w["x"] + w["w"] / 2
            for col, (x1, x2) in columns.items():
                if x1 <= x_mid <= x2:
                    if col == "description":
                        if not re.fullmatch(r"\d{1,3}", w["text"].strip()):
                            item["description"] += " " + w["text"]
                    else:
                        try:
                            val = float(w["text"].replace(",", ""))
                            if item[col] is None:
                                item[col] = val
                        except:
                            pass
        if item["description"].strip() or any(
            item[k] is not None for k in ["quantity", "unit_price", "total_price"]
        ):
            item["description"] = item["description"].strip()
            items.append(item)
    return items

def extract_items_from_synth_image(words):
    if not words:
        return []
    rows = []
    sorted_words = sorted([w for w in words if w.get("page", 0) == 0], key=lambda w: w.get("y", 0))
    if not sorted_words:
        return []
    heights = [w.get("h", 0) for w in sorted_words if w.get("h", 0)]
    row_gap = int(max(6, min(14, (sum(heights) / len(heights)) if heights else 10)))
    current = []
    current_y = None
    for w in sorted_words:
        y = w.get("y", 0)
        if current_y is None or abs(y - current_y) <= row_gap:
            current.append(w)
            current_y = y if current_y is None else min(current_y, y)
        else:
            rows.append(current)
            current = [w]
            current_y = y
    if current:
        rows.append(current)
    header_idx = None
    header_words = None
    for i, row in enumerate(rows):
        tokens = [w.get("text", "").lower() for w in row]
        hit = sum(1 for t in tokens if t in {"item", "description", "unit", "cost", "quantity", "line", "total"})
        if hit >= 4 and "description" in tokens and "quantity" in tokens:
            header_idx = i
            header_words = row
            break
    if header_idx is None:
        return []
    def xs(label):
        return [w.get("x", 0) for w in header_words if (w.get("text", "").lower() == label)]
    item_x = min(xs("item"), default=None)
    desc_x = min(xs("description"), default=None)
    unit_xs = xs("unit")
    cost_xs = xs("cost")
    qty_x = min(xs("quantity"), default=None)
    line_xs = xs("line")
    total_xs = xs("total")
    unit_center = None
    if unit_xs and cost_xs:
        unit_center = (min(unit_xs) + min(cost_xs)) / 2.0
    elif unit_xs:
        unit_center = min(unit_xs)
    elif cost_xs:
        unit_center = min(cost_xs)
    total_center = None
    if line_xs and total_xs:
        total_center = (min(line_xs) + min(total_xs)) / 2.0
    elif line_xs:
        total_center = min(line_xs)
    elif total_xs:
        total_center = min(total_xs)
    centers = []
    for role, x in [
        ("item", item_x),
        ("description", desc_x),
        ("unit_price", unit_center),
        ("quantity", qty_x),
        ("total_price", total_center),
    ]:
        if x is not None:
            centers.append((x, role))
    if len(centers) < 3:
        return []
    centers.sort(key=lambda t: t[0])
    starts = [c[0] for c in centers]
    roles_by_x = [c[1] for c in centers]
    edges = [starts[0] - 10]
    for i in range(len(starts) - 1):
        edges.append((starts[i] + starts[i + 1]) / 2.0)
    edges.append(float("inf"))
    def assign_role(x_mid):
        for i in range(len(edges) - 1):
            if edges[i] <= x_mid < edges[i + 1]:
                return roles_by_x[min(i, len(roles_by_x) - 1)]
        return roles_by_x[-1]
    items = []
    for row in rows[header_idx + 1:]:
        line_text = " ".join(w.get("text", "") for w in row).lower()
        if any(k in line_text for k in ["subtotal", "discount", "total", "terms", "notes", "please pay"]):
            break
        if "description" in line_text and "quantity" in line_text:
            continue
        item = {"description": "", "quantity": None, "unit_price": None, "total_price": None}
        for w in row:
            x_mid = w.get("x", 0) + (w.get("w", 0) / 2.0)
            role = assign_role(x_mid)
            txt = w.get("text", "").strip()
            if role in ("quantity", "unit_price", "total_price"):
                val = safe_float(txt)
                if val is None:
                    m = re.search(r"-?\d+(?:\.\d+)?", txt.replace(",", ""))
                    if m:
                        val = safe_float(m.group(0))
                if val is not None and item[role] is None:
                    item[role] = val
            elif role == "description":
                if not re.fullmatch(r"\d{1,3}", txt):
                    item["description"] = (item["description"] + " " + txt).strip()
        if item["description"] and sum(item[k] is not None for k in ["quantity", "unit_price", "total_price"]) >= 2:
            items.append(item)
    return items

def extract_items_from_text_lines(text):
    items = []
    for line in text.split("\n"):
        clean = line.strip()
        if not clean:
            continue
        numbers = re.findall(r"\d+\.\d{2}|\d+", clean)
        if not numbers:
            continue
        desc = re.sub(r"\d+\.\d{2}|\d+", "", clean).strip()
        if len(desc) < 3:
            continue
        item = {
            "description": desc,
            "quantity": None,
            "unit_price": None,
            "total_price": safe_float(numbers[-1])
        }
        items.append(item)
    return items

def extract_items_from_inline_text(text):
    items = []
    pat3 = re.compile(r"([A-Za-z][A-Za-z\- ]{3,})\s+(\d+(?:\.\d{1,2})?)\s+(\d+(?:\.\d{1,2})?)\s+(\d+(?:\.\d{1,2})?)")
    pat2 = re.compile(r"([A-Za-z][A-Za-z\- ]{3,})\s+(\d+(?:\.\d{1,2})?)\s+(\d+(?:\.\d{1,2})?)")
    for m in pat3.finditer(text):
        desc = m.group(1).strip()
        items.append({
            "description": desc,
            "quantity": safe_float(m.group(2)),
            "unit_price": safe_float(m.group(3)),
            "total_price": safe_float(m.group(4)),
        })
    if not items:
        for m in pat2.finditer(text):
            desc = m.group(1).strip()
            items.append({
                "description": desc,
                "quantity": None,
                "unit_price": safe_float(m.group(1)),
                "total_price": safe_float(m.group(2)),
            })
    return items

def extract_items_from_synth_table(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    header_idx = None
    for i, line in enumerate(lines):
        if re.search(r"\bitem\b", line, re.I) and re.search(r"\bdescription\b", line, re.I) \
           and re.search(r"\bunit\s*cost\b", line, re.I) and re.search(r"\bquantity\b", line, re.I) \
           and re.search(r"\bline\s*total\b", line, re.I):
            header_idx = i
            break
    if header_idx is None:
        return []
    items = []
    row_re = re.compile(r"^\s*\d+\s+(.*?)\s+(\d+(?:\.\d{1,2})?)\s+(\d+(?:\.\d{1,2})?)\s+(\d+(?:\.\d{1,2})?)\s*$")
    for line in lines[header_idx+1:]:
        if re.search(r"subtotal|discount|total|terms|notes|please pay", line, re.I):
            break
        m = row_re.match(line)
        if not m:
            continue
        desc = m.group(1).strip()
        unit = safe_float(m.group(2))
        qty = safe_float(m.group(3))
        total = safe_float(m.group(4))
        items.append({
            "description": desc,
            "quantity": qty,
            "unit_price": unit,
            "total_price": total
        })
    return items

def extract_items_from_unitcost_qty_total(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    header_idx = None
    for i, line in enumerate(lines):
        if re.search(r"unit\\s+cost", line, re.I) and re.search(r"quantity", line, re.I) and re.search(r"line\\s+total", line, re.I):
            header_idx = i
            break
    if header_idx is None:
        return []
    items = []
    row_re = re.compile(r"^\\s*\\d+\\s+(.*?)\\s+(\\d+(?:\\.\\d{1,2})?)\\s+(\\d+(?:\\.\\d{1,2})?)\\s+(\\d+(?:\\.\\d{1,2})?)\\s*$")
    for line in lines[header_idx+1:]:
        if re.search(r"subtotal|total|tax|vat|discount|shipping", line, re.I):
            break
        m = row_re.match(line)
        if not m:
            continue
        desc = m.group(1).strip()
        unit = safe_float(m.group(2))
        qty = safe_float(m.group(3))
        total = safe_float(m.group(4))
        items.append({
            "description": desc,
            "quantity": qty,
            "unit_price": unit,
            "total_price": total
        })
    return items

def normalize_item_math(item):
    return item

def extract_totals(text):
    totals = {}
    def last_match(pattern):
        matches = list(re.finditer(pattern, text, re.I))
        if matches:
            return safe_float(matches[-1].group(1))
        return None
    totals["subtotal"] = last_match(r"\bsub\s*total\b[^0-9\-]*(-?\d[\d,]*\.\d{1,2})")
    discount_val = None
    disc_lines = [l for l in text.splitlines() if re.search(r"\bdiscount\b", l, re.I)]
    if disc_lines:
        line = disc_lines[-1]
        nums = re.findall(r"-?\d[\d,]*\.\d{1,2}", line)
        if len(nums) >= 2:
            discount_val = safe_float(nums[-1])
        elif len(nums) == 1:
            discount_val = safe_float(nums[0])
    if discount_val is None:
        discount_val = last_match(r"\bdiscount\b[^0-9\-]*(-?\d[\d,]*\.\d{1,2})")
    totals["discount"] = discount_val
    totals["tax"] = last_match(r"\b(?:tax|vat)\b[^0-9\-]*(-?\d[\d,]*\.\d{1,2})")
    totals["shipping"] = last_match(r"\b(?:shipping|delivery|freight)\b[^0-9\-]*(-?\d[\d,]*\.\d{1,2})")
    totals["invoice_total"] = last_match(r"\b(?:grand total|invoice total|total amount|total)\b(?!\s*vat)[^0-9\-]*(-?\d[\d,]*\.\d{1,2})")
    totals["amount_due"] = last_match(r"\b(?:amount due|balance due|total due|amount payable)\b[^0-9\-]*(-?\d[\d,]*\.\d{1,2})")
    if "invoice_total" in totals and "amount_due" in totals:
        if totals["invoice_total"] is not None and totals["amount_due"] is not None:
            if abs(totals["invoice_total"] - totals["amount_due"]) > 0.01:
                totals["note"] = (
                    "invoice_total != amount_due "
                    "(expected when discount / credit / advance applied)"
                )
    return totals

def normalize_items(items):
    clean = []
    for i in items:
        if not i.get("description"):
            continue
        clean.append(i)
    return clean

def _filter_items(items):
    return [i for i in items if is_valid_line_item(i)]

def _best_items(candidates):
    if not candidates:
        return []
    best = []
    best_score = (-1, -1)
    for items in candidates:
        score = (len(items), sum(1 for it in items for k in ["quantity", "unit_price", "total_price"] if it.get(k) is not None))
        if score > best_score:
            best = items
            best_score = score
    return best

def process_invoice(file_path: str) -> dict:
    words = extract_words_with_layout(file_path)
    lang = detect_language_safe(words)
    words = translate_words(words, lang)
    full_text = reconstruct_text(words)
    currency = detect_currency(full_text)
    header = extract_header_fields(full_text)
    candidates = []
    candidates.append(_filter_items(extract_items_from_synth_image(words)))
    candidates.append(_filter_items(extract_items_by_columns(words)))
    candidates.append(_filter_items(extract_items_from_synth_table(full_text)))
    candidates.append(_filter_items(extract_items_from_unitcost_qty_total(full_text)))
    candidates.append(_filter_items(extract_items_from_inline_text(full_text)))
    candidates.append(_filter_items(extract_items_from_text_lines(full_text)))
    raw_items = _best_items([c for c in candidates if c])
    clean_items = [
        item for item in raw_items
        if is_valid_line_item(item)
    ]
    line_items = clean_items
    line_items = normalize_items(clean_items)
    line_items = [normalize_item_math(i) for i in line_items]
    totals = extract_totals(full_text)
    return {
        "file": os.path.basename(file_path),
        "language": lang,
        "currency": currency,
        "header": header,
        "line_items": line_items,
        "totals": totals,
        "raw_text": full_text,
        "words": words
    }

def print_audit_report(invoices):
    print("========== INVOICE AUDIT REPORT ==========")
    for inv in invoices:
        if "error" in inv:
            print(f"\nInvoice: {inv.get('file','UNKNOWN')} failed -> {inv.get('error')}")
            continue
        raw_text = inv.get("raw_text", "")
        words = inv.get("words", [])
        header = inv.get("header", {})
        meta = extract_metadata(raw_text, inv.get("file", "UNKNOWN"), words, header)
        print(f"\n------------ Invoice: {meta['file']} ------------")
        print(f"Company      : {meta.get('company') or 'UNKNOWN'}")
        print(f"Invoice No   : {meta.get('invoice_number') or 'UNKNOWN'}")
        print(f"Date         : {meta.get('invoice_date') or 'UNKNOWN'}")
        print(f"Currency     : {inv.get('currency') or 'UNKNOWN'}")
        print(f"Language     : {inv.get('language') or 'en'}")
        print(f"Ship To      : {meta.get('ship_to') or 'N/A'}")
        print(f"VAT No       : {header.get('vat') or 'N/A'}")
        t = inv.get('totals', {})
        line_items = inv.get('line_items', [])
        subtotal_computed = compute_subtotal(line_items)
        invoice_total_computed = compute_invoice_total(t, line_items)
        def fmt(v):
            return "N/A" if v is None or v == "" else str(v)
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
        print("\nTotals")
        print(" ---------------------------- ")
        print(f" Subtotal      : {fmt(t.get('subtotal'))}")
        print(f" Subtotal(C)   : {fmt(subtotal_computed)}")
        print(f" Discount      : {fmt(t.get('discount'))}")
        print(f" Shipping      : {fmt(t.get('shipping'))}")
        print(f" Tax/VAT       : {fmt(t.get('tax'))}")
        print(f" Amount Due    : {fmt(t.get('amount_due'))}")
        print(f" Invoice Total : {fmt(t.get('invoice_total'))}")
        print(f" Invoice(C)    : {fmt(invoice_total_computed)}")
        print(" ---------------------------- ")
        subtotal_diff = pct_diff(t.get('subtotal'), subtotal_computed)
        invoice_diff = pct_diff(t.get('invoice_total'), invoice_total_computed)
        subtotal_status = status_from_diff(subtotal_diff)
        invoice_status = status_from_diff(invoice_diff)
        overall_status = "PASS"
        if subtotal_status == "FAIL" or invoice_status == "FAIL":
            overall_status = "FAIL"
        elif subtotal_status == "MINOR MISMATCH" or invoice_status == "MINOR MISMATCH":
            overall_status = "MINOR MISMATCH"
        manual_review = "YES" if (invoice_diff is not None and invoice_diff > 0.05) else "NO"
        print("\nStatus")
        print(f" Subtotal      : {subtotal_status}")
        print(f" Invoice Total : {invoice_status}")
        print(f" Overall       : {overall_status}")
        print(f" Manual Review : {manual_review}")
        print(f"\nLine Items ({len(line_items)})")
        print(" Description                | Qty     | Unit Price | Line Total | Computed  | Status")
        print(" ---------------------------+---------+------------+------------+-----------+--------")
        for li in line_items:
            desc = (li.get('description') or "")[:26]
            qty = li.get('quantity')
            unit = li.get('unit_price')
            total = li.get('total_price')
            computed = compute_line_total(li)
            line_diff = pct_diff(total, computed)
            line_status = "MATCH" if line_diff is not None and line_diff <= 0.005 else ("MISMATCH" if line_diff is not None else "N/A")
            print(f" {desc:<26} | {fmt(qty):<7} | {fmt(unit):<10} | {fmt(total):<10} | {fmt(computed):<9} | {line_status}")

if __name__ == "__main__":
    image_dir = r"D:\Invoice_Extractor\7000_invoice_images_with_json\image"
    scanned_results = process_invoice_dir_img(
        file_dir=image_dir,
        limit=5)
    print_audit_report(scanned_results)
