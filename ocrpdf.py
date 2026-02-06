import os
import re
import pdfplumber
from langdetect import detect
from collections import defaultdict
from deep_translator import GoogleTranslator

ENABLE_TRANSLATION = False  # set True if you want translation attempts
TRANSLATION_MAX_CHARS = 5000

def process_invoice_dir_pdf(file_dir: str, limit: int = 5):
    results = []
    for i, fname in enumerate(os.listdir(file_dir)):
        if i >= limit:
            break
        if not fname.lower().endswith(".pdf"):
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
    if val is None:
        return None
    s = str(val).strip()
    s = re.sub(r"[^\d,.\-]", "", s)
    if s.count(",") == 1 and s.count(".") >= 1 and s.rfind(",") > s.rfind("."):
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", "")
    try:
        f = float(s)
        if f > 1e7:  # filter out huge IDs
            return None
        return f
    except:
        return None

def sane_money(val):
    v = safe_float(val)
    if v is None:
        return None
    if v > 1_000_000:
        return None
    if v.is_integer() and YEAR_PATTERN.fullmatch(str(int(v))):
        return None
    return v

def safe_val(v):
    if isinstance(v, dict) and "value" in v:
        return v["value"]
    return v

def contains_blocked_keyword(text, blocklist):
    low = (text or "").lower()
    for k in blocklist:
        if "@" in k:
            if k in low:
                return True
            continue
        if re.search(rf"\b{re.escape(k)}\b", low):
            return True
    return False

def is_valid_line_item(item):
    desc_raw = (item.get("description") or "").strip()
    desc = desc_raw.lower()
    if len(desc) < 3:
        return False
    BLOCKLIST = [
        "invoice","subtotal","discount","tax","total","balance",
        "payment","due","code","phone","fax","vat","bankgiro","plusgiro","sample",
        "billing","delivery","shipping","address","bill to","ship to",
        "street","st","road","rd","avenue","ave","blvd","suite",
        "city","state","zip","county","country"
    ]
    if contains_blocked_keyword(desc, BLOCKLIST):
        return False
    SHORT_BLOCKLIST = {"po", "id", "no"}
    if re.search(rf"\b({'|'.join(SHORT_BLOCKLIST)})\b", desc):
        return False
    tp = sane_money(item.get("total_price"))
    up = sane_money(item.get("unit_price"))
    if tp is None and up is None:
        return False
    if re.fullmatch(r"[A-Z]{2}\s+[A-Z]{2}", desc_raw) and item.get("quantity") is None and item.get("unit_price") is None:
        if tp is not None and tp <= 5:
            return False
    if item.get("quantity") is None and item.get("unit_price") is None and tp is not None:
        if float(tp).is_integer() and tp >= 10000:
            return False
    q = item.get("quantity")
    if q is not None and (q<=0 or q>1e6):
            return False
    if not re.search(r"[A-Za-z]", desc):
        return False
    return True

def extract_words_with_layout(path):
    words = []
    with pdfplumber.open(path) as pdf:
        for page_no, page in enumerate(pdf.pages):
            for w in page.extract_words(use_text_flow=True):
                words.append({
                    "text": w["text"],
                    "x": w["x0"],
                    "y": w["top"],
                    "w": w["x1"] - w["x0"],
                    "h": w["bottom"] - w["top"],
                    "line": int(w["top"] // 5),
                    "page": page_no,
                    "source": "pdf"
                })
    return words

YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")

def extract_items_from_structured_pdf(text: str):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    start_idx = None
    for i in range(len(lines) - 3):
        if (
            lines[i].lower() == "item"
            and lines[i+1].lower() == "quantity"
            and lines[i+2].lower() in ("rate", "unit price", "price")
            and lines[i+3].lower() in ("amount", "total")
        ):
            start_idx = i + 4
            break
    if start_idx is None:
        return []
    end_idx = len(lines)
    for i in range(start_idx, len(lines)):
        if lines[i].lower().startswith(
            ("subtotal", "discount", "shipping", "tax", "total")
        ):
            end_idx = i
            break
    items = []
    i = start_idx
    while i + 3 < end_idx:
        desc = lines[i]
        if desc.startswith("$") or desc.replace(".", "").isdigit():
            i += 1
            continue
        try:
            qty = safe_float(lines[i+1])
            unit = safe_float(lines[i+2].replace("$", ""))
            total = safe_float(lines[i+3].replace("$", ""))
        except:
            i += 1
            continue
        items.append({
            "description": desc,
            "quantity": qty,
            "unit_price": unit,
            "total_price": total
        })
        i += 4  # jump to next row
    return items

def extract_items_from_pdf_text(text: str):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    start_idx = None
    for i, line in enumerate(lines):
        if re.search(r"(item|id|line).*(qty|quantity).*(price|unit price|rate|unit).*(total|amount|line total)", line, re.I):
            start_idx = i + 1
            break
    if start_idx is None:
        return []
    end_idx = len(lines)
    for i in range(start_idx, len(lines)):
        if re.match(r"(subtotal|discount|shipping|tax|total)\b", lines[i], re.I):
            end_idx = i
            break
    items = []
    for line in lines[start_idx:end_idx]:
        nums = re.findall(r"\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?", line)
        if len(nums) < 2:
            continue
        nums = [safe_float(n.replace("$", "")) for n in nums if safe_float(n)]
        if len(nums) < 2:
            continue
        desc = re.sub(r"\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?", "", line).strip()
        if len(desc) < 5:
            continue
        item = {
            "description": desc,
            "quantity": None,
            "unit_price": None,
            "total_price": None
        }
        if len(nums) >= 3:
            item["quantity"] = nums[0]
            item["unit_price"] = nums[1]
            item["total_price"] = nums[-1]
        else:
            item["unit_price"] = nums[0]
            item["total_price"] = nums[-1]
        items.append(item)
    return items

def extract_items_from_id_table(text: str):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    start_idx = None
    for i, line in enumerate(lines):
        if re.search(r"\bID\b", line, re.I) and re.search(r"\bDESCRIPTION\b", line, re.I) \
           and re.search(r"\b(QTY|QUANTITY)\b", line, re.I) and re.search(r"\b(PRICE|RATE)\b", line, re.I) \
           and re.search(r"\b(TOTAL|AMOUNT)\b", line, re.I):
            start_idx = i + 1
            break
    if start_idx is None:
        return []
    end_idx = len(lines)
    for i in range(start_idx, len(lines)):
        if re.search(r"\b(sub\s*total|subtotal|tax|vat|shipping|total)\b", lines[i], re.I):
            end_idx = i
            break
    items = []
    row_re = re.compile(
        r"^\s*\d+\s+(.*?)\s+(\d+(?:\.\d+)?)\s+\$?([\d,]+(?:\.\d{2})?)\s+([\d,]+(?:\.\d{2})?)\s*$"
    )
    for line in lines[start_idx:end_idx]:
        m = row_re.match(line)
        if not m:
            continue
        desc = m.group(1).strip()
        qty = safe_float(m.group(2))
        unit = safe_float(m.group(3))
        total = safe_float(m.group(4))
        items.append({
            "description": desc,
            "quantity": qty,
            "unit_price": unit,
            "total_price": total
        })
    return items

def detect_language_safe(words):
    tokens = []
    for w in words:
        t = (w.get("text") or "").strip()
        if len(t) >= 4 and t.isalpha():
            tokens.append(t)
    if len(tokens) < 20:  # require minimum signal
        return "en"
    sample = " ".join(tokens)[:TRANSLATION_MAX_CHARS]
    try:
        lang = detect(sample)
        return lang or "en"
    except:
        return "en"

def translate_words(words, lang):
    if lang == "en":
        return words
    alpha_tokens = [w["text"] for w in words if re.search(r"[A-Za-z]", w["text"])]
    if not alpha_tokens:
        return words
    try:
        translated = GoogleTranslator(source=lang, target="en").translate(" ".join(alpha_tokens))
        translated_tokens = translated.split()
    except:
        return words
    it = iter(translated_tokens)
    for w in words:
        if re.search(r"[A-Za-z]", w["text"]):
            try:
                w["text"] = next(it)
            except StopIteration:
                break
    return words

def reconstruct_text(words):
    lines = defaultdict(list)
    for w in words:
        key = (w.get("page", 0), round(w.get("y", 0) / 6) * 6)
        lines[key].append(w)
    out = []
    for (page, y), ws in sorted(lines.items(), key=lambda k: (k[0][0], k[0][1])):
        ws = sorted(ws, key=lambda x: x.get("x", 0))
        out.append(" ".join(w["text"] for w in ws))
    return "\n".join(out)

DATE_PATTERN = re.compile(r"""
    \b(
        \d{1,2}[./-]\d{1,2}[./-]\d{2,4} |
        (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}
    )\b
""", re.IGNORECASE | re.VERBOSE)

def extract_header_fields(text: str):
    fields = {}
    m = re.search(r"INVOICE NUMBER[^0-9]*([0-9]{3,})", text, re.I)
    if m:
        fields["invoice_number"] = m.group(1)
    else:
        m = re.search(r"Invoice\s*(?:No|Number|#)[:\s]*([0-9]{3,})", text, re.I)
        if m:
            fields["invoice_number"] = m.group(1)
    m = DATE_PATTERN.search(text)
    if m:
        fields["invoice_date"] = m.group(1)
    m = re.search(r"VAT\s*(?:Number|No)?\s*[:\-]?\s*([A-Z]{2}\s*\d[\d\s]+)", text, re.I)
    if m:
        fields["vat"] = re.sub(r"\s+", " ", m.group(1)).strip()
    return fields

def compute_line_total(item):
    q = safe_val(item.get("quantity"))
    u = safe_val(item.get("unit_price"))
    t = safe_val(item.get("total_price"))
    rebate = safe_val(item.get("rebate"))
    if q is None or u is None:
        return round(float(t), 2) if t is not None else None
    try:
        qf = float(q)
        uf = float(u)
        base = qf * uf
        if t is not None:
            try:
                tf = float(t)
                if abs(base - tf) <= 0.05:
                    return round(base, 2)
            except Exception:
                pass
        rebate_val = None
        if rebate is not None:
            if isinstance(rebate, str):
                nums = re.findall(r"\d+(?:\.\d+)?", rebate)
                if nums:
                    rebate_val = float(nums[-1])
            else:
                rebate_val = float(rebate)
        if rebate_val is not None:
            alt = qf * (uf - rebate_val)
            if t is not None:
                try:
                    tf = float(t)
                    if abs(alt - tf) < abs(base - tf):
                        return round(alt, 2)
                except Exception:
                    pass
            return round(alt, 2)
        return round(base, 2)
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
    t = text.upper()
    iso_codes = {
        "USD","EUR","GBP","INR","JPY","CNY","AUD","CAD","CHF","NZD","SGD","HKD",
        "AED","SAR","ZAR","KRW","RUB","TRY","VND","NGN","PHP","THB","CRC","UAH",
        "ILS","LAK","PYG","GHS","KZT","MNT","DKK","NOK","SEK","PLN","HUF","CZK",
        "RON","BGN","MXN","BRL","ARS","CLP","COP","PEN","IDR","MYR","TWD"
    }
    for code in iso_codes:
        if re.search(rf"\b{code}\b", t):
            return code
    if re.search(r"VAT.*GB", t): return "GBP"
    if re.search(r"VAT.*CN", t): return "CNY"
    if re.search(r"VAT.*EU", t): return "EUR"
    keyword_map = {
        "RUPEE": "INR", "₹": "INR",
        "EURO": "EUR", "€": "EUR",
        "POUND": "GBP", "£": "GBP",
        "YEN": "JPY", "¥": "JPY",
        "WON": "KRW", "₩": "KRW",
        "RUBLE": "RUB", "₽": "RUB",
        "LIRA": "TRY", "₺": "TRY",
        "DONG": "VND", "₫": "VND",
        "NAIRA": "NGN", "₦": "NGN",
        "PESO": "PHP", "₱": "PHP",
        "BAHT": "THB", "฿": "THB",
        "SHEKEL": "ILS", "₪": "ILS",
        "KIP": "LAK", "₭": "LAK",
        "GUARANI": "PYG", "₲": "PYG",
        "CEDI": "GHS", "₵": "GHS",
        "TENGE": "KZT", "₸": "KZT",
        "TUGRIK": "MNT", "₮": "MNT"
    }
    for k, code in keyword_map.items():
        if k in t:
            return code
    if "$" in text:
        if re.search(r"\bCAD\b|\bC\$|\bCA\$|\bCAN\b", t): return "CAD"
        if re.search(r"\bAUD\b|\bA\$|\bAU\$|\bAUS\b", t): return "AUD"
        if re.search(r"\bNZD\b|\bNZ\$|\bN\$|\bNZ\b", t): return "NZD"
        if re.search(r"\bSGD\b|\bS\$|\bSG\b", t): return "SGD"
        if re.search(r"\bHKD\b|\bHK\$|\bHK\b", t): return "HKD"
        if re.search(r"\bMXN\b|\bMEX\b", t): return "MXN"
        return "USD"
    return "UNKNOWN"

INVOICE_TOTAL_PRIORITY = ["grand total", "invoice total", "total amount", "total"]
AMOUNT_DUE_PRIORITY = ["amount due", "balance due", "total due", "amount payable"]

def extract_items_by_columns(words):
    HEADER_ALIASES = {
        "item_code": ["item", "item #", "id", "line"],
        "description": ["description", "details"],
        "quantity": ["qty", "quantity"],
        "unit_price": ["unit", "unit cost", "rate", "price", "unit price"],
        "rebate": ["rebate", "discount"],
        "total_price": ["line total", "amount", "total"]
    }
    BLOCKLIST = {"phone","fax","bankgiro","plusgiro","vat","org","ocr","street","corp","inc","info@", "invoice", "billing", "delivery", "shipping", "subtotal", "balance", "due", "order", "sample"}
    for w in words:
        w["text_norm"] = (w.get("text") or "").lower().strip()
    items = []
    pages = sorted(set(w["page"] for w in words))
    header_positions_global = None
    header_y_global = None
    for page in pages:
        page_words = [w for w in words if w["page"] == page]
        lines = defaultdict(list)
        for w in page_words:
            y_bucket = int(round(w.get("y", 0) / 6.0))
            lines[y_bucket].append(w)
        header_positions = {}
        header_y = None
        header_detected_on_page = False
        for y in sorted(lines.keys()):
            ws = lines[y]
            roles_found = set()
            for w in ws:
                for role, aliases in HEADER_ALIASES.items():
                    if w["text_norm"] in aliases:
                        roles_found.add(role)
            if len(roles_found) >= 2:
                header_y = y * 6
                header_detected_on_page = True
                for role, aliases in HEADER_ALIASES.items():
                    cands = [w for w in ws if w["text_norm"] in aliases]
                    if cands:
                        header_positions[role] = min(cands, key=lambda x: x["x"])["x"]
                break
        if not header_positions and header_positions_global:
            header_positions = header_positions_global.copy()
        else:
            if header_positions_global:
                for k, v in header_positions_global.items():
                    header_positions.setdefault(k, v)
            header_positions_global = header_positions.copy() if header_positions else header_positions_global
        if header_y is None:
            header_y = header_y_global
        else:
            header_y_global = header_y
        if not header_positions:
            continue
        bands = sorted(header_positions.items(), key=lambda kv: kv[1])
        roles = [r for r, _ in bands]
        xs = [x for _, x in bands]
        if not xs:
            continue
        edges = []
        left_margin = xs[0] - 50
        edges.append(left_margin)
        for i in range(len(xs) - 1):
            edges.append((xs[i] + xs[i+1]) / 2.0)
        edges.append(float("inf"))
        footer_y = None
        footer_patterns = re.compile(
            r"\b(sub\s*total|subtotal|total\s*due|amount\s*due|balance\s*due|"
            r"tax|vat|discount|shipping|grand\s*total|invoice\s*total)\b",
            re.I,
        )
        for y in sorted(lines.keys()):
            line_text = " ".join((w.get("text") or "").strip() for w in lines[y]).strip()
            if not line_text:
                continue
            if header_detected_on_page and header_y is not None and y * 6 <= header_y + 20:
                continue
            if footer_patterns.search(line_text):
                footer_y = y * 6
                break
        for y in sorted(lines.keys()):
            if header_detected_on_page and header_y is not None and y * 6 <= header_y + 4:
                continue
            if footer_y is not None and y * 6 >= footer_y - 4:
                continue
            ws = sorted(lines[y], key=lambda x: x["x"])
            line_text = " ".join((w.get("text") or "").strip() for w in ws).strip()
            low = line_text.lower()
            if not line_text:
                continue
            if contains_blocked_keyword(low, BLOCKLIST):
                continue
            if re.search(r"\b(st|street|ave|avenue|rd|road|blvd|suite|city|state|zip)\b", low):
                continue
            tokens = [t for t in re.split(r"\s+", line_text) if t]
            if len(tokens) < 2:
                continue
            item = {}
            total_candidates = []
            for w in ws:
                band_idx = None
                for i in range(len(edges) - 1):
                    if edges[i] <= w["x"] < edges[i+1]:
                        band_idx = i
                        break
                if band_idx is None:
                    band_idx = len(roles) - 1
                role_idx = max(0, min(len(roles) - 1, band_idx))
                role = roles[role_idx]
                if role in BLOCKLIST:
                    continue
                txt = (w.get("text") or "").strip()
                if role in ("quantity", "unit_price", "total_price"):
                    val = safe_float(txt)
                    if val is not None:
                        if role == "total_price":
                            total_candidates.append(val)
                        else:
                            item.setdefault(role, val)
                else:
                    item[role] = (item.get(role, "") + " " + txt).strip()
            if "total_price" not in item and total_candidates:
                item["total_price"] = total_candidates[-1]
            if "total_price" not in item:
                for w in reversed(ws):
                    v = safe_float(w.get("text"))
                    if v is not None and sane_money(v) is not None:
                        item["total_price"] = v
                        break
            q = item.get("quantity")
            u = item.get("unit_price")
            t = item.get("total_price")
            if t is not None:
                if q is None and u is not None and u != 0:
                    inferred_q = round(t / u, 2)
                    if 0 < inferred_q < 1e6:
                        item["quantity"] = inferred_q
                if u is None and q is not None and q != 0:
                    inferred_u = round(t / q, 2)
                    if 0 < inferred_u < 1e7:
                        item["unit_price"] = inferred_u
            desc = (item.get("description") or "").strip()
            if not desc or len(desc) < 2:
                continue
            if not any(k in item for k in ("total_price", "unit_price")):
                continue
            if "total_price" in item and sane_money(item["total_price"]) is None:
                continue
            for k in ("quantity", "unit_price", "total_price"):
                if k in item:
                    try:
                        item[k] = float(item[k])
                    except Exception:
                        item[k] = None
            if item and is_valid_line_item(item):
                items.append(item)
    return items

def normalize_item_math(item):
    q = safe_val(item.get("quantity"))
    u = safe_val(item.get("unit_price"))
    t = safe_val(item.get("total_price"))
    if q and u and t:
        if u > t and q < 10000:
            if abs((q * u) - t) > 1.0 and abs((q * t) - u) < abs((q * u) - t):
                item["quantity"], item["unit_price"] = u, q
                q, u = item["quantity"], item["unit_price"]
        if abs((q * u) - t) < 1.0:
            pass
        else:
            if abs(u - t) < 0.01:
                item["quantity"] = 1.0
            elif abs(q - t) < 0.01:
                item["unit_price"] = 1.0
    if t is not None:
        if q is None and u is not None and u != 0:
            item["quantity"] = round(t / u, 2)
        if u is None and q is not None and q != 0:
            item["unit_price"] = round(t / q, 2)
    item["quantity"] = safe_val(item.get("quantity"))
    item["unit_price"] = safe_val(item.get("unit_price"))
    item["total_price"] = safe_val(item.get("total_price"))
    return item

def extract_items_from_text_lines(text):
    items = []
    BLOCKLIST = {
        "invoice", "date", "due", "vat", "tax", "subtotal",
        "total", "balance", "phone", "fax", "ship", "bill",
        "address", "street"
    }
    for line in text.split("\n"):
        clean = line.strip()
        if not clean:
            continue
        low = clean.lower()
        if any(k in low for k in BLOCKLIST):
            continue
        numbers = re.findall(r"\d+\.\d{2}|\d+", clean)
        if not numbers:
            continue
        has_decimal = bool(re.search(r"\d+\.\d{2}\b", clean))
        has_currency = "$" in clean or "€" in clean or "£" in clean
        if not has_decimal and not has_currency and len(numbers) < 2:
            continue
        desc = re.sub(r"\d+\.\d{2}|\d+", "", clean).strip()
        if len(desc) < 3:
            continue
        if not re.search(r"[A-Za-z]", desc):
            continue
        item = {
            "description": desc,
            "quantity": None,
            "unit_price": None,
            "total_price": safe_float(numbers[-1])
        }
        items.append(item)
    return items

def extract_totals(text):
    totals = {}
    money = r"([\d.,]+)"
    def grab(label):
        if label.lower() == "total":
            label_pat = r"\b(?<!sub\s)(?<!sub)total\b"
        elif label.lower() == "subtotal":
            label_pat = r"\bsub\s*total\b"
        else:
            label_pat = rf"{label}"
        m = re.findall(rf"{label_pat}[:\s]*[^\d\n]*{money}", text, re.I)
        if not m:
            return None
        if isinstance(m[0], tuple):
            vals = [safe_float(x) for tup in m for x in tup if safe_float(x)]
        else:
            vals = [safe_float(x) for x in m if safe_float(x)]
        return max(vals) if vals else None
    subtotal = grab("Subtotal")
    if subtotal is not None: totals["subtotal"] = subtotal
    discount = grab("Discount")
    if discount is not None: totals["discount"] = discount
    shipping = grab("Shipping")
    if shipping is not None: totals["shipping"] = shipping
    tax = grab("Tax|VAT")
    if tax is not None: totals["tax"] = tax
    total = grab("Total")
    if total is not None: totals["invoice_total"] = total
    return totals

def extract_discount(text, totals):
    m = re.search(r"Discount\s*\((\d+(?:\.\d+)?)%\)", text, re.I)
    if m and "subtotal" in totals and totals["subtotal"] is not None:
        pct = float(m.group(1)) / 100.0
        totals["discount"] = round(safe_val(totals["subtotal"]) * pct, 2)
    return totals

def normalize_totals_for_math(totals):
    clean = {}
    for k, v in totals.items():
        if isinstance(v, dict) and "value" in v:
            clean[k] = v["value"]
        elif isinstance(v, (int, float)):
            clean[k] = v
    return clean

def extract_totals_from_layout(words):
    totals = {}
    LABELS = {
        "subtotal": ["subtotal", "sub total"],
        "discount": ["discount"],
        "shipping": ["shipping", "shipping & handling"],
        "tax": ["tax", "vat", "sales tax", "total vat"],
        "invoice_total": ["total", "total amount", "invoice total", "balance due"]
    }
    for w in words:
        w["text_norm"] = w.get("text", "").lower().strip()
    numbers = []
    for w in words:
        val = safe_float(re.sub(r"[^\d,.\-]", "", w.get("text", "")))
        if val is not None and sane_money(val) is not None:
            numbers.append({**w, "value": val})
    def nearest_right(label_word):
        page = label_word["page"]
        y = label_word["y"]
        band_top, band_bot = y - 6, y + 8
        candidates = [
            n for n in numbers if n["page"] == page and band_top <= n["y"] <= band_bot]
        if not candidates:
            return None
        return max(candidates, key=lambda n: n["x"])
    for key, aliases in LABELS.items():
        label_words = [w for w in words if w["text_norm"] in aliases]
        if not label_words:
            continue
        label = max(label_words, key=lambda x: (x["page"], x["y"]))
        best = nearest_right(label)
        if best:
            totals[key] = {"value": best["value"], "source": f"{label['text']} → {best['value']}", "method": "layout"}
    return totals

def resolve_totals_globally(totals, line_items):
    resolved = {k: safe_val(v) for k, v in totals.items()}
    line_sum = round(sum(safe_val(i.get("total_price")) or 0 for i in line_items), 2)
    subtotal = resolved.get("subtotal", line_sum)
    discount = resolved.get("discount", 0.0)
    shipping = resolved.get("shipping", 0.0)
    tax = resolved.get("tax", 0.0)
    recomputed_total = round(subtotal - discount + shipping + tax, 2)
    invoice_total = resolved.get("invoice_total", recomputed_total)
    resolved.update({
        "subtotal": subtotal,
        "discount": discount,
        "shipping": shipping,
        "tax": tax,
        "invoice_total": invoice_total
    })
    return resolved

def validate_invoice(items, totals):
    if not items:
        return "⚠ No line items detected"
    return f"✅ {len(items)} line items detected"

def validate_layout_totals(totals, line_items, tolerance=1.5):
    if not totals or not line_items:
        return totals
    line_sum = sum(
        safe_val(i.get("total_price"))
        for i in line_items
        if i.get("total_price") is not None
    )
    inv_total = safe_val(totals.get("invoice_total"))
    sub_total = safe_val(totals.get("subtotal"))
    if inv_total is not None and abs(inv_total - line_sum) > tolerance:
        totals["invoice_total_mismatch"] = True
    if sub_total is not None and abs(sub_total - line_sum) > tolerance:
        totals["subtotal_mismatch"] = True
    return totals

def normalize_items(items):
    out = []
    seen = {}
    for it in items:
        desc = (it.get("description") or "").strip()
        key = (desc.lower(), round(safe_val(it.get("total_price") or 0), 2))
        if key in seen:
            idx = seen[key]
            existing = out[idx]
            try:
                if existing.get("quantity") is not None and it.get("quantity") is not None:
                    existing["quantity"] = float(existing["quantity"]) + float(it["quantity"])
                elif existing.get("quantity") is None and it.get("quantity") is not None:
                    existing["quantity"] = float(it["quantity"])
            except Exception:
                pass
            try:
                if existing.get("total_price") is not None and it.get("total_price") is not None:
                    existing["total_price"] = float(existing["total_price"]) + float(it["total_price"])
            except Exception:
                pass
        else:
            seen[key] = len(out)
            out.append(it.copy())
    return [normalize_item_math(i) for i in out]

def infer_totals_from_items(line_items, totals, tolerance=0.01):
    if not line_items:
        return totals
    line_sum = round(
        sum(
            i["total_price"]
            for i in line_items
            if i.get("total_price") is not None
        ),
        2
    )
    if "subtotal" not in totals:
        totals["subtotal"] = line_sum
        totals["_subtotal_inferred"] = True
    if any(k in totals for k in ("discount", "shipping", "tax")):
        return totals
    if "invoice_total" not in totals:
        totals["invoice_total"] = line_sum
        totals["_invoice_total_inferred"] = True
    return totals

def detect_company(words, text):
    KEYWORDS = {"invoice","bill","ship","date","subtotal","total","discount",
                "balance","amount","notes","terms","order","mode","payment"}
    MONTHS = {"jan","feb","mar","apr","may","jun","jul","aug","sep","sept","oct","nov","dec"}
    SUFFIXES = {"inc", "ltd", "llc", "gmbh", "srl", "bv", "oy", "sa", "sas", "spa", "pte", "plc"}
    candidates = []
    for w in words:
        if w.get("page", 0) != 0:
            continue
        clean = (w.get("text") or "").strip()
        if not clean or len(clean) > 60:
            continue
        low = clean.lower()
        if any(k in low for k in KEYWORDS) or low in MONTHS:
            continue
        if clean.isdigit() or clean.lower() in {"from","to"}:
            continue
        if clean.isdigit():
            continue
        if "," in clean:
            continue
        if re.fullmatch(r"[A-Za-z0-9&.\- ]{2,}", clean):
            tokens = {t.strip(".") for t in low.split()}
            if tokens & SUFFIXES or len(clean) >= 2:
                candidates.append((w["y"], clean))
    if candidates:
        candidates.sort(key=lambda t: t[0])
        return candidates[0][1]
    for l in [l.strip() for l in text.splitlines() if l.strip()][:10]:
        low = l.lower()
        if any(k in low for k in KEYWORDS) or low in MONTHS or "," in l or l.isdigit():
            continue
        if re.fullmatch(r"[A-Za-z0-9&.\- ]{2,}", l):
            return l
    return None

def extract_metadata(text, file_name, words):
    company = detect_company(words, text)
    ship_to = None
    invoice_date = None
    buyer = None
    header = extract_header_fields(text)
    invoice_number = header.get("invoice_number")
    def _clean_party(s):
        if not s:
            return None
        s = re.sub(r"\s+", " ", s).strip()
        m = re.match(r"^(.+)\s+\1$", s, re.I)
        if m:
            s = m.group(1).strip()
        if company and company.lower() in s.lower():
            s = re.sub(re.escape(company), "", s, flags=re.I).strip()
        suffixes = {"inc","ltd","llc","gmbh","srl","bv","oy","sa","sas","spa","pte","plc","corp","co"}
        tokens = s.split()
        last_idx = None
        for i, t in enumerate(tokens):
            if t.strip(".").lower() in suffixes:
                last_idx = i
        if last_idx is not None and last_idx > 0:
            s = " ".join(tokens[last_idx-1:last_idx+1])
        return s or None
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    def next_meaningful_line(start_idx):
        for j in range(start_idx, min(start_idx + 6, len(lines))):
            if re.search(r"\b(invoice|date|terms|total|tax|vat|order)\b", lines[j], re.I):
                continue
            if re.search(r"^\d{1,2}/\d{1,2}/\d{2,4}$", lines[j]):
                continue
            return lines[j].strip()
        return None
    m = re.search(r"(Ship To|Bill To)\s*:\s*([^\n]+)", text, re.I)
    if m:
        buyer = m.group(2).strip()
    else:
        for i, line in enumerate(lines):
            if re.search(r"\bBILL\s*FROM\b.*\bBILL\s*TO\b", line, re.I):
                candidate = next_meaningful_line(i + 1)
                if candidate:
                    parts = re.split(r"\s{2,}", candidate)
                    if len(parts) >= 2:
                        buyer = parts[-1].strip()
                    else:
                        buyer = candidate
                break
        if not buyer:
            for i, line in enumerate(lines):
                if re.search(r"\b(DELIVERY|SHIPPING|INVOICE)\s+ADDRESS\b", line, re.I):
                    buyer = next_meaningful_line(i + 1)
                    break
        if not buyer:
            for i, line in enumerate(lines):
                if re.fullmatch(r"INVOICED TO", line, re.I):
                    candidates = []
                    for j in range(i + 1, min(i + 5, len(lines))):
                        if re.search(r"\b(invoice|date|terms|total|tax|vat|order)\b", lines[j], re.I):
                            continue
                        if re.search(r"^\d{1,2}/\d{1,2}/\d{2,4}$", lines[j]):
                            continue
                        candidates.append(lines[j].strip())
                    if candidates:
                        suffix_re = re.compile(r"\b(inc|llc|ltd|corp|co|gmbh|srl|bv|oy|sa|sas|spa|pte|plc)\b", re.I)
                        best = None
                        for c in candidates:
                            if "c/o" in c.lower() or suffix_re.search(c):
                                best = c
                                break
                        buyer = best or candidates[0]
                    break
        if not buyer:
            for i, line in enumerate(lines):
                if re.fullmatch(r"(BILL TO|SHIP TO)", line, re.I):
                    buyer = next_meaningful_line(i + 1)
                    break
    ship_to = _clean_party(buyer)
    if "invoice_date" in header:
        invoice_date = header["invoice_date"]
    language = detect_language_safe(words)
    currency = detect_currency(text)
    if not company: #or company.lower() in MONTHS:
        company = "UNKNOWN"
    if invoice_number and invoice_number.lower() == "bill":
        invoice_number = "UNKNOWN"
    return {
        "file": file_name,
        "language": language,
        "currency": currency,
        "company": company,
        "invoice_number": invoice_number,
        "ship_to": ship_to,
        "invoice_date": invoice_date
    }

def print_audit_report(invoices):
    print("========== INVOICE AUDIT REPORT ==========")
    for inv in invoices:
        if "error" in inv:
            print(f"\nInvoice: {inv.get('file','UNKNOWN')} failed -> {inv.get('error')}")
            continue
        raw_text = inv.get("raw_text", "")
        file_name = inv.get("file", "UNKNOWN")
        words = inv.get("words", [])
        meta = extract_metadata(raw_text, file_name, words)
        print(f"\n------------ Invoice: {meta['file']} ------------")
        print(f"Company      : {meta.get('company') or 'UNKNOWN'}")
        print(f"Invoice No   : {meta.get('invoice_number') or 'UNKNOWN'}")
        print(f"Date         : {meta.get('invoice_date') or 'UNKNOWN'}")
        print(f"Currency     : {meta.get('currency') or 'UNKNOWN'}")
        print(f"Language     : {meta.get('language') or 'en'}")
        print(f"Ship To      : {meta.get('ship_to') or ''}")
        print(f"VAT No       : {meta.get('vat') or 'N/A'}")
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

def process_invoice(path: str):
    words = extract_words_with_layout(path)
    raw_text = reconstruct_text(words)
    lang = detect_language_safe(words)
    if ENABLE_TRANSLATION and lang != "en":
        words = translate_words(words, lang)
    id_items = extract_items_from_id_table(raw_text)
    if len(id_items) >= 3:
        line_items = id_items
    else:
        line_items = []
        extracted = extract_items_by_columns(words) or []
        extracted = [normalize_item_math(i) for i in extracted]
        try:
            extracted = normalize_items(extracted)
        except NameError:
            pass
        line_items.extend(extracted)
        if id_items:
            line_items.extend(id_items)
        if not line_items:
            line_items.extend(extract_items_from_pdf_text(raw_text))
        if not line_items:
            line_items.extend(extract_items_from_structured_pdf(raw_text))
        if len(line_items) < 5:
            line_items.extend(extract_items_from_text_lines(raw_text))
    line_items = [normalize_item_math(i) for i in normalize_items(line_items)]
    line_items = [i for i in line_items if is_valid_line_item(i)]
    line_items = [i for i in line_items if is_valid_line_item(i)]
    totals = extract_totals(raw_text)
    totals = extract_discount(raw_text, totals)
    layout_totals = extract_totals_from_layout(words)
    totals = infer_totals_from_items(line_items, totals)
    totals.update(layout_totals)
    totals = validate_layout_totals(totals, line_items)
    totals = resolve_totals_globally(totals, line_items)
    meta = extract_metadata(raw_text, os.path.basename(path), words)
    result = {
        "file": os.path.basename(path),
        "raw_text": raw_text,
        "words": words,
        "line_items": line_items,
        "totals": totals,
        "meta": meta
    }
    return result

if __name__ == "__main__":
    pdf_dir = r"D:\Invoice_Extractor\sample-documents\20-percent-tax-invoices"
    scanned_results = process_invoice_dir_pdf(
        file_dir=pdf_dir,
        limit=5)
    print_audit_report(scanned_results)
