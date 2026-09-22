import io
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger("tempo.timetable.ocr")

# ---------------------------------------------------------------------------
# Day-of-week recognition
# ---------------------------------------------------------------------------
DAY_PATTERNS = {
    "MONDAY": "MONDAY", "MON": "MONDAY",
    "TUESDAY": "TUESDAY", "TUES": "TUESDAY", "TUE": "TUESDAY",
    "WEDNESDAY": "WEDNESDAY", "WED": "WEDNESDAY",
    "THURSDAY": "THURSDAY", "THURS": "THURSDAY", "THUR": "THURSDAY", "THU": "THURSDAY",
    "FRIDAY": "FRIDAY", "FRI": "FRIDAY",
    "SATURDAY": "SATURDAY", "SAT": "SATURDAY",
    "SUNDAY": "SUNDAY", "SUN": "SUNDAY",
}

# Time range patterns, e.g. 09:00-10:00, 9.00 to 10.00 AM, 09:00 — 10:00
TIME_RANGE_REGEX = re.compile(
    r"(\d{1,2})[:.](\d{2})\s*(AM|PM)?\s*(?:-|TO|—|–)\s*(\d{1,2})[:.](\d{2})\s*(AM|PM)?",
    re.IGNORECASE,
)

# Single time with optional meridian, e.g. 10:30 AM
TIME_SINGLE_REGEX = re.compile(r"\b(\d{1,2})[:.](\d{2})\s*(AM|PM)?\b", re.IGNORECASE)

# Period label, e.g. "Period 3", "PERIOD-3", "P3"
PERIOD_REGEX = re.compile(r"\bP(?:ERIOD)?\s*[-#]?\s*(\d{1,2})\b", re.IGNORECASE)

# KLU-style course codes: e.g. 24S12, 23CS301, CSE301, 2-PG-SDA, 3-CSE-DL
KLU_COMPACT_CODE_REGEX = re.compile(r"\b(\d{2}[A-Z]{1,4}\d{1,3})\b")
CLASSIC_COURSE_CODE_REGEX = re.compile(
    r"\b([0-9]{2}[A-Z]{2,4}[0-9]{3,4}[A-Z]?|[A-Z]{2,5}[-\s]?[0-9]{3,4}[A-Z]?)\b"
)
# Batch/slot-style course code with hyphens, e.g. 2-PG-SDA, 3-CSE-DL, 2-PG-SDA-A
KLU_BATCH_CODE_REGEX = re.compile(
    r"\b(\d{1,2}\s*-\s*[A-Z0-9]{1,6}\s*-\s*[A-Z0-9]{1,6}(?:\s*-\s*[A-Z0-9]{1,6})?)\b"
)

# OCR-engine character confusions: a letter that is visually part of the course code but
# separated by a space (e.g. "C SE301" read from "CSE301"). Group 1 = letter prefix,
# group 2 = optional digit group, group 3 = letter body, group 4 = numeric suffix;
# the fragments are joined back into a single course code.
OCR_CODE_JOIN_REGEX = re.compile(r"(?:^|\s)([A-Z])\s?(\d{0,2})\s?([A-Z]{2,4})\s?(\d{3,4})(?:\s|$)")

# aSc-style period header, e.g. "1 (09-10)", "2 (10-11)", "8 (04-05)" — period number
# followed by an hour-only range. Hours are disambiguated to 24h with an academic-day
# bias (8-11 = AM, 12 = noon, 1-7 = PM).
PERIOD_HEADER_REGEX = re.compile(
    r"(?<![\dA-Za-z])(\d{1,2})\s*\(\s*(\d{1,2})\s*[-–—]\s*(\d{1,2})\s*\)(?![\dA-Za-z])"
)

# Page-instrumentation text that must never become a timetable slot: institutional
# headers, faculty-name lines, watermark/footer lines and "generator" labels.
ARTIFACT_TEXT_REGEX = re.compile(
    r"(aSc\s*Timetables|Timetables?\s*Generator|SCHOOL\s+OF\s+COMPUTING|"
    r"DEPARTMENT\s+OF\s+CSE|DEPT\.?\s+OF\s+CSE|"  # institution header lines
    r"^CSE$)",
    re.IGNORECASE,
)


def _join_ocr_code(text: str) -> Optional[str]:
    """Recovers a course code split by OCR letter/digit confusions. Returns None if absent."""
    m = OCR_CODE_JOIN_REGEX.search(text.upper())
    if not m:
        return None
    candidate = "".join(g for g in m.groups() if g)
    # Must look like a real code: at least 5 chars with both letters and digits
    if len(candidate) < 5 or not re.search(r"[A-Z]", candidate) or not re.search(r"\d", candidate):
        return None
    return candidate

# Section patterns: PG2, Sec-A, Section 3, Batch 2, S12, 3-CSE-DL (batch group)
SECTION_EXPLICIT_REGEX = re.compile(
    r"\b(?:SEC(?:TION)?|BATCH|GRP|GROUP|SET)\s*[-:#]?\s*([A-Za-z0-9]{1,6})\b", re.IGNORECASE
)
SECTION_COMPACT_REGEX = re.compile(r"\b(PG|UG|S)\s*[-]?\s*(\d{1,3}[A-Za-z]?)\b", re.IGNORECASE)

# Room patterns (optional metadata; prototype timetable does not require rooms)
ROOM_REGEX = re.compile(
    r"\b(?:ROOM|R|LAB|LH|HALL|BLOCK)\s*[-:#]?\s*([A-Z0-9-]{1,8})\b", re.IGNORECASE
)

# Bare room tokens as printed in aSc/KARE timetable cells (no keyword prefix):
#   "8607" (all digits), "PG-8509B" (letters-digits-letters), "Lab8301B" (lab + digits).
BARE_ROOM_REGEX = re.compile(r"^\d{3,5}$")
# Keyword-prefixed rooms only: Lab8301B, LH402, RM12, BLOCK-C. Deliberately does NOT
# match bare letter+digit tokens so course codes like CSE301 stay course codes.
PREFIXED_ROOM_REGEX = re.compile(
    r"^(?:(?:LAB|LH|RM|ROOM|BLOCK)[-\s]?)(?:\d{1,5}|[A-Z]-?\d{1,5})[A-Z]?$", re.IGNORECASE
)
# Distinguishes room tokens like PG-8509B from batch codes: rooms do not start
# with a digit group followed by hyphenated letter groups.
ROOM_HYPHEN_REGEX = re.compile(r"^[A-Z]{2,4}-\d{3,5}[A-Z]?$", re.IGNORECASE)

# Tokens that mark a non-teaching cell
NON_TEACHING_TOKENS = {"LUNCH", "BREAK", "FREE", "NIL", "NULL", "NONE", "-", "--", "N/A", "NA", "TEA"}

# Table header artifacts (column titles), never timetable entries
HEADER_TOKENS = {
    "DAY", "DAYS", "PERIOD", "PERIODS", "TIME", "TIMES", "TIMINGS", "HOURS",
    "TABLE", "TIMETABLE", "SCHEDULE", "DATE", "CLASS", "SEC", "SECTION",
}

DAY_COLUMN_ORDER = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY"]


def _clean_cell(text: Optional[str]) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def _is_non_teaching(text: str) -> bool:
    upper = text.strip().upper()
    return upper in NON_TEACHING_TOKENS or upper in HEADER_TOKENS or len(upper) < 2


class TimetableOCRService:
    """
    OCR & table extraction service for faculty timetable images and PDFs.

    Extracts: Day, Period, Start Time, End Time, Course Code/Subject, Section/Batch.

    Faculty (KLU) naming convention for cell values:
      - Batch-style tokens (3-CSE-DL, 2-PG-SDA) and PG/UG tokens (PG2) are the
        SUBJECT identity (returned as subject_name; subject_code stays empty).
      - Compact codes (24S12) are SECTION/BATCH identifiers.
    When a cell contains both, the fields are assigned accordingly.

    IMPORTANT: This service never invents timetable data. Fields that cannot be
    read from the document are returned as empty strings and must be completed by
    the faculty during the confirmation/editing step. If nothing can be extracted,
    an EMPTY slot list is returned - never a fabricated template.
    """

    @classmethod
    def extract_from_file_bytes(cls, file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        lower_name = (filename or "").lower()
        if lower_name.endswith(".pdf"):
            slots = cls._extract_from_pdf(file_bytes)
        else:
            slots = cls._extract_from_image(file_bytes)
        return cls._normalize_and_deduplicate(slots)

    # ------------------------------------------------------------------
    # PDF extraction
    # ------------------------------------------------------------------
    @classmethod
    def _extract_from_pdf(cls, file_bytes: bytes) -> List[Dict[str, Any]]:
        slots: List[Dict[str, Any]] = []
        try:
            import pdfplumber

            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    tables = page.extract_tables()
                    for table in tables:
                        slots.extend(cls._parse_table_grid(table))
                    if not slots:
                        text = page.extract_text() or ""
                        slots.extend(cls._parse_raw_text(text))
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}. Trying rasterized image parsing.")
            slots = cls._extract_pdf_as_image(file_bytes)
        return slots

    @classmethod
    def _extract_pdf_as_image(cls, file_bytes: bytes) -> List[Dict[str, Any]]:
        try:
            import pypdfium2 as pdfium

            pdf = pdfium.PdfDocument(file_bytes)
            if len(pdf) == 0:
                return []
            page = pdf[0]
            pil_image = page.render(scale=2).to_pil()
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            return cls._extract_from_image(buf.getvalue())
        except Exception as e:
            logger.error(f"PDF image fallback failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Image extraction
    # ------------------------------------------------------------------
    @classmethod
    def _extract_from_image(cls, file_bytes: bytes) -> List[Dict[str, Any]]:
        try:
            import cv2
            import numpy as np

            np_arr = np.frombuffer(file_bytes, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if image is None:
                logger.error("Uploaded image could not be decoded.")
                return []
        except Exception as e:
            logger.error(f"Image decoding failed: {e}")
            return []

        # 1. RapidOCR (pure-pip ONNX OCR engine, bundles its own models) - primary engine.
        #    Returns per-text-box results with pixel positions, enabling timetable grid
        #    reconstruction (day rows x time columns).
        ocr_items: List[Dict[str, Any]] = []  # [{"text", "x", "y", "conf"}]
        try:
            from rapidocr_onnxruntime import RapidOCR

            engine = RapidOCR()
            result, _ = engine(image)
            if result:
                for box, text, conf in result:
                    xs = [p[0] for p in box]
                    ys = [p[1] for p in box]
                    ocr_items.append({
                        "text": str(text).strip(),
                        "x": float(min(xs)),
                        "x1": float(max(xs)),
                        "y": float(min(ys)),
                        "conf": float(conf),
                    })
        except Exception as ocr_err:
            logger.warning(f"RapidOCR unavailable or failed: {ocr_err}")

        # 2. pytesseract fallback (if installed and configured)
        if not ocr_items:
            try:
                import pytesseract

                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                ocr_text = pytesseract.image_to_string(thresh)
                lines = [l.strip() for l in ocr_text.splitlines() if l.strip()]
                ocr_items = [{"text": l, "x": 0.0, "y": float(i), "conf": 0.5} for i, l in enumerate(lines)]
            except Exception as ocr_err:
                logger.warning(f"pytesseract unavailable or failed: {ocr_err}")

        if ocr_items:
            slots = cls._parse_ocr_items(ocr_items)
            if slots:
                return slots

        # 3. No OCR engine available (or it found nothing): the caller must be told that
        #    manual entry is required. We NEVER generate a fake template.
        logger.warning(
            "No OCR engine available or no text recognized; returning empty extraction. "
            "Faculty must enter timetable entries manually on the confirmation page."
        )
        return []

    # ------------------------------------------------------------------
    # Grid / table parsing
    # ------------------------------------------------------------------
    @classmethod
    def _parse_table_grid(cls, table: List[List[Optional[str]]]) -> List[Dict[str, Any]]:
        """Parses a 2D table: header row with times, first column with days."""
        slots: List[Dict[str, Any]] = []
        if not table or len(table) < 2:
            return []

        header_row = [_clean_cell(c) for c in table[0]]
        time_cols: Dict[int, Dict[str, Any]] = {}

        for col_idx, col_val in enumerate(header_row):
            if col_idx == 0:
                continue
            entry: Dict[str, Any] = {"label": col_val, "start_time": "", "end_time": ""}
            time_match = TIME_RANGE_REGEX.search(col_val)
            if time_match:
                entry["start_time"] = cls._format_time(time_match.group(1), time_match.group(2), time_match.group(3))
                entry["end_time"] = cls._format_time(time_match.group(4), time_match.group(5), time_match.group(6))
            else:
                period_match = PERIOD_REGEX.search(col_val)
                if period_match:
                    entry["period_number"] = int(period_match.group(1))
            time_cols[col_idx] = entry

        # Rows: first column may hold the day
        for row in table[1:]:
            if not row:
                continue
            first_cell = _clean_cell(row[0]).upper()
            day_match = cls._match_day(first_cell)
            if not day_match:
                # Some layouts embed the day in every cell; try each cell individually
                for col_idx, cell in enumerate(row):
                    cell_text = _clean_cell(cell)
                    if not cell_text or _is_non_teaching(cell_text):
                        continue
                    info = cls._parse_cell_content(cell_text)
                    if info["subject_code"] or info["section_name"]:
                        col_entry = time_cols.get(col_idx, {})
                        slots.append(cls._make_slot(
                            day_of_week="",  # unknown -> faculty must assign
                            period_label=col_entry.get("label", ""),
                            start_time=col_entry.get("start_time", ""),
                            end_time=col_entry.get("end_time", ""),
                            cell_info=info,
                        ))
                continue

            for col_idx, cell in enumerate(row[1:], start=1):
                cell_text = _clean_cell(cell)
                if not cell_text or _is_non_teaching(cell_text):
                    continue
                col_entry = time_cols.get(col_idx, {})
                info = cls._parse_cell_content(cell_text)
                slots.append(cls._make_slot(
                    day_of_week=day_match,
                    period_label=col_entry.get("label", ""),
                    start_time=col_entry.get("start_time", ""),
                    end_time=col_entry.get("end_time", ""),
                    cell_info=info,
                    period_number=col_entry.get("period_number"),
                ))
        return slots

    # ------------------------------------------------------------------
    # OCR item parsing with grid reconstruction
    # ------------------------------------------------------------------
    @classmethod
    def _parse_ocr_items(cls, ocr_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Parses positioned OCR text boxes into timetable slots.

        With real pixel positions (RapidOCR) the timetable grid is reconstructed:
        period-header columns ("1 (09-10)" or "09:00-10:00") x day row bands, with
        every cell's stacked fragments (room / code / batch) merged into one slot.
        Without reliable positions (pytesseract line fallback) a row-wise parser is
        used instead. Unreadable fields stay empty - nothing is invented.
        """
        items = [i for i in ocr_items if i.get("text")]
        if not items:
            return []

        has_positions = any(float(i.get("x", 0.0)) > 1.0 for i in items)
        day_label_count = sum(
            1 for i in items if len(i["text"].strip()) <= 12 and cls._match_day(i["text"])
        )
        period_header_count = sum(1 for i in items if PERIOD_HEADER_REGEX.search(i["text"]))
        if has_positions and (day_label_count >= 1 or period_header_count >= 2):
            return cls._parse_ocr_items_grid(items)
        return cls._parse_ocr_items_rowwise(items)

    # ------------------------------------------------------------------
    # Grid reconstruction (aSc/KARE style and generic grids)
    # ------------------------------------------------------------------
    @staticmethod
    def _is_compact_only_code(code: str) -> bool:
        """True when a code is a bare compact token (24S12) - which is a
        section/batch identifier per the faculty convention, never a course code
        that should block merging adjacent double-period cells."""
        return bool(code) and bool(KLU_COMPACT_CODE_REGEX.fullmatch(code))

    @classmethod
    def _parse_ocr_items_grid(cls, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Reconstructs the timetable grid from positioned OCR fragments:
          1. Period columns from headers ("1 (09-10)" hour-ranges or "09:00-10:00").
          2. Day row bands from the y positions of the day labels.
          3. Every other fragment is assigned to its (day band x column) cell;
             fragments straddling two columns (merged double-period cells) are
             added to both.
          4. Each cell's fragments are merged and parsed once. Page titles, faculty
             names and footers fall outside all day bands (or outside the grid) and
             never become slots.
        """
        img_h = max(i["y"] for i in items) + 80.0

        # ---- 1. Period columns ---------------------------------------------
        headers: List[Dict[str, Any]] = []
        for it in items:
            t = it["text"]
            pm = PERIOD_HEADER_REGEX.search(t)
            if pm:
                start = cls._academic_hour_to_time(int(pm.group(2)))
                end = cls._academic_hour_to_time(int(pm.group(3)))
                if start and end and end > start:
                    headers.append({
                        "x0": it["x"], "x1": cls._item_x1(it), "y": it["y"],
                        "period": int(pm.group(1)), "label": f"Period {pm.group(1)}",
                        "start": start, "end": end,
                    })
                    continue
            tm = TIME_RANGE_REGEX.search(t)
            if tm and len(t) <= 40:
                start = cls._format_time(tm.group(1), tm.group(2), tm.group(3))
                end = cls._format_time(tm.group(4), tm.group(5), tm.group(6))
                if start and end and end > start:
                    headers.append({
                        "x0": it["x"], "x1": cls._item_x1(it), "y": it["y"],
                        "period": None, "label": t, "start": start, "end": end,
                    })

        headers.sort(key=lambda h: h["x0"])
        columns: List[Dict[str, Any]] = []
        for h in headers:
            if columns and h["x0"] < columns[-1]["x1"] * 0.7:
                continue  # duplicate read of the same header
            columns.append(h)

        # ---- 2. Day bands ----------------------------------------------------
        day_labels = sorted(
            (
                {"day": cls._match_day(i["text"]), "y": i["y"]}
                for i in items
                if len(i["text"].strip()) <= 12 and cls._match_day(i["text"])
            ),
            key=lambda d: d["y"],
        )
        header_bottom = max((h["y"] for h in headers), default=0.0)
        bands: List[Dict[str, Any]] = []
        for idx, d in enumerate(day_labels):
            if idx == 0:
                band_start = (header_bottom + d["y"]) / 2.0 if headers else 0.0
            else:
                band_start = (day_labels[idx - 1]["y"] + d["y"]) / 2.0
            band_end = img_h if idx == len(day_labels) - 1 else (d["y"] + day_labels[idx + 1]["y"]) / 2.0
            bands.append({"day": d["day"], "start": band_start, "end": band_end})

        # ---- 3. Assign fragments to (day, column) cells ----------------------
        cells: Dict[tuple, Dict[str, Any]] = {}
        for it in items:
            text = it["text"].strip()
            if not text or _is_non_teaching(text):
                continue
            if len(text) <= 12 and cls._match_day(text):
                continue
            if ARTIFACT_TEXT_REGEX.search(text):
                continue
            if PERIOD_HEADER_REGEX.search(text) or (TIME_RANGE_REGEX.search(text) and len(text) <= 40):
                continue

            band = next((b for b in bands if b["start"] <= it["y"] < b["end"]), None)
            if band is None:
                continue  # page title / faculty name / footer

            x0 = float(it["x"])
            x1 = cls._item_x1(it)
            width = max(x1 - x0, 20.0)
            matched_cols: List[Optional[Dict[str, Any]]] = [
                c for c in columns if (min(x1, c["x1"]) - max(x0, c["x0"])) >= 0.3 * width
            ]
            if not matched_cols:
                matched_cols = [None]

            for col in matched_cols:
                key = (band["day"], id(col) if col else None)
                if key not in cells:
                    cells[key] = {"day": band["day"], "col": col, "texts": []}
                cells[key]["texts"].append((it["y"], text))

        # ---- 4. Merge adjacent compatible cells, then parse once -------------
        # aSc-style double periods split their fragments across two header columns
        # (code+batch on the left, room on the right). Adjacent same-day cells are
        # merged when code-compatible (equal codes, or exactly one side has a code),
        # then the merged fragments are parsed as ONE cell so fields combine.
        by_day: Dict[str, List[Dict[str, Any]]] = {}
        for cell in cells.values():
            by_day.setdefault(cell["day"], []).append(cell)

        slots: List[Dict[str, Any]] = []
        for day, day_cells in by_day.items():
            pre_parsed: List[Dict[str, Any]] = []
            for cell in day_cells:
                texts = [t for _, t in sorted(cell["texts"])]
                combined = " ".join(texts)
                if ARTIFACT_TEXT_REGEX.search(combined):
                    continue
                probe = cls._parse_cell_content(combined)
                has_core = bool(probe["subject_code"] or probe["subject_name"] or probe["section_name"])
                # Room-only cells (a double period's room fragment lands in its own
                # column) are kept so they merge into the adjacent coded cell.
                if not (has_core or probe["room_number"] or (probe["subject_name"] and cell["col"] is not None)):
                    continue
                pre_parsed.append({
                    "cell": cell,
                    "texts": texts,
                    # Merge identity uses the raw token (pre-demotion): two KLU
                    # cells with different subjects must never merge just because
                    # subject_code was moved to subject_name.
                    "code": probe.get("raw_code") or probe["subject_code"],
                    "period": (cell["col"] or {}).get("period"),
                })
            pre_parsed.sort(key=lambda c: c["period"] if c["period"] is not None else 999)

            groups: List[List[Dict[str, Any]]] = []
            for cand in pre_parsed:
                if (
                    groups
                    and cand["cell"]["col"] is not None
                    and groups[-1][-1]["cell"]["col"] is not None
                    and cand["period"] is not None
                    and groups[-1][-1]["period"] is not None
                    and cand["period"] == groups[-1][-1]["period"] + 1
                ):
                    prev_code = groups[-1][-1]["code"]
                    cand_code = cand["code"]
                    # Compact-only tokens (24S12) are sections, not codes: they
                    # never block a double-period merge.
                    if cls._is_compact_only_code(prev_code):
                        prev_code = ""
                    if cls._is_compact_only_code(cand_code):
                        cand_code = ""
                    if not prev_code or not cand_code or prev_code == cand_code:
                        groups[-1].append(cand)
                        continue
                groups.append([cand])

            for group in groups:
                combined = " ".join(t for c in group for t in c["texts"])
                info = cls._parse_cell_content(combined)
                # A standalone group that yielded only a room is a page fragment,
                # not a lesson - never emit it as a slot.
                if not (info["subject_code"] or info["subject_name"] or info["section_name"]):
                    continue
                col_first = group[0]["cell"]["col"]
                col_last = group[-1]["cell"]["col"]
                if col_first is None:
                    label = ""
                    start_time = ""
                    end_time = ""
                    period_number = None
                elif len(group) > 1:
                    label = f"Periods {col_first['period']}-{col_last['period']}"
                    start_time = col_first["start"]
                    end_time = col_last["end"]
                    period_number = None
                else:
                    label = col_first["label"]
                    start_time = col_first["start"]
                    end_time = col_first["end"]
                    period_number = col_first["period"]
                slots.append(cls._make_slot(
                    day_of_week=day,
                    period_label=label,
                    start_time=start_time,
                    end_time=end_time,
                    cell_info=info,
                    period_number=period_number,
                ))
        return slots

    @staticmethod
    def _item_x1(it: Dict[str, Any]) -> float:
        """Right edge of an OCR box (real when available, estimated otherwise)."""
        x1 = it.get("x1")
        if x1:
            return float(x1)
        return float(it["x"]) + max(len(it["text"]) * 11.0, 30.0)

    @staticmethod
    def _academic_hour_to_time(hour: int) -> str:
        """Disambiguates an hour-only period time (aSc headers like '5 (01-02)')
        using an academic-day bias: 8-11 => AM, 12 => noon, 1-7 => PM.
        So '1 (09-10)' is 09:00-10:00 and '5 (01-02)' is 13:00-14:00.
        """
        if hour < 0 or hour > 23:
            return ""
        if 8 <= hour <= 11 or hour == 12:
            h24 = hour
        elif 1 <= hour <= 7:
            h24 = hour + 12
        else:
            h24 = hour
        return f"{h24:02d}:00"

    # ------------------------------------------------------------------
    # Row-wise fallback (line-based OCR without reliable x positions)
    # ------------------------------------------------------------------
    @classmethod
    def _parse_ocr_items_rowwise(cls, ocr_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not ocr_items:
            return []

        # Drop page artifacts and period-header fragments before row clustering
        items = [
            i for i in ocr_items
            if i.get("text")
            and not ARTIFACT_TEXT_REGEX.search(i["text"])
            and not PERIOD_HEADER_REGEX.search(i["text"])
        ]
        if not items:
            return []

        # Row clustering tolerance (y-axis). OCR boxes are positioned by top-left corner.
        y_tol = 26.0

        # Sort by y then x, cluster into rows
        items_sorted = sorted(items, key=lambda i: (i["y"], i["x"]))
        rows: List[List[Dict[str, Any]]] = []
        current_row: List[Dict[str, Any]] = []
        current_y: Optional[float] = None
        for it in items_sorted:
            if current_y is None or abs(it["y"] - current_y) <= y_tol:
                current_row.append(it)
                current_y = it["y"] if current_y is None else current_y
            else:
                rows.append(current_row)
                current_row = [it]
                current_y = it["y"]
        if current_row:
            rows.append(current_row)

        # Detect time-range column headers: cells whose text is predominantly a time range
        # Map column center-x -> (start_time, end_time, label)
        columns: List[Dict[str, Any]] = []
        for row in rows:
            for it in row:
                text = it["text"]
                tm = TIME_RANGE_REGEX.search(text)
                if tm and len(text) <= 40:
                    start = cls._format_time(tm.group(1), tm.group(2), tm.group(3))
                    end = cls._format_time(tm.group(4), tm.group(5), tm.group(6))
                    if start and end:
                        center_x = it["x"] + (len(text) * 8) / 2.0
                        columns.append({"x": it["x"], "center_x": center_x, "start": start, "end": end, "label": text})

        # Deduplicate columns by approximate x
        deduped_columns: List[Dict[str, Any]] = []
        for col in columns:
            if not any(abs(col["x"] - c["x"]) < 40 for c in deduped_columns):
                deduped_columns.append(col)
        columns = deduped_columns

        slots: List[Dict[str, Any]] = []
        current_day: Optional[str] = None

        for row in rows:
            row_text = " ".join(i["text"] for i in row)
            day_in_row = cls._match_day(row_text)

            # Row 0/1 with a day AND a course code: treat the whole row as a course line
            has_course = bool(KLU_COMPACT_CODE_REGEX.search(row_text) or KLU_BATCH_CODE_REGEX.search(row_text) or CLASSIC_COURSE_CODE_REGEX.search(row_text))

            if day_in_row and not has_course:
                current_day = day_in_row
                continue

            if day_in_row:
                current_day = day_in_row

            # Assign each cell to nearest time column by x-position
            for it in row:
                text = it["text"]
                if not text or _is_non_teaching(text):
                    continue
                if TIME_RANGE_REGEX.search(text) and len(text) <= 40:
                    # Column header itself
                    continue
                if cls._match_day(text) and len(text) <= 12:
                    continue  # day cell itself

                info = cls._parse_cell_content(text)
                if not (info["subject_code"] or info["subject_name"] or info["section_name"]):
                    continue

                # Find nearest column strictly by x overlap / distance
                col = None
                best_dist = float("inf")
                for c in columns:
                    dist = abs(it["x"] - c["x"])
                    if dist < best_dist:
                        best_dist = dist
                        col = c

                # Only assign a column when reasonably aligned (< 350px)
                start_time = col["start"] if (col and best_dist < 350) else ""
                end_time = col["end"] if (col and best_dist < 350) else ""

                slots.append(cls._make_slot(
                    day_of_week=current_day or "",
                    period_label=col["label"] if (col and best_dist < 350) else "",
                    start_time=start_time,
                    end_time=end_time,
                    cell_info=info,
                ))

        return slots

    # ------------------------------------------------------------------
    # Raw text parsing (OCR output / PDF text layer)
    # ------------------------------------------------------------------
    @classmethod
    def _parse_raw_text(cls, text: str) -> List[Dict[str, Any]]:
        slots: List[Dict[str, Any]] = []
        current_day: Optional[str] = None

        for line in text.splitlines():
            line_str = line.strip()
            if not line_str:
                continue

            upper_line = line_str.upper()
            found_day = cls._match_day(upper_line)
            if found_day:
                current_day = found_day

            time_match = TIME_RANGE_REGEX.search(line_str)
            cell_info = cls._parse_cell_content(line_str)

            has_course_signal = bool(
                cell_info["subject_code"]
                or (cell_info["section_name"] and cell_info["subject_name"])
            )

            if time_match or has_course_signal:
                if time_match:
                    start_time = cls._format_time(time_match.group(1), time_match.group(2), time_match.group(3))
                    end_time = cls._format_time(time_match.group(4), time_match.group(5), time_match.group(6))
                    period_label = f"{start_time}-{end_time}"
                else:
                    # A course mention without an explicit range: keep only what was actually read
                    start_time = ""
                    end_time = ""
                    period_label = ""
                    period_match = PERIOD_REGEX.search(line_str)
                    if period_match:
                        period_label = f"Period {period_match.group(1)}"

                if cell_info["subject_code"] or cell_info["subject_name"] or cell_info["section_name"]:
                    slots.append(cls._make_slot(
                        day_of_week=current_day or "",
                        period_label=period_label,
                        start_time=start_time,
                        end_time=end_time,
                        cell_info=cell_info,
                    ))
        return slots

    # ------------------------------------------------------------------
    # Cell content parsing
    # ------------------------------------------------------------------
    @classmethod
    def _parse_cell_content(cls, cell_text: str) -> Dict[str, str]:
        """
        Parses one timetable cell. Only text actually present in the document is used.
        Missing values are returned as empty strings - never guessed.

        Handles KLU/aSc cell values such as:
          - 3-CSE-DL / 2-PG-SDA / PG2 -> subject (batch/PG-style tokens)
          - 24S12                     -> section/batch (compact code)
          - 8607, PG-8509B, Lab8301B, Room 301 -> room
        The subject token is returned as subject_name (subject_code stays empty
        unless a classic code like 23CS301 was actually read). Only text actually
        present in the document is used; missing values are returned as empty
        strings - never guessed.
        """
        clean_text = cell_text.replace("\n", " ").strip()

        # Repair OCR-fragmented hyphenated batch codes: '3-C SE-DL' -> '3-CSE-DL'
        clean_text = re.sub(
            r"\b(\d{1,2}\s*-\s*[A-Z]{1,2})\s+([A-Z]{1,6}(?:\s*-\s*[A-Z0-9]{1,6})+)",
            r"\1\2",
            clean_text,
            flags=re.IGNORECASE,
        )

        room_number = ""
        room_spans: List[str] = []
        consumed_spans: List[str] = []

        # 1. Rooms first (so room-like tokens are never mistaken for sections/codes)
        room_match = ROOM_REGEX.search(clean_text)
        if room_match:
            room_number = room_match.group(1).upper()
            consumed_spans.append(room_match.group(0))

        for token in clean_text.split():
            token_u = token.upper().strip("(),")
            if not token_u or token_u in consumed_spans:
                continue
            is_room = (
                BARE_ROOM_REGEX.match(token_u)
                or (PREFIXED_ROOM_REGEX.match(token_u) and re.search(r"\d{3,5}", token_u))
                or ROOM_HYPHEN_REGEX.match(token_u)
            )
            if is_room and not KLU_BATCH_CODE_REGEX.fullmatch(token_u) and not KLU_COMPACT_CODE_REGEX.fullmatch(token_u):
                if not room_number:
                    room_number = token_u
                consumed_spans.append(token)

        # 2. Course codes: compact (24S12), batch (3-CSE-DL / 2-PG-SDA), classic (23CS301).
        #    Classic/joined search runs on the room-stripped text so room labels like
        #    PG-8509B can never be mistaken for a course code.
        compact = KLU_COMPACT_CODE_REGEX.search(clean_text)
        batch = KLU_BATCH_CODE_REGEX.search(clean_text)
        batch_token = re.sub(r"\s+", "", batch.group(1)).upper() if batch else ""
        compact_token = compact.group(1).upper() if compact else ""

        room_stripped_text = clean_text
        for span in consumed_spans:
            room_stripped_text = room_stripped_text.replace(span, " ")

        subject_code = ""
        if compact_token:
            subject_code = compact_token
        elif batch_token:
            subject_code = batch_token
        else:
            joined = _join_ocr_code(room_stripped_text)
            classic = CLASSIC_COURSE_CODE_REGEX.search(room_stripped_text)
            if joined:
                subject_code = joined
            elif classic:
                subject_code = classic.group(1).upper()

        # 3. Section/batch: explicit "Sec A"/"Batch 2", compact "PG2", else the batch token
        section_name = ""
        sec_match = SECTION_EXPLICIT_REGEX.search(clean_text)
        sec_compact_token = ""
        if not sec_match:
            for token in clean_text.split():
                tu = token.upper().strip("(),")
                m = SECTION_COMPACT_REGEX.fullmatch(tu)
                if m:
                    sec_compact_token = f"{m.group(1).upper()}{m.group(2)}"
                    break
        if sec_match:
            section_name = f"Section {sec_match.group(1).upper()}"
        elif sec_compact_token:
            section_name = sec_compact_token
        elif batch_token:
            section_name = batch_token

        # 4. Subject name = remaining human-readable text
        remainder = clean_text
        for span in consumed_spans:
            remainder = remainder.replace(span, " ")
        if batch_token:
            remainder = remainder.replace(batch.group(0), " ")
        if compact_token:
            remainder = remainder.replace(compact.group(0), " ")
        if sec_match:
            remainder = remainder.replace(sec_match.group(0), " ")
        if sec_compact_token:
            remainder = re.sub(
                r"\b" + re.escape(sec_compact_token) + r"\b", " ", remainder, flags=re.IGNORECASE
            )
        if subject_code and not compact_token and not batch_token:
            remainder = remainder.replace(subject_code, " ")
        remainder = TIME_RANGE_REGEX.sub(" ", remainder)
        remainder = re.sub(r"\b\d{1,2}[:.]\d{2}\s*(?:AM|PM)?\b", " ", remainder, flags=re.IGNORECASE)
        remainder = re.sub(r"\bP(?:ERIOD)?\s*[-#]?\s*\d{1,2}\b", " ", remainder, flags=re.IGNORECASE)
        remainder = re.sub(r"[()\[\]:;,|]", " ", remainder)
        remainder = re.sub(r"\s+", " ", remainder).strip(" -–—.")

        # Filter fragments that are really code/section leftovers
        fragments = [f for f in remainder.split(" ") if f]
        kept = []
        for f in fragments:
            fu = f.upper()
            if re.fullmatch(r"[A-Z]{1,2}\d{1,3}", fu) and len(f) <= 5:
                continue  # leftover compact code tokens
            kept.append(f)
        subject_name = " ".join(kept)
        if len(subject_name) < 3:
            subject_name = ""

        # Faculty (KLU) convention: in a cell holding both a compact code (24S12)
        # and a batch/PG token (3-CSE-DL, PG2), the batch/PG token is the COURSE
        # CODE and the compact code is the SECTION/BATCH. Explicit "Sec-X" text is
        # never swapped, and classic codes (23CS301) always remain course codes.
        klu_style_code = bool(compact_token or batch_token)
        klu_style_section = bool(sec_compact_token or batch_token) and not sec_match
        if klu_style_code and klu_style_section and subject_code and section_name:
            subject_code, section_name = section_name, subject_code

        # 6. The extracted subject token (3-CSE-DL / PG2) is the SUBJECT, not a
        #    course code: it is returned as subject_name and subject_code stays
        #    empty. Only fires for KLU-style tokens; classic course codes
        #    (23CS301, CSE301) are kept as subject_code.
        is_classic_code = bool(subject_code) and bool(CLASSIC_COURSE_CODE_REGEX.fullmatch(subject_code))
        raw_code = subject_code
        if subject_code and not is_classic_code and (compact_token or batch_token):
            subject_name = f"{subject_name} {subject_code}".strip() if subject_name else subject_code
            subject_code = ""

        return {
            "subject_code": subject_code,
            # Token as read, before demotion - used for grid cell merge decisions
            "raw_code": raw_code,
            "subject_name": subject_name,
            "section_name": section_name,
            "room_number": room_number,
        }

    @classmethod
    def _make_slot(
        cls,
        day_of_week: str,
        period_label: str,
        start_time: str,
        end_time: str,
        cell_info: Dict[str, str],
        period_number: Optional[int] = None,
    ) -> Dict[str, Any]:
        return {
            "day_of_week": (day_of_week or "").upper(),
            "period_name": period_label or (f"Period {period_number}" if period_number else ""),
            "start_time": start_time or "",
            "end_time": end_time or "",
            "subject_code": cell_info.get("subject_code", ""),
            "subject_name": cell_info.get("subject_name", ""),
            "section_name": cell_info.get("section_name", ""),
            "room_number": cell_info.get("room_number", ""),
            "confidence": 0.0,  # set during normalization
        }

    @classmethod
    def _match_day(cls, text: str) -> Optional[str]:
        upper = text.upper()
        # Longest names first so "THURSDAY" wins over "THU"
        for day_full in ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]:
            if re.search(r"\b" + day_full + r"\b", upper):
                return day_full
        for abbr, day_full in DAY_PATTERNS.items():
            if abbr == day_full:
                continue
            if re.search(r"\b" + abbr + r"\b", upper):
                return day_full
        return None

    @classmethod
    def _format_time(cls, hour_str: str, min_str: str, meridian: Optional[str] = None) -> str:
        try:
            h = int(hour_str)
            m = int(min_str)
            if meridian:
                mer = meridian.upper()
                if mer == "PM" and h < 12:
                    h += 12
                elif mer == "AM" and h == 12:
                    h = 0
            if 0 <= h <= 23 and 0 <= m <= 59:
                return f"{h:02d}:{m:02d}"
            return ""
        except Exception:
            return ""

    @classmethod
    def _slot_completeness(cls, slot: Dict[str, Any]) -> float:
        """Fraction of the required fields actually read from the document."""
        required = ["day_of_week", "start_time", "end_time"]
        optional = ["subject_name", "section_name"]
        filled_required = sum(1 for k in required if slot.get(k))
        subject_identity = 1 if (slot.get("subject_code") or slot.get("subject_name")) else 0
        filled_optional = sum(1 for k in optional if slot.get(k))
        return round((filled_required + subject_identity + 0.5 * filled_optional) / (len(required) + 1 + 0.5 * len(optional)), 2)

    @classmethod
    def _normalize_and_deduplicate(cls, slots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        deduped = []
        for s in slots:
            if not isinstance(s, dict):
                continue
            # Drop slots with nothing useful at all
            if not (s.get("subject_code") or s.get("subject_name") or s.get("section_name")):
                continue
            day = (s.get("day_of_week") or "").upper()
            start = s.get("start_time") or ""
            end = s.get("end_time") or ""
            code = s.get("subject_code") or ""
            name = (s.get("subject_name") or "")[:40]
            key = (day, start, end, code, name)
            if key in seen:
                continue
            seen.add(key)
            s["day_of_week"] = day
            s["confidence"] = cls._slot_completeness(s)
            deduped.append(s)
        return deduped
