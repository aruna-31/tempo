"""
Tests for the timetable OCR service and ML configuration integrity.

Verifies:
1. KLU timetable cell patterns (24S12, PG2, 3-CSE-DL, 2-PG-SDA) parse correctly.
2. The OCR service NEVER invents values: unreadable fields are empty strings and
   no fallback template is generated.
3. models/model_metadata.json is the strict single source of truth and the
   checkpoint exactly matches it.
"""
import json
import os

import pytest

from app.services.timetable_ocr_service import TimetableOCRService

METADATA_PATH = os.path.join("models", "model_metadata.json")
WEIGHTS_PATH = os.path.join("models", "classroom_temporal_model.pth")


# ---------------------------------------------------------------------------
# KLU cell pattern parsing
# ---------------------------------------------------------------------------
class TestKLUCellPatterns:
    def test_compact_token_is_subject_not_code(self):
        # Faculty convention: 24S12 alone is the subject identity (subject_name),
        # subject_code stays empty
        info = TimetableOCRService._parse_cell_content("24S12")
        assert info["subject_name"] == "24S12"
        assert info["subject_code"] == ""

    def test_batch_style_token_is_subject(self):
        info = TimetableOCRService._parse_cell_content("2-PG-SDA")
        assert info["subject_name"] == "2-PG-SDA"
        assert info["subject_code"] == ""

    def test_batch_style_code_3_cse_dl(self):
        info = TimetableOCRService._parse_cell_content("3-CSE-DL")
        assert info["subject_name"] == "3-CSE-DL"
        assert info["subject_code"] == ""

    def test_compact_section_pg2(self):
        # Faculty convention: PG2 is the subject, 24S12 is the section/batch
        info = TimetableOCRService._parse_cell_content("24S12 Deep Learning PG2")
        assert "PG2" in info["subject_name"]
        assert "Deep Learning" in info["subject_name"]
        assert info["subject_code"] == ""
        assert info["section_name"] == "24S12"

    def test_classic_course_code_stays_code(self):
        info = TimetableOCRService._parse_cell_content("23CS301 Machine Learning")
        assert info["subject_code"] == "23CS301"
        assert "Machine Learning" in info["subject_name"]

    def test_explicit_section_and_room(self):
        info = TimetableOCRService._parse_cell_content("23CS301 Machine Learning Sec-B R-402")
        assert info["subject_code"] == "23CS301"
        assert info["section_name"] == "Section B"
        assert info["room_number"] == "402"
        assert "Machine Learning" in info["subject_name"]

    def test_time_range_extraction(self):
        slots = TimetableOCRService._parse_raw_text(
            "MONDAY 09:00-10:00 CSE301 Deep Learning Sec-A"
        )
        assert len(slots) == 1
        assert slots[0]["day_of_week"] == "MONDAY"
        assert slots[0]["start_time"] == "09:00"
        assert slots[0]["end_time"] == "10:00"
        assert slots[0]["subject_code"] == "CSE301"

    def test_non_teaching_cells_are_dropped(self):
        slots = TimetableOCRService._parse_raw_text("LUNCH\nBREAK\nFREE\n")
        assert slots == []


# ---------------------------------------------------------------------------
# No-invention guarantee
# ---------------------------------------------------------------------------
class TestNoInvention:
    def test_garbage_image_returns_empty_not_template(self):
        """A non-decodable 'image' must yield zero slots - never a fabricated template."""
        slots = TimetableOCRService.extract_from_file_bytes(b"not-an-image", "timetable.png")
        assert slots == []

    def test_blank_image_returns_empty(self):
        import cv2
        import numpy as np

        blank = np.full((300, 400, 3), 255, dtype=np.uint8)
        ok, buf = cv2.imencode(".png", blank)
        assert ok
        slots = TimetableOCRService.extract_from_file_bytes(buf.tobytes(), "blank.png")
        assert slots == []

    def test_no_slot_ever_contains_invented_defaults(self):
        """Parsed slots must never contain the old invented defaults (CS301, Section 1, Room 301)."""
        slots = TimetableOCRService._parse_raw_text("24S12 PG2")
        for s in slots:
            assert s["subject_code"] != "CS301"
            assert s["section_name"] != "Section 1"
            assert s["room_number"] != "Room 301"
            if not s["start_time"]:
                assert s["start_time"] == ""  # missing stays missing

    def test_cell_with_unreadable_fields_yields_empty_strings(self):
        info = TimetableOCRService._parse_cell_content("24S12")
        # Only the subject token was present: everything else must be empty, not guessed
        assert info["subject_name"] == "24S12"
        assert info["subject_code"] == ""
        assert info["section_name"] == ""
        assert info["room_number"] == ""

    def test_klu_convention_swap(self):
        """Cell with both values: batch/PG token = subject, compact code = section."""
        info = TimetableOCRService._parse_cell_content("24S12 3-CSE-DL")
        assert info["subject_name"] == "3-CSE-DL"
        assert info["section_name"] == "24S12"
        assert info["subject_code"] == ""

        info = TimetableOCRService._parse_cell_content("2-PG-SDA PG2")
        assert info["subject_name"] == "PG2"
        assert info["section_name"] == "2-PG-SDA"

        # Explicit Sec-X text is never swapped
        info = TimetableOCRService._parse_cell_content("3-CSE-DL Sec-B")
        assert info["subject_name"] == "3-CSE-DL"
        assert info["section_name"] == "Section B"

    def test_klu_timetable_image_end_to_end(self):
        """Full OCR extraction on an image containing the documented KLU cell values."""
        from PIL import Image, ImageDraw
        import io

        img = Image.new("RGB", (900, 300), color=(255, 255, 255))
        d = ImageDraw.Draw(img)
        d.text((20, 30), "DAY", fill=(0, 0, 0))
        d.text((150, 30), "09:00 - 10:00", fill=(0, 0, 0))
        d.text((500, 30), "10:00 - 11:00", fill=(0, 0, 0))
        d.text((20, 90), "MONDAY", fill=(0, 0, 0))
        d.text((150, 90), "24S12 PG2", fill=(0, 0, 0))
        d.text((500, 90), "2-PG-SDA", fill=(0, 0, 0))
        d.text((20, 160), "TUESDAY", fill=(0, 0, 0))
        d.text((150, 160), "3-CSE-DL Sec-B", fill=(0, 0, 0))
        d.text((500, 160), "24S12 Machine Learning PG2", fill=(0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, "PNG")

        slots = TimetableOCRService.extract_from_file_bytes(buf.getvalue(), "klu_timetable.png")
        subjects = {s["subject_name"] for s in slots}
        # Faculty convention: PG2 / 2-PG-SDA / 3-CSE-DL are subjects
        assert "PG2" in subjects
        assert "2-PG-SDA" in subjects
        assert "3-CSE-DL" in subjects
        # 24S12 is a section/batch identifier
        assert any(s["section_name"] == "24S12" for s in slots)
        # KLU slots carry no course code; only text from the image appears
        for s in slots:
            assert s["subject_code"] in {"", "23CS301", "CSE301", "CSE402", "CSE305"}
            assert s["subject_name"] in {
                "24S12", "2-PG-SDA", "3-CSE-DL", "PG2", "",
                "24S12 Machine Learning", "Machine Learning",
                "Machine Learning PG2",
            }


class TestAScGridReconstruction:
    """Grid reconstruction for aSc/KARE-format timetables: period headers like
    '1 (09-10)', day rows, and cells stacked as room / code / batch fragments."""

    def _items(self, rows):
        """rows: list of (y, x, text) -> positioned OCR items (x1 estimated)."""
        return [
            {"text": t, "x": float(x), "x1": float(x) + max(len(t) * 11.0, 30.0), "y": float(y), "conf": 0.9}
            for (y, x, t) in rows
        ]

    def test_period_header_times_use_academic_day_bias(self):
        # 5 (01-02) is an afternoon period -> 13:00-14:00, not 01:00-02:00
        assert TimetableOCRService._academic_hour_to_time(1) == "13:00"
        assert TimetableOCRService._academic_hour_to_time(6) == "18:00"
        assert TimetableOCRService._academic_hour_to_time(9) == "09:00"
        assert TimetableOCRService._academic_hour_to_time(12) == "12:00"

    def test_asc_grid_full_extraction(self):
        """Fragments of the real uploaded aSc/KARE timetable reconstruct to 7 slots."""
        items = self._items([
            (13, 301, "KARE - School of Computing -Department of CSE"),
            (45, 313, "DrRAJASUBRAMANIANR"),
            (111, 122, "1 (09-10)"),
            (110, 239, "2 (10-11)"),
            (110, 359, "3 (11-12)"),
            (111, 479, "4 (12-01)"),
            (111, 599, "5 (01-02)"),
            (111, 719, "6 (02-03)"),
            (111, 839, "7 (03-04)"),
            (111, 956, "8 (04-05)"),
            (160, 310, "8607"),
            (206, 18, "Monday"),
            (207, 241, "24S12"),
            (264, 229, "3-CSE-DL"),
            (336, 20, "Tuesday"),
            (336, 360, "24S12"),
            (394, 349, "3-CSE-DL"),
            (290, 430, "8607"),
            (416, 429, "8607"),
            (335, 732, "PG2"),
            (291, 763, "PG-8509B"),
            (392, 709, "2-PG-SDA"),
            (468, 22, "Wednesday"),
            (420, 286, "PG-8509B"),
            (463, 254, "PG2"),
            (521, 228, "2-PG-SDA"),
            (464, 360, "24S12"),
            (518, 346, "3-CSE-DL"),
            (545, 398, "Lab8301B"),
            (593, 20, "Thursday"),
            (593, 301, "24S12"),
            (646, 232, "3-CSE-DL"),
            (674, 400, "PG-8509B"),
            (717, 20, "Friday"),
            (717, 313, "PG2"),
            (776, 234, "2-PG-SDA"),
            (795, 15, "CSE"),
            (797, 972, "aSc Timetables"),
        ])
        slots = TimetableOCRService._parse_ocr_items(items)
        assert len(slots) == 7

        by_day = {s["day_of_week"]: s for s in slots}
        assert set(by_day) == {"MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY"}

        mon = by_day["MONDAY"]
        assert mon["start_time"] == "10:00" and mon["end_time"] == "11:00"
        assert mon["subject_name"] == "3-CSE-DL"
        assert mon["section_name"] == "24S12"
        assert mon["room_number"] == "8607"

        # Thursday is a printed double period spanning columns 2-3 (10:00-12:00)
        thu = by_day["THURSDAY"]
        assert thu["start_time"] == "10:00" and thu["end_time"] == "12:00"
        assert "Periods 2-3" in thu["period_name"]
        assert thu["subject_name"] == "3-CSE-DL"

        # Tuesday period 6 (02-03) is an afternoon slot -> 14:00-15:00
        tue_slots = [s for s in slots if s["day_of_week"] == "TUESDAY"]
        assert any(s["start_time"] == "14:00" and s["end_time"] == "15:00" for s in tue_slots)
        pg2_tue = [s for s in tue_slots if s["subject_name"] == "PG2"][0]
        assert pg2_tue["section_name"] == "2-PG-SDA"

        # No page artifacts leak into slots
        for s in slots:
            assert "KARE" not in s["subject_name"]
            assert "RAJA" not in s["subject_name"].upper()
            assert s["subject_code"] != "PG-8509B"  # room never becomes a code

    def test_title_and_faculty_name_never_become_slots(self):
        items = self._items([
            (13, 301, "KARE - School of Computing -Department of CSE"),
            (45, 313, "DrRAJASUBRAMANIANR"),
            (111, 122, "1 (09-10)"),
            (206, 18, "Monday"),
            (207, 241, "24S12"),
        ])
        slots = TimetableOCRService._parse_ocr_items(items)
        assert len(slots) == 1
        assert slots[0]["day_of_week"] == "MONDAY"
        assert slots[0]["subject_name"] == "24S12"


# ---------------------------------------------------------------------------
# Single source of truth integrity
# ---------------------------------------------------------------------------
class TestModelMetadataIntegrity:
    def test_metadata_file_exists_with_required_keys(self):
        assert os.path.exists(METADATA_PATH), "model_metadata.json is required"
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        required = [
            "model_version", "spatial_backbone", "temporal_model_type", "hidden_dim",
            "num_layers", "sequence_length", "sampling_fps", "input_size",
            "classes", "class_mapping", "normalization",
        ]
        for key in required:
            assert key in meta, f"model_metadata.json missing required key: {key}"

    def test_checkpoint_matches_metadata_strictly(self):
        import torch

        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)

        from app.ml.model import ResNet18TemporalModel

        model = ResNet18TemporalModel(
            temporal_type=meta["temporal_model_type"],
            hidden_dim=meta["hidden_dim"],
            num_layers=meta["num_layers"],
            num_classes=len(meta["classes"]),
        )
        state_dict = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=True)  # raises on any mismatch

    def test_five_observable_behaviour_classes(self):
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["classes"] == [
            "Looking_Toward_Instruction",
            "Reading",
            "Writing",
            "Peer_Interaction",
            "Looking_Away",
        ]

    def test_ml_service_rejects_metadata_settings_mismatch(self, tmp_path):
        """A wrong .env value must fail loudly (single source of truth is the metadata file)."""
        from app.core.config import settings
        from app.ml.service import MLInferenceService

        original = settings.TEMPORAL_MODEL_TYPE
        settings.TEMPORAL_MODEL_TYPE = "GRU"  # metadata says RNN
        try:
            with pytest.raises(RuntimeError, match="single source of truth"):
                MLInferenceService(device_name="cpu")
        finally:
            settings.TEMPORAL_MODEL_TYPE = original
