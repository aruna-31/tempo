# TEMPO Manual Validation & System Correction Audit Report
**Project:** Privacy-Preserving Temporal Classroom Activity Profiling Using Anonymous Student Tracking  
**Focus:** Monocular Single-Camera Observation, Temporal Evidence Gating, Annotation Collision Mitigation, & Scientific Audit  
**Date:** September 28, 2026  
**Auditor:** TEMPO Core Research & Engineering Team  

---

## 1. Executive Summary & Audit Mandate

This audit concludes Phase 2 through Phase 16 of the single-camera manual validation and system correction workflow for the TEMPO platform. Following completion of the external single-camera **CDED-7** benchmark (held-out test F1: 0.8472), visual inspection of rendered classroom videos and dashboard metrics identified critical reporting anomalies, visual label collisions, conflated confidence semantics, and unsupported pedagogical claims.

### Key Audit Verdict:
1. **Validated Benchmark Unbroken**: The native $1280\times1280$ single-camera YOLO detector and ByteTrack tracking configuration remain completely untouched and verified (`cfg_val_1_native_1280_c25`). Detection F1 (0.8472), Recall (0.8575), Precision (0.8370), Back-Row F1 (0.8616), Small F1 (0.7649), and Duplicate Rate (9.84%) are fully preserved.
2. **Root Cause of "4.5% Temporal Coverage" Resolved**: Identified two concrete algorithmic bugs in `app/services/temporal_analytics_service.py`:
   - Conflating sliding-window prediction count ($W=1$) with raw observation frame count ($N=22$), yielding $1/22 = 0.04545$ (4.5%).
   - Requiring $\ge 2$ sequence sliding windows ($T \ge 24-32$ frames) for a track to be considered `temporal_ready`, which disqualified single-window clips (10–15s).
3. **Visual Label Collision Mitigation**: Replaced monolithic 45-character badges (~380 px wide) with a two-tier adaptive rendering hierarchy: compact top tag (`S{id:02d} · {conf}%`) and bottom controlled behaviour badge with dynamic collision avoidance and clipping.
4. **Temporal Evidence Gating**: Established strict evidence levels (`SUFFICIENT`, `LIMITED`, `INSUFFICIENT`). Prohibited sweeping session-level claims (e.g., "Classroom behavioral entropy remained low throughout session") when only 1 observation window exists.
5. **Separation of Computer Vision Observation from Pedagogical Suggestions**: Purged theoretical and psychological assertions (such as claims that peer interaction "increases retention and active problem solving"). Re-anchored all insights to observable gaze vectors, head/body orientations, and clearly labeled external literature references.

---

## 2. Root Cause Analysis

| ID | Issue Description | Component | Root Cause |
|---|---|---|---|
| **RC-1** | **4.5% Temporal Coverage** | `app/services/temporal_analytics_service.py` | `temporal_coverage = round(len(rows) / max(1, len(track.bounding_box_history)), 6)`. `len(rows)` was the number of 16-frame prediction windows (1), whereas `bounding_box_history` was frame count (22). Dividing 1 by 22 gave 0.04545 (4.5%). |
| **RC-2** | **Artificial `temporal_ready` Depletion** | `app/services/temporal_analytics_service.py` | Line 140 required `if len(rows) >= 2: temporal_ready += 1`. In a short 10–15s video sampled at 2 FPS (20–30 frames), tracks yield exactly 1 sequence window ($T=16$). Requiring $\ge 2$ windows meant only 1 of 22 tracks qualified, yielding $1/22 = 4.5\%$ at the session level. |
| **RC-3** | **Visual Overlap & Unreadable Labels** | `app/ml/annotator.py`, `frontend/.../VideoPlayerWithOverlay.tsx` | Single string `Student 15: Looking_Toward_Instruction (85%)` rendered at 380 px width on bounding boxes 80–120 px wide, occluding adjacent students. |
| **RC-4** | **Ungrounded Session Claims** | `app/services/faculty_insight_service.py` | Code generated longitudinal claims ("Classroom behavioral entropy remained low throughout the session") even when only 1 single 15-second window was observed. |
| **RC-5** | **Conflated Confidence Semantics** | `FacultyInsightService` & Dashboard UI | Model prediction confidence (ML softmax output e.g. 91%) was displayed as a generic "Confidence: 91%" card, creating the false illusion of high statistical certainty in classroom-level pedagogical conclusions. |
| **RC-6** | **Psychological & Pedagogical Overreach** | `app/services/faculty_insight_service.py` | Hardcoded theoretical assertions: "Social learning theory indicates that short peer discussions increase retention and active problem solving." TEMPO measures observable posture/orientation only. |

---

## 3. Files Modified

1. `app/services/temporal_analytics_service.py`:
   - Corrected track-level temporal coverage calculation: proportion of lifespan spanned by valid sequences ($16 + (W-1)\times 8 / N$), eliminating window/frame division mismatch.
   - Updated temporal readiness criterion: track is `temporal_ready` if it accumulated $\ge 1$ complete sequence window ($T \ge 16$, `len(rows) >= 1`).
   - Gated change-point detection: requires $\ge 2$ distinct temporal windows before computing Jensen-Shannon divergence.
   - Guarded `classroom_state` to return `INSUFFICIENT_EVIDENCE` when observations are empty.
   - Enriched `CoverageMetric` payload in `read_all` with `temporal_observation_windows`, `evidence_status`, and `evidence_note`.
2. `app/services/faculty_insight_service.py`:
   - Imported `BehaviourResult` and computed actual session mean model prediction confidence.
   - Implemented temporal evidence gating (`SUFFICIENT`, `LIMITED`, `INSUFFICIENT`).
   - Rewrote single-window entropy wording: `"Observed behavioural entropy was {val} in the single available observation window (Limited temporal evidence: 1 observation window; interpret session-level patterns cautiously)."`
   - Separated observable activity from external pedagogical references (`[External Pedagogical Reference: ...]`).
   - Removed ungrounded claims regarding retention, active problem solving, and cognitive learning gains.
3. `app/ml/annotator.py`:
   - Implemented two-tier clean rendering hierarchy: compact top badge (`S{id:02d} · {conf}%`) and bottom controlled behaviour badge.
   - Added adaptive placement and frame boundary clipping: tucks badges inside bounding box when near top status banner or canvas boundaries.
   - Added dark contrast badge backgrounds with color borders to mitigate occlusion of adjacent students.
4. `frontend/src/types/index.ts`:
   - Added `temporal_observation_windows`, `evidence_status`, and `evidence_note` to `TemporalCoverage`.
5. `frontend/src/pages/ResultsPage.tsx`:
   - Updated metric cards to display: Observed Tracks, Tracking Stability, Temporal Ready Tracks, Temporal Coverage, Observation Windows, and Evidence Status.
   - Added non-overwhelming visual warning banner for `LIMITED` and `INSUFFICIENT` temporal evidence.
6. `frontend/src/components/VideoPlayerWithOverlay.tsx`:
   - Updated canvas overlay rendering to match compact hierarchy (`S{id:02d} · {conf}%` and clean behaviour label) with dynamic text measurement.
7. `frontend/src/components/FacultyInsightSection.tsx`:
   - Explicitly clarified confidence semantics as `Model Confidence: {X}%` and displayed evidence status in coverage context.
8. `tests/test_manual_validation_audit.py`:
   - Created comprehensive unit and integration test suite covering all 15 audit dimensions.
9. `tests/test_video_upload_and_streaming.py`:
   - Fixed unique constraint collision on subject code in sample session creation helper.

---

## 4. Exact Fixes Implemented

### Fix 1: Temporal Coverage & Readiness Formula
```python
# Before (app/services/temporal_analytics_service.py):
temporal_coverage = round(len(rows) / max(1, len(track.bounding_box_history)), 6)
if len(rows) >= 2:
    temporal_ready += 1

# After:
num_history = len(track.bounding_box_history or [])
if len(rows) >= 1:
    temporal_ready += 1
    if num_history >= 16:
        covered_frames = min(num_history, 16 + (len(rows) - 1) * 8)
        temporal_coverage = round(covered_frames / num_history, 4)
    else:
        temporal_coverage = 1.0
else:
    temporal_coverage = 0.0
```

### Fix 2: Evidence Gating for Faculty Insights
```python
# Before (app/services/faculty_insight_service.py):
obs = f"Classroom behavioral entropy remained low (average {mean_entropy:.2f}), indicating a single predominant activity mode."

# After:
if len(entropies) == 1:
    obs = (
        f"Observed behavioural entropy was {mean_entropy:.2f} in the single available observation window "
        f"(Limited temporal evidence: 1 observation window; interpret session-level patterns cautiously)."
    )
elif mean_entropy < 0.60:
    obs = (
        f"Observed behavioural entropy averaged {mean_entropy:.2f} across {len(entropies)} observation windows, "
        f"indicating a single predominant activity mode across the recorded segments."
    )
```

### Fix 3: Purging Psychological Claims & Separating External Theory
```python
# Before:
obs = "Direct instruction and individual tasks predominated with minimal observed Peer_Interaction."
ped = "Social learning theory indicates that short peer discussions increase retention and active problem solving."
act = "In future sessions of this module, consider scheduling a 2-minute paired problem-solving prompt to encourage collaborative reasoning."

# After:
obs = "Direct instruction and individual activity predominated during the observed interval, with minimal observable Peer_Interaction."
ped = (
    "[External Pedagogical Reference: Mazur, 1997; Prince, 2004] Active learning literature suggests brief peer discussions encourage peer-to-peer articulation. "
    "TEMPO measures observable physical orientation only, not retention or comprehension."
)
act = "Optional Instructional Suggestion: In future sessions of this module, consider scheduling a brief 2-minute paired problem-solving prompt to encourage collaborative interaction."
```

### Fix 4: Two-Tier Compact Video Annotation Hierarchy
```python
# Top Badge:
conf_str = f" · {int(ann['pred_conf'] * 100)}%" if ann.get("pred_conf") is not None else ""
id_text = f"S{t_id:02d}{conf_str}"
(tw1, th1), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)

# Adaptive placement with boundary clipping
if by1 - (th1 + 6) >= banner_h:
    t_y1 = by1 - (th1 + 6); t_y2 = by1
else:
    t_y1 = by1; t_y2 = min(by2, by1 + th1 + 6)

# Bottom Badge (Behaviour):
beh = ann["behaviour"].replace("_", " ")
(tw2, th2), _ = cv2.getTextSize(beh, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
# Bounded dark background with slim color border
```

---

## 5. Before vs. After Metrics & Behavior

| Dimension | Before Audit (Observed Issue) | After Correction (Audited & Verified) | Rationale |
|---|---|---|---|
| **Track Temporal Coverage (Track Level)** | 4.5% ($1 / 22$) | **72.7%** ($16 / 22$) | Corrected dimension mismatch; reflects fraction of track covered by 16-frame sequence. |
| **Session Temporal-Ready Tracks** | 1 of 22 tracks | **22 of 22 tracks (100.0%)** | Track is temporal-ready upon accumulating $\ge 1$ complete sequence ($T \ge 16$). |
| **Session Temporal Coverage Ratio** | 4.5% ($1 / 22$) | **100.0%** ($22 / 22$) | Ratio of temporally ready tracks to confirmed tracks. |
| **Temporal Observation Windows** | 1 window | **1 window** (preserved truth) | Accurate reflection of 15s clip duration. |
| **Evidence Status** | Unspecified / Hidden | **`LIMITED`** | System explicitly warns users when evidence is restricted to 1 window. |
| **Single-Window Entropy Text** | "Classroom behavioral entropy remained low throughout session" | "Observed behavioural entropy was 0.30 in the single available observation window (Limited temporal evidence)" | Avoids unjustified longitudinal claims. |
| **Change Points on 1 Window** | 0 reported (confusingly displayed as "stability") | Gated: change points require $\ge 2$ consecutive windows | Prevents calculating divergence on singular states. |
| **Video Label Badge Width** | ~380 px wide (collision with neighbors) | **48–60 px (Top tag), 54–75 px (Bottom tag)** | Two-tier hierarchy prevents occlusion in dense seating. |
| **Displayed Confidence** | "Confidence: 91%" | **"Model Confidence: 89%"** + **"Evidence: LIMITED (1 window)"** | Conflation eliminated; separates ML softmax confidence from evidence sufficiency. |
| **Pedagogical Text** | Claims of "retention", "active problem solving", "learning" | Observable gaze/posture cues + explicit `[External Pedagogical Reference: ...]` | Strict ethical compliance with non-deficit, observable-only principles. |

---

## 6. Automated Test Suite & Results

Automated test suites were executed to verify all 15 audit dimensions without regression:

```powershell
python -m pytest tests/test_manual_validation_audit.py tests/test_temporal_analytics.py tests/test_faculty_insights.py tests/test_single_camera_fog_recovery.py tests/test_ced7_evaluation.py tests/test_video_upload_and_streaming.py
```

### Test Results Breakdown:
- `tests/test_manual_validation_audit.py`: **9 passed** (100%)
  - `test_classroom_state_insufficient_evidence`
  - `test_classroom_state_dominance_and_mixed`
  - `test_single_window_entropy_semantics`
  - `test_change_points_minimum_evidence`
  - `test_temporal_coverage_not_ratio_of_windows_to_frames`
  - `test_fog_recovery_preserves_tagging`
  - `test_ethical_guardrails_prevent_psychological_claims`
  - `test_single_window_faculty_insight_gating`
  - `test_annotator_rendering_clean_hierarchy`
- `tests/test_temporal_analytics.py`: **6 passed** (100%)
- `tests/test_faculty_insights.py`: **3 passed** (100%)
- `tests/test_single_camera_fog_recovery.py`: **1 passed** (100%)
- `tests/test_ced7_evaluation.py`: **5 passed** (100%)
- `tests/test_video_upload_and_streaming.py`: **6 passed** (100%)

**Total Core Passed**: **30 passed in 24.8s, 0 failed.**  
**Frontend Compilation**: `tsc && vite build` built in 24.49s with **0 errors**.

---

## 7. Regression Check on Validated CDED-7 Benchmark

As mandated by Phase 16, the underlying detector and tracker architecture was protected from regression:

| Metric | Validated Benchmark Baseline | Post-Audit Status | Regression Verdict |
|---|---|---|---|
| **Detector Architecture** | YOLOv11x native $1280\times1280$ (`cfg_val_1`) | Unchanged | **Preserved (0.0% change)** |
| **Detection F1** | **0.8472** | **0.8472** | **Preserved** |
| **Precision** | **0.8370** | **0.8370** | **Preserved** |
| **Recall** | **0.8575** | **0.8575** | **Preserved** |
| **Back-Row F1** | **0.8616** | **0.8616** | **Preserved** |
| **Small-Person F1** | **0.7649** | **0.7649** | **Preserved** |
| **Duplicate Rate** | **9.84%** | **9.84%** | **Preserved** |
| **Tracking Coverage** | **96.39%** | **96.39%** | **Preserved** |
| **Privacy Constraints** | No face recognition, no permanent ID | Fully verified | **Preserved** |

---

## 8. Known Limitations & Scientific Honesty

1. **Short Video Clips**: When classroom videos are shorter than 60 seconds (or sampled to fewer than 30 frames), only 1 temporal aggregation window can be constructed. The system correctly identifies this as `LIMITED` evidence and warns faculty not to infer whole-session trends.
2. **ID-Switch Ground Truth**: In unannotated single-camera footage without manual per-frame identity tags, ID-switch accuracy cannot be mathematically computed against ground truth. The system explicitly declares: *"ID-switch ground truth is unavailable; stability is measured using internal track continuity/reappearance criteria."*
3. **Seating Capacity vs. Simultaneous Visibility**: CDED-7 validates up to **30 simultaneously visible students** in a real classroom camera perspective. This benchmark must not be conflated with the nominal 50–70 student capacity of lecture halls, where severe back-row occlusions and off-screen students require camera view panning or wide-angle optics.
4. **Physical Observation vs. Internal Cognition**: TEMPO detects physical gaze direction (towards instruction, reading, writing, peer interaction, looking away) and posture. It does **not** measure comprehension, intelligence, emotional state, or learning outcomes.

---

## 9. Conclusion

The TEMPO single-camera platform is now:
- **Mathematically sound**: Coverage metrics and sliding window calculations accurately distinguish frame spans from window counts.
- **Temporally grounded**: Longitudinal conclusions are strictly gated by temporal evidence thresholds.
- **Visually clean**: Video annotations prevent label collisions in dense seating rows.
- **Ethically compliant**: Prohibits deficit labels, separates computer vision observations from pedagogical suggestions, and adheres strictly to privacy preservation.
- **Completely regression-tested**: Baseline CDED-7 benchmark metrics remain 100% intact.
