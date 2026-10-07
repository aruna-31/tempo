"""Post-Class Faculty Insight & Recommendation Engine.

Transforms TEMPO's validated temporal analytics (classroom states, transitions,
entropy, change points, coverage) into objective, class-level instructional
recommendations for future classes.

Strict Pedagogical & Ethical Guardrails:
1. Zero Deficit Labeling: Never label any student or group as bored, disengaged,
   weak, lazy, inattentive, or not learning.
2. Observable Only: Ground all statements in observable posture, gaze direction,
   and activity transitions.
3. Future-Oriented: All suggestions are actionable guidance for FUTURE class design.
4. Coverage-Aware: Explicitly state tracking coverage, stability, and sample context.
"""

from typing import Any, Dict, List, Optional
from uuid import UUID
from sqlalchemy.orm import Session

from app.models.analysis import (
    BehaviourResult,
    BehaviourTransition,
    ChangePoint,
    ClassroomEntropy,
    ClassroomTemporalState,
    CoverageMetric,
    FacultyInsight,
    StudentTemporalProfile,
)

FORBIDDEN_DEFICIT_WORDS = {
    "bored",
    "boredom",
    "disengaged",
    "disengagement",
    "inattentive",
    "inattention",
    "weak",
    "weakness",
    "lazy",
    "not learning",
    "distracted student",
    "bad student",
    "poor student",
    "failing",
    "unfocused",
}


def assert_ethical_guardrails(text: str) -> None:
    """Validate that text strictly complies with ethical and pedagogical guardrails."""
    lower = text.lower()
    for word in FORBIDDEN_DEFICIT_WORDS:
        if word in lower:
            raise ValueError(
                f"Ethical Guardrail Violation: prohibited deficit term '{word}' found in generated insight text."
            )


class FacultyInsightService:
    @staticmethod
    def generate_and_persist_insights(db: Session, job_id: UUID) -> List[FacultyInsight]:
        """Generate pedagogical insights for future class planning and persist to DB."""
        # Clean existing insights for this job
        db.query(FacultyInsight).filter(FacultyInsight.job_id == job_id).delete(synchronize_session=False)

        states = (
            db.query(ClassroomTemporalState)
            .filter_by(job_id=job_id)
            .order_by(ClassroomTemporalState.start_time.asc())
            .all()
        )
        entropies = (
            db.query(ClassroomEntropy)
            .filter_by(job_id=job_id)
            .order_by(ClassroomEntropy.timestamp.asc())
            .all()
        )
        change_points = (
            db.query(ChangePoint)
            .filter_by(job_id=job_id)
            .order_by(ChangePoint.timestamp.asc())
            .all()
        )
        transitions = (
            db.query(BehaviourTransition)
            .filter_by(job_id=job_id)
            .all()
        )
        coverage = (
            db.query(CoverageMetric)
            .filter_by(job_id=job_id)
            .first()
        )
        profiles = (
            db.query(StudentTemporalProfile)
            .filter_by(job_id=job_id)
            .all()
        )

        behaviours = (
            db.query(BehaviourResult)
            .filter_by(job_id=job_id)
            .all()
        )
        avg_model_conf = (
            sum(b.confidence for b in behaviours) / len(behaviours)
            if behaviours
            else 0.85
        )

        num_windows = len(states)
        if num_windows == 0:
            evidence_status = "INSUFFICIENT"
        elif num_windows < 4:
            evidence_status = "LIMITED"
        else:
            evidence_status = "SUFFICIENT"

        insights: List[FacultyInsight] = []

        # -------------------------------------------------------------
        # 1. PACING: Analyze direct instruction duration & monologue blocks
        # -------------------------------------------------------------
        instruction_runs: List[List[ClassroomTemporalState]] = []
        current_run: List[ClassroomTemporalState] = []
        for s in states:
            if s.state == "Instruction-Dominant":
                current_run.append(s)
            else:
                if current_run:
                    instruction_runs.append(current_run)
                    current_run = []
        if current_run:
            instruction_runs.append(current_run)

        # Look for longest or significant instruction run
        if instruction_runs:
            longest_run = max(instruction_runs, key=len)
            run_duration = longest_run[-1].end_time - longest_run[0].start_time
            start_t = longest_run[0].start_time
            end_t = longest_run[-1].end_time

            if len(longest_run) >= 2 or run_duration >= 60.0:
                obs = (
                    f"Classroom remained primarily in Instruction-Dominant activity from {start_t:.0f}s to {end_t:.0f}s "
                    f"({run_duration:.0f}s continuous block)."
                )
            else:
                obs = (
                    f"Classroom was observed in Instruction-Dominant activity in the available {run_duration:.0f}s temporal window "
                    f"(from {start_t:.0f}s to {end_t:.0f}s; {evidence_status.lower()} temporal evidence)."
                )
            ped = (
                "[External Pedagogical Reference: Sweller, 1988] Cognitive load theory suggests that continuous direct presentation "
                "can saturate working memory capacity. TEMPO measures observable posture and gaze orientation only, not cognitive retention or comprehension."
            )
            act = (
                "Optional Instructional Suggestion: In future classes covering this material, consider inserting an active pause—such as a 90-second "
                "think-pair-share or quick concept check—around the midpoint of extended lecture blocks."
            )
            assert_ethical_guardrails(obs + " " + ped + " " + act)
            insights.append(
                FacultyInsight(
                    job_id=job_id,
                    category="PACING",
                    start_time=start_t,
                    end_time=end_t,
                    observation=obs,
                    pedagogical_context=ped,
                    suggested_action=act,
                    coverage_context=f"Observed across {longest_run[0].students_contributing} contributing tracks. Model confidence: {avg_model_conf * 100:.1f}%. Evidence: {evidence_status} ({num_windows} window(s)).",
                    confidence=round(avg_model_conf, 4),
                )
            )

        # -------------------------------------------------------------
        # 2. ATTENTION PATTERNS: Observable Looking_Away shifts
        # -------------------------------------------------------------
        looking_away_windows = []
        for s in states:
            dist = s.behaviour_distribution_json or {}
            la_pct = dist.get("Looking_Away", 0.0)
            if la_pct >= 0.20:
                looking_away_windows.append((s, la_pct))

        if looking_away_windows:
            peak_window, peak_pct = max(looking_away_windows, key=lambda x: x[1])
            start_t = peak_window.start_time
            end_t = peak_window.end_time
            obs = (
                f"Observable Looking_Away activity reached {peak_pct * 100:.1f}% during the {start_t:.0f}s - {end_t:.0f}s segment."
            )
            ped = (
                "[External Pedagogical Reference: Fisher et al., 2014] In classroom observation research, gaze orientation away from the primary instructional vector "
                "often coincides with conceptual transitions, note consolidation, or activity shifting. TEMPO measures observable gaze vectors only, not internal attentiveness."
            )
            act = (
                "Optional Instructional Suggestion: In future classes covering this topic, an instructor may consider placing a low-stakes formative check or interactive "
                "prompt around this point to re-anchor cohort attention."
            )
            assert_ethical_guardrails(obs + " " + ped + " " + act)
            insights.append(
                FacultyInsight(
                    job_id=job_id,
                    category="ATTENTION_PATTERNS",
                    start_time=start_t,
                    end_time=end_t,
                    observation=obs,
                    pedagogical_context=ped,
                    suggested_action=act,
                    coverage_context=f"Based on {peak_window.students_contributing} contributing tracks. Model confidence: {avg_model_conf * 100:.1f}%. Evidence: {evidence_status} ({num_windows} window(s)).",
                    confidence=round(avg_model_conf, 4),
                )
            )

        # -------------------------------------------------------------
        # 3. INTERACTION: Peer interaction & Collaborative segments
        # -------------------------------------------------------------
        peer_windows = []
        for s in states:
            dist = s.behaviour_distribution_json or {}
            peer_pct = dist.get("Peer_Interaction", 0.0)
            if peer_pct >= 0.15:
                peer_windows.append((s, peer_pct))

        if peer_windows:
            best_peer, best_peer_pct = max(peer_windows, key=lambda x: x[1])
            start_t = best_peer.start_time
            end_t = best_peer.end_time
            obs = (
                f"Observable Peer_Interaction accounted for {best_peer_pct * 100:.1f}% of observed activities between {start_t:.0f}s and {end_t:.0f}s."
            )
            ped = (
                "[External Pedagogical Reference: Chi, 2009] In active-learning taxonomies, collaborative peer exchanges provide opportunities for verbal articulation and peer clarification. "
                "TEMPO measures observable physical head and body orientation only, not spoken content or academic performance."
            )
            act = (
                "Optional Instructional Suggestion: For future cohorts, following collaborative discussion intervals with a designated 2-minute synthesis wrap-up "
                "helps crystallize shared peer insights into structured takeaways."
            )
            assert_ethical_guardrails(obs + " " + ped + " " + act)
            insights.append(
                FacultyInsight(
                    job_id=job_id,
                    category="INTERACTION",
                    start_time=start_t,
                    end_time=end_t,
                    observation=obs,
                    pedagogical_context=ped,
                    suggested_action=act,
                    coverage_context=f"Recorded with {best_peer.students_contributing} contributing tracks. Model confidence: {avg_model_conf * 100:.1f}%. Evidence: {evidence_status} ({num_windows} window(s)).",
                    confidence=round(avg_model_conf, 4),
                )
            )
        else:
            # Suggest introducing collaborative opportunities if none observed
            obs = "Direct instruction and individual activity predominated during the observed interval, with minimal observable Peer_Interaction."
            ped = (
                "[External Pedagogical Reference: Mazur, 1997; Prince, 2004] Active learning literature suggests brief peer discussions encourage peer-to-peer articulation. "
                "TEMPO measures observable physical orientation only, not retention or comprehension."
            )
            act = (
                "Optional Instructional Suggestion: In future sessions of this module, consider scheduling a brief 2-minute paired problem-solving prompt "
                "to encourage collaborative reasoning."
            )
            assert_ethical_guardrails(obs + " " + ped + " " + act)
            insights.append(
                FacultyInsight(
                    job_id=job_id,
                    category="INTERACTION",
                    start_time=None,
                    end_time=None,
                    observation=obs,
                    pedagogical_context=ped,
                    suggested_action=act,
                    coverage_context=f"Aggregate session-level analysis. Model confidence: {avg_model_conf * 100:.1f}%. Evidence: {evidence_status} ({num_windows} window(s)).",
                    confidence=round(avg_model_conf, 4),
                )
            )

        # -------------------------------------------------------------
        # 4. VARIETY: Behavioral Entropy & Multimodal Delivery
        # -------------------------------------------------------------
        if entropies:
            mean_entropy = sum(e.entropy for e in entropies) / len(entropies)
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
            else:
                obs = (
                    f"Observed behavioural entropy averaged {mean_entropy:.2f} across {len(entropies)} observation windows, "
                    f"reflecting multiple concurrent observable activities across the recorded segments."
                )
            ped = (
                "[External Pedagogical Reference: Mayer, 2002] Multi-activity classroom designs combining lecture presentation with writing or problem-solving prompts "
                "support varied learning preferences. TEMPO measures observable activity distributions only, not student learning or comprehension."
            )
            act = (
                "Optional Instructional Suggestion: In future classes, consider interleaving multimodal tasks—such as guided note templates or diagram-labeling exercises—"
                "to enrich instructional variety."
            )
            assert_ethical_guardrails(obs + " " + ped + " " + act)
            insights.append(
                FacultyInsight(
                    job_id=job_id,
                    category="VARIETY",
                    start_time=entropies[0].timestamp,
                    end_time=entropies[-1].timestamp,
                    observation=obs,
                    pedagogical_context=ped,
                    suggested_action=act,
                    coverage_context=f"Computed across {len(entropies)} temporal observation window(s). Model confidence: {avg_model_conf * 100:.1f}%. Evidence: {evidence_status}.",
                    confidence=round(avg_model_conf, 4),
                )
            )

        # -------------------------------------------------------------
        # 5. COVERAGE & OBSERVATIONAL INTEGRITY
        # -------------------------------------------------------------
        if coverage:
            det_count = coverage.detected_student_count
            trk_cov = (coverage.tracking_coverage or 0.0) * 100
            temp_cov = (coverage.temporal_coverage or 0.0) * 100
            ref_note = (
                f" (benchmark reference: {coverage.manual_reference_student_count} students)"
                if coverage.manual_reference_student_count
                else ""
            )
            evidence_clause = f" across {num_windows} observation window(s) ({evidence_status} temporal evidence)" if num_windows else ""
            obs = (
                f"Session analysis captured {det_count} anonymous student tracks with {trk_cov:.1f}% tracking stability "
                f"and {temp_cov:.1f}% temporal coverage{ref_note}{evidence_clause}. ID-switch ground truth is unavailable; "
                f"stability is measured using internal track continuity/reappearance criteria."
            )
            ped = (
                "TEMPO measures observable physical orientation and posture cues from single-camera computer vision. "
                "It strictly does NOT measure internal cognitive comprehension, emotional state, intelligence, or academic capability."
            )
            act = (
                "Optional Instructional Suggestion: Use these temporal metrics as reflective class-level feedback for instructional design. "
                "Always correlate computer vision observations with direct student feedback and academic assessments."
            )
            assert_ethical_guardrails(obs + " " + ped + " " + act)
            insights.append(
                FacultyInsight(
                    job_id=job_id,
                    category="COVERAGE",
                    start_time=None,
                    end_time=None,
                    observation=obs,
                    pedagogical_context=ped,
                    suggested_action=act,
                    coverage_context=f"Overall session tracking integrity: {trk_cov:.1f}%. Temporal coverage: {temp_cov:.1f}%. Evidence: {evidence_status} ({num_windows} window(s)).",
                    confidence=round(avg_model_conf, 4),
                )
            )

        db.add_all(insights)
        db.commit()
        return db.query(FacultyInsight).filter_by(job_id=job_id).order_by(FacultyInsight.created_at.asc()).all()

    @staticmethod
    def get_insights_for_job(db: Session, job_id: UUID) -> List[FacultyInsight]:
        """Retrieve stored insights or generate them if not present."""
        existing = (
            db.query(FacultyInsight)
            .filter_by(job_id=job_id)
            .order_by(FacultyInsight.created_at.asc())
            .all()
        )
        if not existing:
            return FacultyInsightService.generate_and_persist_insights(db, job_id)
        return existing
