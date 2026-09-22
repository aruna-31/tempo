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
                ped = (
                    "Cognitive load research indicates that working memory benefit from periodic processing breaks "
                    "during continuous direct instruction."
                )
                act = (
                    "In future classes covering this material, consider inserting an active pause—such as a 90-second "
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
                        coverage_context=f"Observed across {longest_run[0].students_contributing} contributing tracks.",
                        confidence=0.92,
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
                "Observable gaze orientation away from the primary instructional vector often coincides with "
                "conceptual transitions, note consolidation, or cognitive processing saturation."
            )
            act = (
                "In future classes covering this topic, consider placing a low-stakes formative check or interactive "
                "cold-call question around this point to re-anchor cohort attention."
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
                    coverage_context=f"Based on {peak_window.students_contributing} contributing tracks.",
                    confidence=0.88,
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
                f"Peer_Interaction accounted for {best_peer_pct * 100:.1f}% of observed activities between {start_t:.0f}s and {end_t:.0f}s."
            )
            ped = (
                "Collaborative peer exchanges promote active verbal articulation and peer clarification of concepts."
            )
            act = (
                "For future cohorts, following collaborative discussion intervals with a designated 2-minute synthesis wrap-up "
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
                    coverage_context=f"Recorded with {best_peer.students_contributing} contributing tracks.",
                    confidence=0.90,
                )
            )
        else:
            # Suggest introducing collaborative opportunities if none observed
            obs = "Direct instruction and individual tasks predominated with minimal observed Peer_Interaction."
            ped = "Social learning theory indicates that short peer discussions increase retention and active problem solving."
            act = (
                "In future sessions of this module, consider scheduling a 2-minute paired problem-solving prompt "
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
                    coverage_context="Aggregate session-level analysis.",
                    confidence=0.85,
                )
            )

        # -------------------------------------------------------------
        # 4. VARIETY: Behavioral Entropy & Multimodal Delivery
        # -------------------------------------------------------------
        if entropies:
            mean_entropy = sum(e.entropy for e in entropies) / len(entropies)
            if mean_entropy < 0.60:
                obs = f"Classroom behavioral entropy remained low (average {mean_entropy:.2f}), indicating a single predominant activity mode."
                ped = "Multimodal instruction combining auditory explanation, slide reading, and active writing engages complementary sensory channels."
                act = (
                    "In future classes, consider interleaving multimodal tasks—such as guided note templates or diagram-labeling exercises—"
                    "to enrich instructional variety."
                )
            else:
                obs = f"Classroom exhibited balanced activity entropy (average {mean_entropy:.2f}), reflecting diverse instructional modalities."
                ped = "Frequent alternation among explanation, reading, and writing supports sustained classroom alertness."
                act = (
                    "Maintain this dynamic pacing in future class designs, ensuring clear verbal transition markers between each activity change."
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
                    coverage_context=f"Computed across {len(entropies)} temporal observation windows.",
                    confidence=0.91,
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
            obs = (
                f"Session analysis captured {det_count} anonymous student tracks with {trk_cov:.1f}% tracking stability "
                f"and {temp_cov:.1f}% temporal coverage{ref_note}."
            )
            ped = (
                "TEMPO measures observable physical orientation and posture cues from computer vision. "
                "It does NOT measure internal cognitive comprehension, emotional state, intelligence, or student capability."
            )
            act = (
                "Use these temporal metrics as reflective class-level feedback for instructional design. "
                "Always correlate video observations with direct student feedback and formative quiz results."
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
                    coverage_context=f"Overall session tracking integrity: {trk_cov:.1f}%.",
                    confidence=0.95,
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
