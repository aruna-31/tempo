export type ObservableBehaviour = 
  | "Looking_Toward_Instruction"
  | "Reading"
  | "Writing"
  | "Peer_Interaction"
  | "Looking_Away";

export interface Faculty {
  id: string;
  email: string;
  full_name: string;
  department: string;
  designation: string;
  role?: string;
  is_superuser?: boolean;
  is_active: boolean;
}

export interface AuthState {
  token: string | null;
  faculty: Faculty | null;
  isAuthenticated: boolean;
  isLoading: boolean;
}

export interface Subject {
  id: string;
  code: string;
  name: string;
  faculty_id: string;
  created_at: string;
}

export interface Section {
  id: string;
  subject_id: string;
  name: string;
  academic_year: string;
  semester: string;
  created_at: string;
}

export interface Student {
  id: string;
  section_id: string;
  roll_number: string;
  name: string;
  email: string;
  created_at: string;
}

export interface Camera {
  id: string;
  room_id: string;
  camera_name: string;
  device_code: string;
  stream_url_or_channel?: string | null;
  ip_address?: string | null;
  location_in_room: string;
  status: string;
  created_at: string;
}

export interface Room {
  id: string;
  room_number: string;
  building: string;
  floor: number;
  capacity: number;
  is_active: boolean;
  cameras?: Camera[];
  created_at: string;
}

export interface FacultySchedule {
  id: string;
  faculty_id: string;
  subject_id: string;
  section_id: string;
  room_id?: string | null;
  day_of_week: string;
  start_time: string;
  end_time: string;
  academic_year: string;
  semester: string;
  is_active: boolean;
  created_at: string;
}

export interface TodayClass {
  session_id: string;
  schedule_id?: string | null;
  title: string;
  subject_code: string;
  subject_name: string;
  section_name: string;
  room_number?: string | null;
  start_time: string;
  end_time: string;
  day_of_week: string;
  session_date: string;
  status: string;
  videos_count: number;
  latest_video_id?: string | null;
  latest_job_id?: string | null;
  latest_job_status?: string | null;
  has_analysis: boolean;
}

export interface ClassSession {
  id: string;
  section_id: string;
  faculty_id: string;
  title: string;
  session_date: string;
  start_time: string;
  end_time: string;
  room_number?: string | null;
  room_id?: string | null;
  day_of_week?: string | null;
  status?: string;
  created_at: string;
}

export interface Video {
  id: string;
  session_id: string;
  faculty_id: string;
  file_path: string;
  original_filename: string;
  file_size_bytes: number;
  duration_seconds: number | null;
  status: "UPLOADED" | "PROCESSING" | "ANALYZED" | "FAILED";
  created_at: string;
}

export interface JobLogEntry {
  timestamp: string;
  stage: string;
  level: string;
  message: string;
}

export interface AnalysisJob {
  id: string;
  video_id: string;
  status: "PENDING" | "QUEUED" | "PROCESSING" | "COMPLETED" | "FAILED" | "RETRYING";
  current_stage?: string;
  progress_pct: number;
  retry_count?: number;
  max_retries?: number;
  source_type?: string;
  started_at?: string | null;
  completed_at?: string | null;
  execution_time_seconds?: number | null;
  error_message?: string | null;
  processing_logs?: JobLogEntry[];
  config_json: Record<string, any>;
  created_at: string;
}

export interface BoundingBoxPoint {
  frame: number;
  timestamp: number;
  bbox: [number, number, number, number]; // [x1, y1, x2, y2]
  confidence: number;
}

export interface StudentTrackResult {
  id: string;
  job_id: string;
  track_id: number;
  student_id?: string | null;
  bounding_box_history: BoundingBoxPoint[];
  start_frame: number;
  end_frame: number;
  created_at: string;
}

export interface BehaviourResult {
  id: string;
  job_id: string;
  track_id: number;
  frame_number: number;
  timestamp_seconds: number;
  behaviour_type: ObservableBehaviour;
  confidence: number;
  student_id?: string | null;
  metadata_json: {
    student_label?: string;
    model?: string;
    window_start_sec?: number;
    window_end_sec?: number;
    probability_distribution?: Record<ObservableBehaviour, number>;
  };
  created_at: string;
}

export interface TemporalBehaviourBucket {
  timestamp_start: number;
  timestamp_end: number;
  looking_toward_instruction_count: number;
  reading_count: number;
  writing_count: number;
  peer_interaction_count: number;
  looking_away_count: number;
  total_detections: number;
  engagement_index: number;
}

export interface StudentTrackMetrics {
  track_id: number;
  student_id?: string | null;
  display_label: string; // e.g., "Student 01"
  looking_toward_instruction_pct: number;
  reading_pct: number;
  writing_pct: number;
  peer_interaction_pct: number;
  looking_away_pct: number;
  dominant_behaviour: ObservableBehaviour;
}

export interface SessionAnalyticsSummary {
  job_id: string;
  video_id: string;
  session_id: string;
  status: string;
  total_frames_analyzed: number;
  total_tracks_identified: number;
  overall_engagement_score: number;
  timeline_buckets: TemporalBehaviourBucket[];
  track_metrics: StudentTrackMetrics[];
}

export interface ModelInfo {
  model_version: string;
  spatial_backbone: string;
  temporal_model_type: string;
  hidden_dim: number;
  num_layers: number;
  sequence_length: number;
  sampling_fps: number;
}

export interface TemporalStudentProfile {
  track_id: number;
  timeline: Array<{ behaviour: ObservableBehaviour; start_time: number; end_time: number; duration_seconds: number; confidence: number }>;
  behaviour_distribution: Record<string, number>;
  transition_matrix: number[][];
  behaviour_duration: Record<string, number>;
  total_observed_duration: number;
  segment_count: number;
  temporal_coverage: number;
}

export interface TemporalState {
  start_time: number;
  end_time: number;
  state: string;
  behaviour_distribution: Record<string, number>;
  students_contributing: number;
}

export interface TemporalEntropy {
  timestamp: number;
  entropy: number;
  behaviour_distribution: Record<string, number>;
}

export interface TemporalChangePoint {
  timestamp: number;
  change_score: number;
  previous_state: string;
  new_state: string;
}

export interface TemporalCoverage {
  manual_reference_student_count: number | null;
  detected_student_count: number;
  tracked_student_count: number;
  temporal_ready_track_count: number;
  detection_coverage: number;
  tracking_coverage: number;
  temporal_coverage: number;
  temporal_observation_windows?: number;
  evidence_status?: 'SUFFICIENT' | 'LIMITED' | 'INSUFFICIENT';
  evidence_note?: string;
}

export interface TemporalAnalyticsProfile {
  students: TemporalStudentProfile[];
  transitions: Array<{ track_id: number | null; from_behaviour: string; to_behaviour: string; count: number; probability: number }>;
  classroom_states: TemporalState[];
  entropy: TemporalEntropy[];
  change_points: TemporalChangePoint[];
  coverage: TemporalCoverage | null;
}

export interface AuditLog {
  id: string;
  user_id?: string | null;
  user_email?: string | null;
  action: string;
  resource_type: string;
  resource_id?: string | null;
  ip_address?: string | null;
  details_json: Record<string, any>;
  created_at: string;
}

export interface DropzoneScanResult {
  files_scanned: number;
  matched_and_ingested: number;
  unmatched_files: string[];
  errors: string[];
}

export interface TimetableExtractedSlot {
  id?: string | null;
  day_of_week: string;
  period_name?: string | null;
  start_time: string;
  end_time: string;
  subject_code: string;
  subject_name: string;
  section_name: string;
  room_number?: string | null;
  confidence: number;
}

export interface TimetableExtractResponse {
  filename: string;
  total_slots_extracted: number;
  slots: TimetableExtractedSlot[];
  message: string;
}

export interface TimetableConfirmResponse {
  saved_slots_count: number;
  created_subjects_count: number;
  created_sections_count: number;
  message: string;
}

export interface FacultyInsight {
  id: string;
  job_id: string;
  category: 'PACING' | 'VARIETY' | 'INTERACTION' | 'ATTENTION_PATTERNS' | 'COVERAGE' | string;
  start_time?: number | null;
  end_time?: number | null;
  observation: string;
  pedagogical_context: string;
  suggested_action: string;
  coverage_context?: string | null;
  confidence: number;
  created_at: string;
}

// --- Exam Mode & Invigilator Dashboard Types ---
export interface CameraSummary {
  id: string;
  camera_name: string;
  device_code: string;
  source_type: 'MP4_DEMO' | 'RTSP_STREAM' | 'NVR_CHANNEL' | string;
  location_in_room: string;
  status: string;
  is_healthy: boolean;
  demo_video_path?: string | null;
}

export interface RoomCameraItem {
  id: string;
  room_number: string;
  building: string;
  floor: number;
  capacity: number;
  cameras: CameraSummary[];
}

// ==========================================
// HOD / Admin Academic & Teaching Analytics
// ==========================================

export interface TeachingDeliveryBreakdown {
  didactic_lectern_pct: number;
  active_aisle_circulation_pct: number;
  board_projection_presentation_pct: number;
  desk_consultation_facilitation_pct: number;
}

export interface SessionDeliveryRecord {
  session_id: string;
  title: string;
  session_date: string;
  start_time: string;
  end_time: string;
  duration_minutes: number;
  dominant_mode: string;
  faculty_mobility_score: number;
  student_attention_pct: number;
  shannon_entropy: number;
  pedagogical_note: string;
}

export interface SectionSummary {
  id: string;
  name: string;
  academic_year: string;
  semester: string;
  subject_id?: string | null;
  subject_code?: string | null;
  subject_name?: string | null;
  student_count: number;
  subjects_count: number;
  total_sessions_conducted: number;
  avg_engagement_pct: number;
  faculty_mobility_pct: number;
}

export interface SectionSubjectDetail {
  subject_id: string;
  subject_code: string;
  subject_name: string;
  section_id: string;
  section_name: string;
  faculty_id: string;
  faculty_name: string;
  faculty_email: string;
  faculty_designation?: string | null;
  total_classes_conducted: number;
  avg_student_engagement: number;
  faculty_mobility_score: number;
  dominant_teaching_mode: string;
}

export interface FacultyTeachingSummary {
  subject_id: string;
  subject_code: string;
  subject_name: string;
  section_id: string;
  section_name: string;
  faculty_id: string;
  faculty_name: string;
  faculty_email: string;
  faculty_designation?: string | null;
  total_classes_conducted: number;
  total_instructional_hours: number;
  teaching_delivery: TeachingDeliveryBreakdown;
  interaction_density_score: number;
  classroom_mobility_index: string;
  student_synchronous_attention_pct: number;
  student_active_collaboration_pct: number;
  average_shannon_entropy: number;
  pedagogical_summary: string;
  session_history: SessionDeliveryRecord[];
}

export interface FacultySectionAssignment {
  section_id: string;
  section_name: string;
  subject_id: string;
  subject_code: string;
  subject_name: string;
  total_classes: number;
}

export interface FacultyItem {
  id: string;
  full_name: string;
  email: string;
  department: string;
  designation?: string | null;
  role: string;
  sections_count: number;
  assignments: FacultySectionAssignment[];
}

export interface CrossSectionItem {
  section_id: string;
  section_name: string;
  subject_code: string;
  subject_name: string;
  total_classes: number;
  avg_student_attention_pct: number;
  faculty_mobility_pct: number;
  interactive_discussion_pct: number;
  shannon_entropy: number;
  dominant_style: string;
  teaching_breakdown: TeachingDeliveryBreakdown;
}

export interface FacultyCrossSectionComparison {
  faculty_id: string;
  faculty_name: string;
  faculty_email: string;
  department: string;
  designation?: string | null;
  total_sections_taught: number;
  sections: CrossSectionItem[];
  cross_section_consistency_score: number;
  comparative_analysis: string;
  hod_actionable_insight: string;
}

export interface TimetableUploadResponse {
  message: string;
  faculties_count: number;
  sections_count: number;
  subjects_count: number;
  rooms_count: number;
  slots_count: number;
  date_processed: string;
  status: string;
}

export interface VideoSourceInfo {
  class_number: number;
  label: string;
  file_path: string;
  file_size_mb: number;
  description: string;
}

export interface ClassSessionAnalysisDetail {
  class_number: number;
  session_id: string;
  title: string;
  session_date: string;
  duration_minutes: number;
  video_filename: string;
  faculty_mobility_pct: number;
  didactic_lectern_pct: number;
  aisle_circulation_pct: number;
  board_exposition_pct: number;
  desk_consultation_pct: number;
  student_focus_pct: number;
  cell_phone_distraction_pct: number;
  teacher_idle_stationary_pct: number;
  shannon_entropy: number;
  dominant_mode: string;
  fiac_category: string;
  pedagogical_notes: string;
}

export interface ThreeClassSummaryResponse {
  faculty_id: string;
  faculty_name: string;
  faculty_email: string;
  faculty_designation?: string | null;
  section_id: string;
  section_name: string;
  subject_id: string;
  subject_code: string;
  subject_name: string;
  classes: ClassSessionAnalysisDetail[];
  aggregate_mobility_index: string;
  aggregate_student_focus_pct: number;
  aggregate_teacher_active_pct: number;
  cell_phone_idle_alert: string;
  longitudinal_trend_summary: string;
  fiac_matrix_summary: string;
  video_sources: VideoSourceInfo[];
}

export interface TodayScheduleSlot {
  period: number;
  time_window: string;
  day_of_week: string;
  faculty_name: string;
  section_name: string;
  subject_code: string;
  subject_name: string;
  room_number: string;
}

export interface DepartmentTimetableOverview {
  current_date: string;
  day_of_week: string;
  total_faculties: number;
  total_sections: number;
  total_subjects: number;
  today_active_classes_count: number;
  today_schedule: TodayScheduleSlot[];
}


