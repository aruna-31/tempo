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
