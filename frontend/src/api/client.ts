import {
  AnalysisJob,
  AuditLog,
  BehaviourResult,
  ClassSession,
  DropzoneScanResult,
  Faculty,
  FacultyInsight,
  FacultySchedule,
  JobLogEntry,
  ModelInfo,
  TemporalAnalyticsProfile,
  TemporalStudentProfile,
  Room,
  Section,
  SessionAnalyticsSummary,
  Student,
  StudentTrackResult,
  Subject,
  TimetableConfirmResponse,
  TimetableExtractResponse,
  TodayClass,
  Video
} from '../types';

const API_BASE_URL = '/api/v1';

export function extractErrorMessage(errorData: any, fallback: string = 'An error occurred'): string {
  if (!errorData) return fallback;
  if (typeof errorData === 'string') return errorData;
  if (typeof errorData.detail === 'string') return errorData.detail;
  if (Array.isArray(errorData.detail)) {
    return errorData.detail
      .map((item: any) => (typeof item === 'string' ? item : item?.msg || item?.message || JSON.stringify(item)))
      .join('; ');
  }
  if (errorData.detail && typeof errorData.detail === 'object') {
    return errorData.detail.msg || errorData.detail.message || JSON.stringify(errorData.detail);
  }
  if (typeof errorData.message === 'string') return errorData.message;
  return fallback;
}

class ApiClient {
  private getToken(): string | null {
    return localStorage.getItem('tempo_access_token');
  }

  getRefreshToken(): string | null {
    return localStorage.getItem('tempo_refresh_token');
  }

  clearTokens(): void {
    localStorage.removeItem('tempo_access_token');
    localStorage.removeItem('tempo_refresh_token');
    localStorage.removeItem('tempo_faculty_user');
  }

  /** Exchange the stored refresh token for a fresh access token. */
  async refreshAccessToken(): Promise<boolean> {
    const refreshToken = this.getRefreshToken();
    if (!refreshToken) return false;

    try {
      const resp = await fetch(`${API_BASE_URL}/auth/refresh`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refresh_token: refreshToken }),
      });
      if (!resp.ok) return false;
      const data = await resp.json();
      localStorage.setItem('tempo_access_token', data.access_token);
      if (data.refresh_token) {
        localStorage.setItem('tempo_refresh_token', data.refresh_token);
      }
      return true;
    } catch {
      return false;
    }
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {},
    isRetry: boolean = false
  ): Promise<T> {
    const token = this.getToken();
    const headers: Record<string, string> = {
      ...(options.headers as Record<string, string> || {}),
    };

    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }

    if (!(options.body instanceof FormData)) {
      headers['Content-Type'] = 'application/json';
    }

    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      ...options,
      headers,
    });

    if (response.status === 401) {
      // Try silent token refresh once, then retry the original request.
      if (!isRetry && (await this.refreshAccessToken())) {
        return this.request<T>(endpoint, options, true);
      }
      this.clearTokens();
      throw new Error('Unauthorized session expired. Please log in again.');
    }

    if (!response.ok) {
      let errorDetail = 'An error occurred';
      try {
        const errorJson = await response.json();
        errorDetail = extractErrorMessage(errorJson, response.statusText || 'An error occurred');
      } catch {
        errorDetail = response.statusText || 'An error occurred';
      }
      throw new Error(errorDetail);
    }

    if (response.status === 204) {
      return {} as T;
    }

    return response.json();
  }

  // --- Auth Endpoints ---
  async login(data: { email: string; password: string }): Promise<{ access_token: string; refresh_token: string; token_type: string; expires_in: number }> {
    const response = await fetch(`${API_BASE_URL}/auth/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      const errorJson = await response.json().catch(() => ({}));
      const msg = extractErrorMessage(errorJson, 'Login failed. Please verify your faculty email and password.');
      throw new Error(msg);
    }

    const resData = await response.json();
    localStorage.setItem('tempo_access_token', resData.access_token);
    if (resData.refresh_token) {
      localStorage.setItem('tempo_refresh_token', resData.refresh_token);
    }
    return resData;
  }

  async register(data: {
    email: string;
    password: string;
    full_name: string;
    department: string;
    designation?: string;
  }): Promise<Faculty> {
    return this.request<Faculty>('/auth/register', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async getMe(): Promise<Faculty> {
    return this.request<Faculty>('/auth/me');
  }


  // --- Academic: Subjects, Sections, Students ---
  async getSubjects(): Promise<Subject[]> {
    return this.request<Subject[]>('/subjects');
  }

  async createSubject(data: { code: string; name: string }): Promise<Subject> {
    return this.request<Subject>('/subjects', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async getSections(subjectId: string): Promise<Section[]> {
    return this.request<Section[]>(`/subjects/${subjectId}/sections`);
  }

  async createSection(subjectId: string, data: { name: string; academic_year: string; semester: string }): Promise<Section> {
    return this.request<Section>(`/subjects/${subjectId}/sections`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async getStudents(sectionId: string): Promise<Student[]> {
    return this.request<Student[]>(`/sections/${sectionId}/students`);
  }

  async createStudent(sectionId: string, data: { roll_number: string; name: string; email: string }): Promise<Student> {
    return this.request<Student>(`/sections/${sectionId}/students`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  // --- Rooms & Cameras ---
  async getRooms(): Promise<Room[]> {
    return this.request<Room[]>('/rooms');
  }

  async createRoom(data: { room_number: string; building: string; floor: number; capacity: number }): Promise<Room> {
    return this.request<Room>('/rooms', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async addCameraToRoom(roomId: string, data: { camera_name: string; device_code: string; location_in_room?: string }): Promise<any> {
    return this.request<any>(`/rooms/${roomId}/cameras`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  // --- Faculty Schedules & Timetables ---
  async getSchedules(): Promise<FacultySchedule[]> {
    return this.request<FacultySchedule[]>('/schedules');
  }

  /** XHR helper with bearer auth: 401 -> silent refresh -> one retry. */
  private authXhr<T>(
    method: string,
    url: string,
    formData: FormData,
    onProgress?: (pct: number) => void
  ): Promise<T> {
    const send = (isRetry: boolean) =>
      new Promise<T>((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open(method, `${API_BASE_URL}${url}`);
        const token = this.getToken();
        if (token) {
          xhr.setRequestHeader('Authorization', `Bearer ${token}`);
        }
        if (onProgress) {
          xhr.upload.onprogress = (e) => {
            if (e.lengthComputable) onProgress(Math.round((e.loaded / e.total) * 100));
          };
        }
        xhr.onload = async () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            try {
              resolve(JSON.parse(xhr.responseText));
            } catch {
              reject(new Error('Invalid server JSON response'));
            }
            return;
          }
          if (xhr.status === 401 && !isRetry && (await this.refreshAccessToken())) {
            try {
              resolve(await send(true));
            } catch (e) {
              reject(e);
            }
            return;
          }
          if (xhr.status === 401) this.clearTokens();
          try {
            const errJson = JSON.parse(xhr.responseText);
            reject(new Error(extractErrorMessage(errJson, xhr.statusText || 'Request failed')));
          } catch {
            reject(new Error(xhr.statusText || 'Request failed'));
          }
        };
        xhr.onerror = () => reject(new Error('Network connection error'));
        xhr.send(formData);
      });
    return send(false);
  }

  /**
   * Uploads a timetable image/PDF and returns OCR-extracted candidate slots.
   * Slots may contain empty fields - the faculty completes/edits them before confirming.
   */
  async extractTimetable(file: File): Promise<TimetableExtractResponse> {
    const formData = new FormData();
    formData.append('file', file);
    return this.authXhr<TimetableExtractResponse>('POST', '/schedules/extract-timetable', formData);
  }

  /** Saves faculty-confirmed/edited timetable slots to PostgreSQL. */
  async confirmTimetable(data: {
    slots: Array<{
      day_of_week: string;
      start_time: string;
      end_time: string;
      subject_code: string;
      subject_name?: string;
      section_name?: string;
      room_number?: string;
      academic_year?: string;
      semester?: string;
    }>;
    clear_existing: boolean;
  }): Promise<TimetableConfirmResponse> {
    return this.request<TimetableConfirmResponse>('/schedules/confirm-timetable', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async getTodayClasses(): Promise<TodayClass[]> {
    return this.request<TodayClass[]>('/schedules/today');
  }

  async createSchedule(data: {
    subject_id: string;
    section_id: string;
    room_id?: string;
    day_of_week: string;
    start_time: string;
    end_time: string;
    academic_year?: string;
    semester?: string;
  }): Promise<FacultySchedule> {
    return this.request<FacultySchedule>('/schedules', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async generateSessionsFromSchedule(data: { start_date: string; end_date: string }): Promise<{ message: string; generated_count: number }> {
    return this.request<{ message: string; generated_count: number }>('/schedules/generate-sessions', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  // --- Sessions & Videos ---
  async getSessions(sectionId?: string): Promise<ClassSession[]> {
    const url = sectionId ? `/sessions?section_id=${sectionId}` : '/sessions';
    return this.request<ClassSession[]>(url);
  }

  async createSession(data: {
    section_id: string;
    title: string;
    session_date: string;
    start_time: string;
    end_time: string;
    room_number?: string;
    room_id?: string;
  }): Promise<ClassSession> {
    return this.request<ClassSession>('/sessions', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async getSessionVideos(sessionId: string): Promise<Video[]> {
    return this.request<Video[]>(`/sessions/${sessionId}/videos`);
  }

  async uploadVideo(sessionId: string, file: File, onProgress?: (pct: number) => void): Promise<Video> {
    const resp = await this.uploadVideoWithJob(sessionId, file, onProgress);
    return resp.video;
  }

  /**
   * Uploads a video for the exact ClassSession and returns the auto-created AnalysisJob.
   * The backend creates exactly one job per upload - no second job is created client-side.
   */
  async uploadVideoWithJob(
    sessionId: string,
    file: File,
    onProgress?: (pct: number) => void
  ): Promise<{ video: Video; analysis_job: AnalysisJob; message: string }> {
    const formData = new FormData();
    formData.append('file', file);
    return this.authXhr<{ video: Video; analysis_job: AnalysisJob; message: string }>(
      'POST',
      `/sessions/${sessionId}/videos`,
      formData,
      onProgress
    );
  }

  // --- Ingestion & Storage ---
  async scanDropzone(): Promise<DropzoneScanResult> {
    return this.request<DropzoneScanResult>('/ingest/scan-dropzone', { method: 'POST' });
  }

  async getStorageStats(): Promise<any> {
    return this.request<any>('/ingest/storage-stats');
  }

  async getAuditLogs(): Promise<AuditLog[]> {
    return this.request<AuditLog[]>('/ingest/audit-logs');
  }

  // --- ML Analysis Jobs & Results ---
  async createJob(videoId: string, config: Record<string, any> = {}): Promise<AnalysisJob> {
    return this.request<AnalysisJob>('/analysis/jobs', {
      method: 'POST',
      body: JSON.stringify({ video_id: videoId, config_json: config }),
    });
  }

  async getJob(jobId: string): Promise<AnalysisJob> {
    return this.request<AnalysisJob>(`/analysis/jobs/${jobId}`);
  }

  async retryJob(jobId: string): Promise<AnalysisJob> {
    return this.request<AnalysisJob>(`/analysis/jobs/${jobId}/retry`, { method: 'POST' });
  }

  async getJobLogs(jobId: string): Promise<{ job_id: string; status: string; current_stage: string; retry_count: number; execution_time_seconds: number | null; logs: JobLogEntry[] }> {
    return this.request<{ job_id: string; status: string; current_stage: string; retry_count: number; execution_time_seconds: number | null; logs: JobLogEntry[] }>(`/analysis/jobs/${jobId}/logs`);
  }

  async getVideoJobs(videoId: string): Promise<AnalysisJob[]> {
    return this.request<AnalysisJob[]>(`/analysis/videos/${videoId}/jobs`);
  }

  async getJobTracks(jobId: string): Promise<StudentTrackResult[]> {
    return this.request<StudentTrackResult[]>(`/analysis/jobs/${jobId}/tracks`);
  }

  async getJobBehaviours(jobId: string, trackId?: number, limit: number = 500): Promise<BehaviourResult[]> {
    let url = `/analysis/jobs/${jobId}/behaviours?limit=${limit}`;
    if (trackId !== undefined) {
      url += `&track_id=${trackId}`;
    }
    return this.request<BehaviourResult[]>(url);
  }

  async getJobSummary(jobId: string, bucketSeconds: number = 60): Promise<SessionAnalyticsSummary> {
    return this.request<SessionAnalyticsSummary>(`/analysis/jobs/${jobId}/summary?bucket_seconds=${bucketSeconds}`);
  }

  async getModelInfo(): Promise<ModelInfo> {
    return this.request<ModelInfo>('/analysis/model-info');
  }

  async getTemporalProfile(jobId: string): Promise<TemporalAnalyticsProfile> {
    return this.request<TemporalAnalyticsProfile>(`/analysis/jobs/${jobId}/temporal/profile`);
  }

  async getTemporalStudent(jobId: string, trackId: number): Promise<TemporalStudentProfile> {
    return this.request<TemporalStudentProfile>(`/analysis/jobs/${jobId}/temporal/students/${trackId}`);
  }

  async getFacultyInsights(jobId: string): Promise<FacultyInsight[]> {
    return this.request<FacultyInsight[]>(`/analysis/jobs/${jobId}/faculty-insights`);
  }
}

export const api = new ApiClient();

