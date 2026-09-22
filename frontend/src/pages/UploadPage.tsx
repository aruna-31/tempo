import React, { useEffect, useRef, useState } from 'react';
import { useLocation, Link } from 'react-router-dom';
import { api } from '../api/client';
import { AnalysisJob, ClassSession } from '../types';
import { UploadCloud, FileVideo, CheckCircle2, AlertCircle, ArrowRight, Cpu, XCircle } from 'lucide-react';

// PROTOTYPE: upload flow for the exact selected ClassSession.
// The upload endpoint automatically creates the AnalysisJob; this page uses that
// job (no duplicate job is created) and displays processing state inline here -
// the separate Processing Status page was removed from the prototype.
export const UploadPage: React.FC = () => {
  const location = useLocation();

  const [sessions, setSessions] = useState<ClassSession[]>([]);
  const [selectedSessionId, setSelectedSessionId] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);

  // Inline processing state
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<string | null>(null);
  const [jobProgress, setJobProgress] = useState(0);
  const [jobError, setJobError] = useState<string | null>(null);
  const pollRef = useRef<number | null>(null);

  useEffect(() => {
    const searchParams = new URLSearchParams(location.search);
    const prefillSessionId = searchParams.get('session_id');

    api.getSessions().then((list) => {
      setSessions(list);
      if (prefillSessionId && list.some((s) => s.id === prefillSessionId)) {
        setSelectedSessionId(prefillSessionId);
      } else if (list.length > 0) {
        setSelectedSessionId(list[0].id);
      }
    });
  }, [location.search]);

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollRef.current !== null) {
        window.clearInterval(pollRef.current);
      }
    };
  }, []);

  const startPolling = (jobId: string) => {
    setActiveJobId(jobId);
    pollRef.current = window.setInterval(async () => {
      try {
        const job: AnalysisJob = await api.getJob(jobId);
        setJobStatus(job.status);
        setJobProgress(job.progress_pct);
        if (job.status === 'FAILED') {
          setJobError(job.error_message || 'Analysis failed.');
        }
        if (job.status === 'COMPLETED' || job.status === 'FAILED') {
          if (pollRef.current !== null) {
            window.clearInterval(pollRef.current);
            pollRef.current = null;
          }
        }
      } catch {
        // transient polling errors are ignored; next tick retries
      }
    }, 2000);
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndSetFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      validateAndSetFile(e.target.files[0]);
    }
  };

  const validateAndSetFile = (selectedFile: File) => {
    setError(null);
    const validExtensions = ['.mp4', '.mov', '.webm'];
    const hasValidExt = validExtensions.some((ext) =>
      selectedFile.name.toLowerCase().endsWith(ext)
    );

    if (!hasValidExt) {
      setError('Invalid video file format. Supported formats: .mp4, .mov, .webm');
      return;
    }

    if (selectedFile.size > 1024 * 1024 * 1024) {
      setError('File size exceeds the 1GB limit.');
      return;
    }

    setFile(selectedFile);
  };

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedSessionId || !file) {
      setError('Please select a class session and video file.');
      return;
    }

    setError(null);
    setUploading(true);
    setProgress(0);
    setJobStatus(null);
    setJobError(null);

    try {
      // The backend creates the AnalysisJob automatically for this exact ClassSession.
      const response = await api.uploadVideoWithJob(selectedSessionId, file, (pct) => {
        setProgress(pct);
      });
      setUploading(false);
      startPolling(response.analysis_job.id);
    } catch (err: any) {
      setError(err.message || 'Video upload failed.');
      setUploading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white tracking-tight">Upload Classroom Video</h1>
        <p className="text-xs text-slate-400">
          Upload recorded lecture footage to run YOLO person tracking, ByteTrack identity preservation, ResNet-18 + temporal sequence modeling, and observable behaviour recognition
        </p>
      </div>

      <div className="glass-panel p-8 border border-slate-800">
        {error && (
          <div className="mb-6 flex items-center gap-2.5 rounded-xl border border-rose-500/30 bg-rose-500/10 p-4 text-xs text-rose-300">
            <AlertCircle className="h-4 w-4 flex-shrink-0 text-rose-400" />
            <span>{error}</span>
          </div>
        )}

        <form onSubmit={handleUpload} className="space-y-6">
          {/* Session Selection */}
          <div>
            <label className="block text-xs font-semibold uppercase tracking-wider text-slate-300 mb-2">
              Target Class Session
            </label>
            <select
              required
              value={selectedSessionId}
              onChange={(e) => setSelectedSessionId(e.target.value)}
              className="input-field"
            >
              {sessions.length === 0 && <option value="">No sessions available</option>}
              {sessions.map((sess) => (
                <option key={sess.id} value={sess.id}>
                  {sess.title} ({sess.session_date})
                </option>
              ))}
            </select>
          </div>

          {/* Drag and Drop Zone */}
          <div>
            <label className="block text-xs font-semibold uppercase tracking-wider text-slate-300 mb-2">
              Video File (.mp4, .mov, .webm)
            </label>
            <div
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              className={`relative flex flex-col items-center justify-center rounded-2xl border-2 border-dashed p-8 text-center transition-all ${
                dragActive
                  ? 'border-indigo-500 bg-indigo-950/30'
                  : file
                  ? 'border-emerald-500/60 bg-emerald-950/20'
                  : 'border-slate-800 bg-slate-900/40 hover:border-slate-700'
              }`}
            >
              <input
                type="file"
                accept=".mp4,.mov,.webm,video/*"
                onChange={handleFileInput}
                className="absolute inset-0 cursor-pointer opacity-0"
              />

              {file ? (
                <div className="space-y-2">
                  <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-emerald-500/20 text-emerald-400 mx-auto">
                    <FileVideo className="h-6 w-6" />
                  </div>
                  <p className="font-semibold text-sm text-white">{file.name}</p>
                  <p className="text-xs text-slate-400 font-mono">
                    {(file.size / (1024 * 1024)).toFixed(2)} MB
                  </p>
                  <span className="text-[11px] text-emerald-400 font-medium flex items-center justify-center gap-1">
                    <CheckCircle2 className="h-3.5 w-3.5" />
                    <span>File Selected • Click to change</span>
                  </span>
                </div>
              ) : (
                <div className="space-y-2">
                  <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-indigo-500/20 text-indigo-400 mx-auto">
                    <UploadCloud className="h-6 w-6" />
                  </div>
                  <p className="font-semibold text-sm text-white">
                    Drag and drop your lecture video here, or <span className="text-indigo-400">browse</span>
                  </p>
                  <p className="text-xs text-slate-500">
                    Supports MP4, MOV, WebM up to 1GB
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Progress Bar if uploading */}
          {uploading && (
            <div className="space-y-2">
              <div className="flex justify-between text-xs font-mono">
                <span className="text-indigo-400">Uploading Video File...</span>
                <span className="text-slate-300">{progress}%</span>
              </div>
              <div className="h-2 w-full rounded-full bg-slate-800 overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-indigo-500 to-sky-400 rounded-full transition-all duration-200"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          )}

          <button
            type="submit"
            disabled={!file || uploading}
            className="btn-primary w-full py-3 text-sm"
          >
            <span>{uploading ? `Uploading (${progress}%)...` : 'Upload and Start Analysis'}</span>
            <ArrowRight className="h-4 w-4" />
          </button>
        </form>

        {/* Inline Processing State */}
        {activeJobId && (
          <div className="mt-8 rounded-xl border border-slate-800 bg-slate-900/60 p-5 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-xs font-semibold text-white">
                <Cpu className="h-4 w-4 text-indigo-400" />
                <span>ML Pipeline Processing</span>
              </div>
              <span className="font-mono text-[11px] text-slate-500">Job {activeJobId.substring(0, 8)}</span>
            </div>

            {jobStatus !== 'COMPLETED' && jobStatus !== 'FAILED' && (
              <>
                <div className="flex justify-between text-xs font-mono">
                  <span className="text-indigo-400">Status: {jobStatus || 'QUEUED'}</span>
                  <span className="text-slate-300">{jobProgress}%</span>
                </div>
                <div className="h-2 w-full rounded-full bg-slate-800 overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-emerald-500 to-sky-400 rounded-full transition-all duration-300"
                    style={{ width: `${jobProgress}%` }}
                  />
                </div>
                <p className="text-[11px] text-slate-500">
                  YOLO person detection → ByteTrack tracking → ResNet-18 features → temporal behaviour classification
                </p>
              </>
            )}

            {jobStatus === 'COMPLETED' && (
              <div className="space-y-2">
                <p className="text-xs text-emerald-400 flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4" />
                  Analysis completed successfully. Results and annotated video are ready.
                </p>
                <Link
                  to={`/results?job_id=${activeJobId}`}
                  className="btn-primary text-xs inline-flex items-center gap-2"
                >
                  <span>View Results</span>
                  <ArrowRight className="h-3.5 w-3.5" />
                </Link>
              </div>
            )}

            {jobStatus === 'FAILED' && (
              <div className="space-y-1">
                <p className="text-xs text-rose-400 flex items-center gap-2">
                  <XCircle className="h-4 w-4" />
                  Analysis failed. The actual error was stored with the job.
                </p>
                <p className="text-[11px] font-mono text-rose-300/90 break-words">{jobError}</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default UploadPage;
