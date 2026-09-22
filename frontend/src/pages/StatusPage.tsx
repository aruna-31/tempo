import React, { useEffect, useState } from 'react';
import { useLocation, useNavigate, Link } from 'react-router-dom';
import { api } from '../api/client';
import { AnalysisJob, JobLogEntry } from '../types';
import {
  Cpu,
  CheckCircle2,
  AlertCircle,
  Clock,
  ArrowRight,
  RotateCw,
  Sparkles,
  Terminal,
  RefreshCw,
  ShieldCheck,
  Radio
} from 'lucide-react';

export const StatusPage: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();

  const searchParams = new URLSearchParams(location.search);
  const jobId = searchParams.get('job_id');

  const [job, setJob] = useState<AnalysisJob | null>(null);
  const [logs, setLogs] = useState<JobLogEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isPolling, setIsPolling] = useState(true);
  const [showLogsModal, setShowLogsModal] = useState(false);
  const [retrying, setRetrying] = useState(false);

  useEffect(() => {
    if (!jobId) {
      setError('No Job ID specified in URL. Please upload a video or select an existing job.');
      setIsPolling(false);
      return;
    }

    const fetchStatus = async () => {
      try {
        const [data, logsData] = await Promise.all([
          api.getJob(jobId),
          api.getJobLogs(jobId).catch(() => ({ logs: [] }))
        ]);
        setJob(data);
        setLogs(logsData.logs || []);

        if (data.status === 'COMPLETED' || data.status === 'FAILED') {
          setIsPolling(false);
        }
      } catch (err: any) {
        setError(err.message || 'Failed to fetch job status');
        setIsPolling(false);
      }
    };

    fetchStatus();

    const interval = setInterval(() => {
      if (isPolling) {
        fetchStatus();
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [jobId, isPolling]);

  const handleRetry = async () => {
    if (!jobId) return;
    try {
      setRetrying(true);
      const updated = await api.retryJob(jobId);
      setJob(updated);
      setIsPolling(true);
      setError(null);
    } catch (err: any) {
      alert(err.message || 'Failed to retry analysis job');
    } finally {
      setRetrying(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Temporal ML Processing Status</h1>
          <p className="text-xs text-slate-400">
            Real-time tracking of individual-student detection, tracking, and temporal behavior inference
          </p>
        </div>

        {job && (
          <button
            onClick={() => setShowLogsModal(true)}
            className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-xs font-mono flex items-center gap-1.5 border border-slate-700 transition"
          >
            <Terminal className="w-3.5 h-3.5 text-indigo-400" />
            <span>Execution Logs ({logs.length})</span>
          </button>
        )}
      </div>

      <div className="glass-panel p-8 border border-slate-800 space-y-6">
        {error ? (
          <div className="flex items-start gap-2.5 rounded-xl border border-rose-500/30 bg-rose-500/10 p-4 text-xs text-rose-300">
            <AlertCircle className="h-4 w-4 flex-shrink-0 text-rose-400 mt-0.5" />
            <span>{error}</span>
          </div>
        ) : !job ? (
          <div className="flex items-center justify-center py-12 gap-3 text-sm text-slate-400">
            <RotateCw className="h-5 w-5 animate-spin text-indigo-400" />
            <span>Connecting to ML pipeline worker...</span>
          </div>
        ) : (
          <div className="space-y-6">
            {/* Header Status & Badge */}
            <div className="flex items-center justify-between border-b border-slate-800 pb-4">
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-[11px] font-mono text-slate-500 uppercase">Analysis Job ID</span>
                  {job.source_type && (
                    <span className="px-2 py-0.5 bg-slate-800 text-slate-400 rounded text-[10px] uppercase font-mono">
                      Source: {job.source_type}
                    </span>
                  )}
                </div>
                <p className="font-mono text-sm font-semibold text-white mt-0.5">{job.id}</p>
              </div>

              <span
                className={`badge text-xs px-3 py-1 ${
                  job.status === 'COMPLETED'
                    ? 'badge-reading'
                    : job.status === 'PROCESSING' || job.status === 'RETRYING'
                    ? 'badge-instruction'
                    : job.status === 'FAILED'
                    ? 'badge-away'
                    : 'badge-peer'
                }`}
              >
                {(job.status === 'PROCESSING' || job.status === 'RETRYING') && (
                  <RotateCw className="h-3 w-3 animate-spin mr-1 inline" />
                )}
                {job.status}
              </span>
            </div>

            {/* Progress Bar & Stage description */}
            <div className="space-y-3">
              <div className="flex justify-between text-xs">
                <span className="font-medium text-slate-300 flex items-center gap-2">
                  {job.status === 'COMPLETED'
                    ? '✓ Complete: Tracks, temporal sequences & behaviour results generated'
                    : job.status === 'PROCESSING'
                    ? `Executing Stage: ${job.current_stage || 'ML Pipeline'}...`
                    : job.status === 'RETRYING'
                    ? `Retrying Stage: ${job.current_stage || 'ML Pipeline'} (Attempt ${(job.retry_count || 0) + 1}/${(job.max_retries || 3) + 1})...`
                    : job.status === 'FAILED'
                    ? 'Analysis Pipeline Encountered an Error'
                    : 'Queued in ML Thread Pool Worker'}
                </span>
                <span className="font-mono font-bold text-indigo-400">{job.progress_pct}%</span>
              </div>

              <div className="h-3 w-full rounded-full bg-slate-800 overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-300 ${
                    job.status === 'FAILED'
                      ? 'bg-rose-500'
                      : 'bg-gradient-to-r from-indigo-500 via-sky-400 to-emerald-400'
                  }`}
                  style={{ width: `${job.progress_pct}%` }}
                />
              </div>
            </div>

            {/* Pipeline Stage Indicators */}
            <div className="grid grid-cols-4 gap-3 pt-2">
              <div
                className={`p-3 rounded-xl border text-xs text-center ${
                  job.progress_pct >= 20
                    ? 'border-emerald-500/40 bg-emerald-950/20 text-emerald-300'
                    : 'border-slate-800 bg-slate-900/30 text-slate-500'
                }`}
              >
                <p className="font-semibold">1. Frame Sampling</p>
                <p className="text-[10px] mt-0.5">2 FPS Sampling</p>
              </div>

              <div
                className={`p-3 rounded-xl border text-xs text-center ${
                  job.progress_pct >= 40
                    ? 'border-emerald-500/40 bg-emerald-950/20 text-emerald-300'
                    : job.progress_pct >= 20
                    ? 'border-indigo-500/40 bg-indigo-950/20 text-indigo-300'
                    : 'border-slate-800 bg-slate-900/30 text-slate-500'
                }`}
              >
                <p className="font-semibold">2. Person Detection</p>
                <p className="text-[10px] mt-0.5">YOLO + Context</p>
              </div>

              <div
                className={`p-3 rounded-xl border text-xs text-center ${
                  job.progress_pct >= 60
                    ? 'border-emerald-500/40 bg-emerald-950/20 text-emerald-300'
                    : job.progress_pct >= 40
                    ? 'border-indigo-500/40 bg-indigo-950/20 text-indigo-300'
                    : 'border-slate-800 bg-slate-900/30 text-slate-500'
                }`}
              >
                <p className="font-semibold">3. ByteTrack</p>
                <p className="text-[10px] mt-0.5">Session Track IDs</p>
              </div>

              <div
                className={`p-3 rounded-xl border text-xs text-center ${
                  job.progress_pct === 100
                    ? 'border-emerald-500/40 bg-emerald-950/20 text-emerald-300'
                    : job.progress_pct >= 60
                    ? 'border-indigo-500/40 bg-indigo-950/20 text-indigo-300'
                    : 'border-slate-800 bg-slate-900/30 text-slate-500'
                }`}
              >
                <p className="font-semibold">4. ResNet18+GRU</p>
                <p className="text-[10px] mt-0.5">5 Behaviours</p>
              </div>
            </div>

            {/* Error Message & Retry Option if Failed */}
            {job.status === 'FAILED' && (
              <div className="space-y-3 rounded-xl border border-rose-500/30 bg-rose-500/10 p-4 text-xs text-rose-300">
                <div className="flex items-center justify-between">
                  <span className="font-semibold flex items-center gap-1.5">
                    <AlertCircle className="w-4 h-4 text-rose-400" />
                    Pipeline Error Details (Attempt {(job.retry_count || 0)}/{(job.max_retries || 3)})
                  </span>
                  <button
                    onClick={handleRetry}
                    disabled={retrying}
                    className="px-3 py-1 bg-rose-600 hover:bg-rose-500 text-white rounded-lg text-xs font-semibold flex items-center gap-1.5 transition"
                  >
                    <RefreshCw className={`w-3.5 h-3.5 ${retrying ? 'animate-spin' : ''}`} />
                    <span>{retrying ? 'Retrying...' : 'Retry Job'}</span>
                  </button>
                </div>
                <p className="font-mono text-[11px] bg-slate-950/60 p-2 rounded border border-rose-500/20">
                  {job.error_message || 'Execution error encountered during pipeline processing.'}
                </p>
              </div>
            )}

            {/* Action to View Results */}
            {job.status === 'COMPLETED' && (
              <div className="pt-4 border-t border-slate-800">
                <Link
                  to={`/results?job_id=${job.id}`}
                  className="btn-primary w-full py-3 text-sm flex items-center justify-center gap-2"
                >
                  <Sparkles className="h-4 w-4" />
                  <span>View Student Track Results & Analytics</span>
                  <ArrowRight className="h-4 w-4" />
                </Link>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Execution Logs Modal */}
      {showLogsModal && (
        <div className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-2xl w-full p-6 space-y-4 shadow-2xl">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <Terminal className="w-4 h-4 text-indigo-400" />
                <span>Job Processing Logs ({job?.id})</span>
              </h3>
              <button
                onClick={() => setShowLogsModal(false)}
                className="text-slate-400 hover:text-white text-xs px-2 py-1 bg-slate-800 rounded"
              >
                Close
              </button>
            </div>

            <div className="bg-slate-950 border border-slate-850 rounded-xl p-4 font-mono text-xs max-h-96 overflow-y-auto space-y-2">
              {logs.length === 0 ? (
                <div className="text-slate-600 italic">No logs recorded yet.</div>
              ) : (
                logs.map((l, idx) => (
                  <div key={idx} className="flex items-start gap-2">
                    <span className="text-slate-500 text-[10px] whitespace-nowrap">
                      {new Date(l.timestamp).toLocaleTimeString()}
                    </span>
                    <span
                      className={`px-1.5 py-0.2 rounded text-[10px] font-bold ${
                        l.level === 'ERROR' || l.level === 'FATAL'
                          ? 'bg-rose-500/20 text-rose-400'
                          : l.level === 'WARNING'
                          ? 'bg-amber-500/20 text-amber-400'
                          : 'bg-indigo-500/20 text-indigo-400'
                      }`}
                    >
                      {l.stage}
                    </span>
                    <span className="text-slate-300">{l.message}</span>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
