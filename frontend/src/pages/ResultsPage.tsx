import React, { useEffect, useState } from 'react';
import { useLocation, Link } from 'react-router-dom';
import { api } from '../api/client';
import {
  BehaviourResult,
  ClassSession,
  FacultyInsight,
  ModelInfo,
  TemporalAnalyticsProfile,
  SessionAnalyticsSummary,
  StudentTrackMetrics,
  StudentTrackResult,
  Video,
} from '../types';
import { VideoPlayerWithOverlay } from '../components/VideoPlayerWithOverlay';
import { BehaviorCharts } from '../components/BehaviorCharts';
import { TrackTimeline } from '../components/TrackTimeline';
import { FacultyInsightSection } from '../components/FacultyInsightSection';
import {
  Activity,
  Download,
  FileText,
  User,
  CheckCircle2,
  AlertCircle,
  HelpCircle,
  ChevronRight,
  TrendingUp,
  Clock,
  Sparkles,
  Layers,
  X,
} from 'lucide-react';

const DOMINANT_BADGE: Record<string, { label: string; classStyle: string }> = {
  Looking_Toward_Instruction: { label: 'Instruction', classStyle: 'badge-instruction' },
  Reading: { label: 'Reading', classStyle: 'badge-reading' },
  Writing: { label: 'Writing', classStyle: 'badge-writing' },
  Peer_Interaction: { label: 'Peer Interaction', classStyle: 'badge-peer' },
  Looking_Away: { label: 'Looking Away', classStyle: 'badge-away' },
};

export const ResultsPage: React.FC = () => {
  const location = useLocation();
  const searchParams = new URLSearchParams(location.search);
  const paramJobId = searchParams.get('job_id');
  const paramSessionId = searchParams.get('session_id');

  const [activeJobId, setActiveJobId] = useState<string | null>(paramJobId);
  const [summary, setSummary] = useState<SessionAnalyticsSummary | null>(null);
  const [tracks, setTracks] = useState<StudentTrackResult[]>([]);
  const [behaviours, setBehaviours] = useState<BehaviourResult[]>([]);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [temporal, setTemporal] = useState<TemporalAnalyticsProfile | null>(null);
  const [facultyInsights, setFacultyInsights] = useState<FacultyInsight[]>([]);
  const [selectedTemporalTrack, setSelectedTemporalTrack] = useState<number | null>(null);
  const [videoUrl, setVideoUrl] = useState<string>('');

  const [selectedTrackId, setSelectedTrackId] = useState<number | null>(null);
  const [inspectModalTrack, setInspectModalTrack] = useState<StudentTrackMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadResultsData();
  }, [paramJobId, paramSessionId]);

  const loadResultsData = async () => {
    setLoading(true);
    setError(null);

    try {
      let resolvedJobId = paramJobId;

      // If sessionId provided but no jobId, fetch video & jobs for that session
      if (!resolvedJobId && paramSessionId) {
        const vList = await api.getSessionVideos(paramSessionId);
        if (vList.length > 0) {
          const jList = await api.getVideoJobs(vList[0].id);
          if (jList.length > 0) {
            resolvedJobId = jList[0].id;
          }
        }
      }

      // If still no jobId, look for any completed job from sessions
      if (!resolvedJobId) {
        const sessList = await api.getSessions();
        for (const s of sessList) {
          const vList = await api.getSessionVideos(s.id).catch(() => []);
          for (const v of vList) {
            const jList = await api.getVideoJobs(v.id).catch(() => []);
            const completed = jList.find((j) => j.status === 'COMPLETED');
            if (completed) {
              resolvedJobId = completed.id;
              break;
            }
          }
          if (resolvedJobId) break;
        }
      }

      if (!resolvedJobId) {
        setError('No completed analysis jobs found. Please upload a video to generate results.');
        setLoading(false);
        return;
      }

      setActiveJobId(resolvedJobId);

      // Fetch Summary, Tracks, Behaviours, ModelInfo, Temporal, and Faculty Insights concurrently
      const [sumData, trackList, behavList, activeModelInfo, temporalProfile, insightsList] = await Promise.all([
        api.getJobSummary(resolvedJobId, 60),
        api.getJobTracks(resolvedJobId),
        api.getJobBehaviours(resolvedJobId, undefined, 1000),
        api.getModelInfo(),
        api.getTemporalProfile(resolvedJobId),
        api.getFacultyInsights(resolvedJobId).catch(() => []),
      ]);

      setSummary(sumData);
      setTracks(trackList);
      setBehaviours(behavList);
      setModelInfo(activeModelInfo);
      setTemporal(temporalProfile);
      setFacultyInsights(insightsList || []);

      // Construct video stream URL
      setVideoUrl(`/api/v1/videos/${sumData.video_id}/output-stream`);
    } catch (err: any) {
      console.error(err);
      setError(err.message || 'Failed to load analysis results.');
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadJSON = () => {
    if (!summary) return;
    const reportData = {
      export_timestamp: new Date().toISOString(),
      disclaimer:
        'Ethical Notice: Results represent temporary observable engagement activity analysis and are not a measure of academic intelligence or student evaluation.',
      summary,
      tracks,
      behaviours: behaviours.slice(0, 500),
    };

    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(reportData, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute('href', dataStr);
    downloadAnchor.setAttribute('download', `TEMPO_Analysis_Report_${activeJobId?.substring(0, 8)}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  const handlePrintReport = () => {
    window.print();
  };

  if (loading) {
    return (
      <div className="flex min-h-[60vh] items-center justify-center gap-3 text-slate-400">
        <Sparkles className="h-5 w-5 animate-spin text-indigo-400" />
        <span className="text-sm">Loading multi-student track analytics & video data...</span>
      </div>
    );
  }

  if (error || !summary) {
    return (
      <div className="glass-panel p-12 text-center max-w-xl mx-auto space-y-4">
        <AlertCircle className="h-10 w-10 text-amber-400 mx-auto" />
        <h2 className="text-lg font-bold text-white">No Analysis Results Available</h2>
        <p className="text-xs text-slate-400">{error}</p>
        <Link to="/upload" className="btn-primary inline-flex">
          Upload Video to Analyze
        </Link>
      </div>
    );
  }

  return (
    <div className="space-y-8 pb-12">
      {/* Header & Export Actions */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-6">
        <div>
          <div className="flex items-center gap-2 text-xs font-semibold text-indigo-400 uppercase tracking-wider mb-1">
            <Sparkles className="h-3.5 w-3.5" />
            <span>Temporal Behavioral Inference Results</span>
          </div>
          <h1 className="text-2xl font-bold text-white tracking-tight">
            Individual Student Track Analysis
          </h1>
          <p className="text-xs text-slate-400 mt-0.5">
            Session-scoped Track IDs • Multi-object tracking (ByteTrack) • 5 Observable Behaviors
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button onClick={handleDownloadJSON} className="btn-secondary text-xs">
            <Download className="h-4 w-4" />
            <span>Export JSON</span>
          </button>
          <button onClick={handlePrintReport} className="btn-primary text-xs">
            <FileText className="h-4 w-4" />
            <span>Print Report</span>
          </button>
        </div>
      </div>

      {/* Metrics Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        <div className="glass-panel p-5">
          <span className="text-xs font-semibold text-slate-400 uppercase">Overall Engagement</span>
          <p className="text-3xl font-bold text-emerald-400 mt-2 font-mono">
            {summary.overall_engagement_score}%
          </p>
          <p className="text-[11px] text-slate-400 mt-1">Instruction + Reading + Writing</p>
        </div>

        <div className="glass-panel p-5">
          <span className="text-xs font-semibold text-slate-400 uppercase">Student Tracks</span>
          <p className="text-3xl font-bold text-sky-400 mt-2 font-mono">
            {summary.total_tracks_identified || summary.track_metrics.length}
          </p>
          <p className="text-[11px] text-slate-400 mt-1">Temporary Session Track IDs</p>
        </div>

        <div className="glass-panel p-5">
          <span className="text-xs font-semibold text-slate-400 uppercase">Total Detections</span>
          <p className="text-3xl font-bold text-indigo-400 mt-2 font-mono">
            {summary.total_frames_analyzed || behaviours.length}
          </p>
          <p className="text-[11px] text-slate-400 mt-1">Temporal Sequence Predictions</p>
        </div>

        <div className="glass-panel p-5">
          <span className="text-xs font-semibold text-slate-400 uppercase">Model Backbone</span>
          <p className="text-xl font-bold text-white mt-2 font-mono">
            {modelInfo
              ? `${modelInfo.spatial_backbone.toUpperCase()} + ${modelInfo.temporal_model_type}`
              : 'Loading model...'}
          </p>
          <p className="text-[11px] text-slate-400 mt-1">
            {modelInfo
              ? `${modelInfo.model_version} | ${modelInfo.sequence_length} frames @ ${modelInfo.sampling_fps} FPS`
              : 'Active production model'}
          </p>
        </div>
      </div>

      {temporal && (
        <div className="space-y-6">
          <section className="grid grid-cols-2 md:grid-cols-5 gap-3">
            {[
              ['Detected', temporal.coverage?.detected_student_count ?? 0],
              ['Tracked', temporal.coverage?.tracked_student_count ?? 0],
              ['Temporal-ready', temporal.coverage?.temporal_ready_track_count ?? 0],
              ['Detection coverage', `${((temporal.coverage?.detection_coverage ?? 0) * 100).toFixed(1)}%`],
              ['Temporal coverage', `${((temporal.coverage?.temporal_coverage ?? 0) * 100).toFixed(1)}%`],
            ].map(([label, value]) => (
              <div key={label} className="glass-panel p-4">
                <span className="text-[10px] uppercase text-slate-500">{label}</span>
                <p className="text-xl font-mono text-white mt-1">{value}</p>
              </div>
            ))}
          </section>

          <section className="glass-panel p-5 space-y-4">
            <div>
              <h2 className="text-base font-semibold text-white">Classroom Activity Timeline</h2>
              <p className="text-xs text-slate-500">Aggregated observable activity across contributing anonymous tracks.</p>
            </div>
            <div className="space-y-2">
              {temporal.classroom_states.map((state) => (
                <div key={`${state.start_time}-${state.state}`} className="flex items-center gap-3 text-xs">
                  <span className="w-16 font-mono text-slate-500">{state.start_time.toFixed(0)}s</span>
                  <div className="h-3 flex-1 rounded bg-slate-800 overflow-hidden flex">
                    {Object.entries(state.behaviour_distribution).map(([label, value]) => <div key={label} style={{ width: `${value * 100}%` }} className="bg-sky-400 first:bg-emerald-400" />)}
                  </div>
                  <span className="w-44 text-slate-300">{state.state}</span>
                </div>
              ))}
            </div>
          </section>

          <section className="grid md:grid-cols-2 gap-5">
            <div className="glass-panel p-5 space-y-3">
              <h2 className="text-base font-semibold text-white">Entropy Over Time</h2>
              <p className="text-xs text-slate-500">Higher values indicate greater diversity of observed activities, not better or worse activity.</p>
              {temporal.entropy.map((item) => <div key={item.timestamp} className="flex items-center gap-3 text-xs"><span className="w-12 text-slate-500">{item.timestamp.toFixed(0)}s</span><div className="h-2 flex-1 bg-slate-800 rounded"><div className="h-full bg-amber-400 rounded" style={{ width: `${Math.min(100, item.entropy * 45)}%` }} /></div><span className="font-mono text-slate-300">{item.entropy.toFixed(3)}</span></div>)}
            </div>
            <div className="glass-panel p-5 space-y-3">
              <h2 className="text-base font-semibold text-white">Change Points</h2>
              {temporal.change_points.length === 0 ? <p className="text-xs text-slate-500">No change points exceeded the configured divergence threshold.</p> : temporal.change_points.map((point) => <div key={`${point.timestamp}-${point.change_score}`} className="text-xs border-l-2 border-amber-400 pl-3"><p className="text-slate-300">{point.timestamp.toFixed(1)}s Activity transition detected</p><p className="text-slate-500">{point.previous_state} -&gt; {point.new_state} ({point.change_score.toFixed(3)})</p></div>)}
            </div>
          </section>

          <section className="glass-panel p-5 space-y-3">
            <h2 className="text-base font-semibold text-white">Anonymous Student Trajectories</h2>
            <div className="flex flex-wrap gap-2">{temporal.students.map((student) => <button key={student.track_id} onClick={() => setSelectedTemporalTrack(student.track_id)} className={`btn-secondary text-xs ${selectedTemporalTrack === student.track_id ? 'border-indigo-400 text-indigo-300' : ''}`}>Student {student.track_id.toString().padStart(2, '0')}</button>)}</div>
            {selectedTemporalTrack !== null && (() => { const student = temporal.students.find((item) => item.track_id === selectedTemporalTrack); return student ? <div className="border-t border-slate-800 pt-3 text-xs space-y-2"><p className="text-slate-300">Student {student.track_id.toString().padStart(2, '0')} - {student.segment_count} behaviour segments - temporal coverage {(student.temporal_coverage * 100).toFixed(1)}%</p><div className="flex flex-wrap gap-2">{student.timeline.map((segment, index) => <span key={`${segment.start_time}-${index}`} className="rounded bg-slate-800 px-2 py-1 text-slate-300">{segment.start_time.toFixed(1)}s {segment.behaviour} ({(segment.confidence * 100).toFixed(1)}%)</span>)}</div></div> : null; })()}
          </section>
        </div>
      )}

      {/* Post-Class Faculty Insights & Suggestions for Future Classes */}
      <FacultyInsightSection insights={facultyInsights} />

      {/* Synchronized Video Player with Canvas Bounding Boxes */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-base font-semibold text-white">Analyzed Classroom Video Feed</h3>
          <span className="text-xs text-slate-400">Synchronized bounding boxes & confidence meters</span>
        </div>

        <VideoPlayerWithOverlay
          videoUrl={videoUrl}
          tracks={tracks}
          behaviours={behaviours}
          selectedTrackId={selectedTrackId}
        />
      </div>

      {/* Recharts Activity Analytics Visualizations */}
      <div className="space-y-3">
        <h3 className="text-base font-semibold text-white">Observable Behavior Distributions</h3>
        <BehaviorCharts
          trackMetrics={summary.track_metrics}
          timelineBuckets={summary.timeline_buckets}
        />
      </div>

      {/* Individual Student Track Breakdown Table */}
      <div className="glass-panel p-6 space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-base font-semibold text-white">Individual Student Track Profiles</h3>
            <p className="text-xs text-slate-400">
              Temporary Track IDs assigned during video analysis (No facial recognition / permanent identity)
            </p>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="data-table">
            <thead>
              <tr>
                <th>Track ID</th>
                <th>Dominant Activity</th>
                <th>Activity Distribution Breakdown</th>
                <th>Instruction</th>
                <th>Reading / Writing</th>
                <th>Looking Away</th>
                <th className="text-right">Action</th>
              </tr>
            </thead>
            <tbody>
              {summary.track_metrics.map((tm) => {
                const domBadge = DOMINANT_BADGE[tm.dominant_behaviour] || {
                  label: tm.dominant_behaviour,
                  classStyle: 'badge-instruction',
                };

                return (
                  <tr
                    key={tm.track_id}
                    className={selectedTrackId === tm.track_id ? 'bg-indigo-950/30' : ''}
                  >
                    <td>
                      <div className="flex items-center gap-2">
                        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-indigo-600/20 text-indigo-400 font-mono text-xs font-bold">
                          {tm.track_id.toString().padStart(2, '0')}
                        </div>
                        <div>
                          <p className="font-semibold text-white">{tm.display_label}</p>
                          <span className="font-mono text-[10px] text-slate-500">
                            Track #{tm.track_id}
                          </span>
                        </div>
                      </div>
                    </td>

                    <td>
                      <span className={`badge ${domBadge.classStyle}`}>
                        {domBadge.label}
                      </span>
                    </td>

                    {/* Compact Stacked Distribution Bar */}
                    <td className="w-48">
                      <div className="h-2 w-full rounded-full bg-slate-800 flex overflow-hidden">
                        <div style={{ width: `${tm.looking_toward_instruction_pct}%` }} className="bg-sky-400" title={`Instruction: ${tm.looking_toward_instruction_pct}%`} />
                        <div style={{ width: `${tm.reading_pct}%` }} className="bg-emerald-400" title={`Reading: ${tm.reading_pct}%`} />
                        <div style={{ width: `${tm.writing_pct}%` }} className="bg-indigo-400" title={`Writing: ${tm.writing_pct}%`} />
                        <div style={{ width: `${tm.peer_interaction_pct}%` }} className="bg-amber-400" title={`Peer: ${tm.peer_interaction_pct}%`} />
                        <div style={{ width: `${tm.looking_away_pct}%` }} className="bg-rose-400" title={`Away: ${tm.looking_away_pct}%`} />
                      </div>
                    </td>

                    <td className="font-mono text-xs text-slate-300">
                      {tm.looking_toward_instruction_pct}%
                    </td>

                    <td className="font-mono text-xs text-slate-300">
                      {(tm.reading_pct + tm.writing_pct).toFixed(1)}%
                    </td>

                    <td className="font-mono text-xs text-slate-300">
                      {tm.looking_away_pct}%
                    </td>

                    <td className="text-right">
                      <button
                        onClick={() => {
                          setSelectedTrackId(tm.track_id);
                          setInspectModalTrack(tm);
                        }}
                        className="btn-secondary text-xs py-1 px-3"
                      >
                        <span>Inspect Stream</span>
                        <ChevronRight className="h-3.5 w-3.5" />
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Modal / Drawer for Track Stream Inspection */}
      {inspectModalTrack && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
          <div className="glass-panel p-6 max-w-2xl w-full border border-slate-700 shadow-2xl space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <div className="flex items-center gap-2">
                <span className="text-base font-bold text-white">
                  Detailed Stream: {inspectModalTrack.display_label}
                </span>
                <span className="badge badge-instruction">Session Track #{inspectModalTrack.track_id}</span>
              </div>
              <button
                onClick={() => setInspectModalTrack(null)}
                className="text-slate-400 hover:text-white"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            <TrackTimeline
              trackId={inspectModalTrack.track_id}
              behaviours={behaviours}
              onSelectTimestamp={(t) => {
                // Scroll up and set video time
                window.scrollTo({ top: 150, behavior: 'smooth' });
              }}
            />

            <div className="flex justify-end pt-2 border-t border-slate-800">
              <button onClick={() => setInspectModalTrack(null)} className="btn-secondary text-xs">
                Close Stream
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Ethical Guidance & Disclaimer */}
      <div className="rounded-2xl border border-indigo-900/40 bg-indigo-950/20 p-5 text-xs text-indigo-200/90 leading-relaxed flex items-start gap-3">
        <HelpCircle className="h-5 w-5 text-indigo-400 flex-shrink-0 mt-0.5" />
        <div>
          <p className="font-semibold text-indigo-300 mb-1">
            Ethical Framing & Responsible AI Notice
          </p>
          <p className="text-slate-400">
            This system evaluates <strong>observable visual engagement patterns</strong> (orientation toward instruction, reading, writing, peer discussions) on session-scoped temporary Track IDs. It does not perform biometric facial recognition, calculate academic grades, or estimate student cognitive intelligence. Track IDs are cleared after each session.
          </p>
        </div>
      </div>
    </div>
  );
};
