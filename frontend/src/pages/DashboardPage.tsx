import React, { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { api } from '../api/client';
import { useAuth } from '../context/AuthContext';
import { TodayClass, Subject, Video } from '../types';
import {
  BookOpen,
  Calendar,
  Video as VideoIcon,
  Activity,
  UploadCloud,
  ChevronRight,
  TrendingUp,
  Clock,
  Sparkles,
  MapPin,
  CheckCircle2,
  PlayCircle
} from 'lucide-react';

export const DashboardPage: React.FC = () => {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [todayClasses, setTodayClasses] = useState<TodayClass[]>([]);
  const [subjects, setSubjects] = useState<Subject[]>([]);
  const [videos, setVideos] = useState<Video[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadDashboardData = async () => {
      try {
        const [todayList, subList] = await Promise.all([
          api.getTodayClasses().catch(() => []),
          api.getSubjects().catch(() => []),
        ]);
        setTodayClasses(todayList);
        setSubjects(subList);

        // Fetch videos for today's classes
        if (todayList.length > 0) {
          const vPromises = todayList.map((c) => api.getSessionVideos(c.session_id).catch(() => []));
          const vResults = await Promise.all(vPromises);
          setVideos(vResults.flat());
        }
      } catch (err) {
        console.error('Error loading dashboard metrics:', err);
      } finally {
        setLoading(false);
      }
    };

    loadDashboardData();
  }, []);

  const todayStr = new Date().toLocaleDateString('en-US', {
    weekday: 'long',
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  });

  return (
    <div className="space-y-8">
      {/* Welcome Banner */}
      <div className="glass-panel p-6 bg-gradient-to-r from-indigo-950/60 via-slate-900/80 to-sky-950/40 border border-indigo-500/20">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1 text-xs font-semibold text-indigo-400 uppercase tracking-wider">
              <Sparkles className="h-4 w-4" />
              <span>Observable Behaviour Analytics Dashboard</span>
            </div>
            <h1 className="text-2xl font-bold text-white tracking-tight">
              Welcome back, {user?.full_name}
            </h1>
            <p className="text-sm text-slate-400 mt-1">
              Today is <span className="text-slate-200 font-medium">{todayStr}</span>. Select a scheduled class below to upload classroom video and run observable behaviour tracking.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <Link to="/schedules" className="btn-secondary text-xs">
              <Calendar className="h-4 w-4" />
              <span>Full Timetable</span>
            </Link>
          </div>
        </div>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        <div className="glass-panel p-5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-400 uppercase">Today's Scheduled Classes</span>
            <div className="p-2 rounded-xl bg-indigo-600/20 text-indigo-400">
              <Calendar className="h-5 w-5" />
            </div>
          </div>
          <p className="text-2xl font-bold text-white mt-3 font-mono">{todayClasses.length}</p>
          <p className="text-xs text-slate-400 mt-1">Slots on your timetable today</p>
        </div>

        <div className="glass-panel p-5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-400 uppercase">Videos Uploaded Today</span>
            <div className="p-2 rounded-xl bg-sky-600/20 text-sky-400">
              <VideoIcon className="h-5 w-5" />
            </div>
          </div>
          <p className="text-2xl font-bold text-white mt-3 font-mono">{videos.length}</p>
          <p className="text-xs text-slate-400 mt-1">Classroom recordings uploaded</p>
        </div>

        <div className="glass-panel p-5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-400 uppercase">Analyses Completed</span>
            <div className="p-2 rounded-xl bg-emerald-600/20 text-emerald-400">
              <CheckCircle2 className="h-5 w-5" />
            </div>
          </div>
          <p className="text-2xl font-bold text-white mt-3 font-mono">
            {todayClasses.filter((c) => c.has_analysis).length}
          </p>
          <p className="text-xs text-slate-400 mt-1">YOLO + ByteTrack + Temporal DL</p>
        </div>

        <div className="glass-panel p-5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-400 uppercase">My Subjects Mapped</span>
            <div className="p-2 rounded-xl bg-purple-600/20 text-purple-400">
              <BookOpen className="h-5 w-5" />
            </div>
          </div>
          <p className="text-2xl font-bold text-white mt-3 font-mono">{subjects.length}</p>
          <p className="text-xs text-slate-400 mt-1">Assigned academic courses</p>
        </div>
      </div>

      {/* TODAY'S CLASSES FROM TIMETABLE */}
      <div className="glass-panel p-6 border border-slate-800">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-4 border-b border-slate-800">
          <div>
            <h2 className="text-lg font-bold text-white tracking-tight flex items-center gap-2">
              <Calendar className="h-5 w-5 text-indigo-400" />
              <span>Today's Classes Scheduled on Timetable</span>
            </h2>
            <p className="text-xs text-slate-400">
              Only your classes scheduled for today ({todayStr}). Click a class to upload video and run behavior analytics.
            </p>
          </div>
        </div>

        <div className="mt-5">
          {loading ? (
            <div className="text-center py-12 text-xs text-slate-500">Loading today's classes from timetable...</div>
          ) : todayClasses.length === 0 ? (
            <div className="text-center py-14 space-y-3">
              <Calendar className="h-10 w-10 text-slate-600 mx-auto" />
              <p className="text-sm font-semibold text-slate-300">No classes scheduled for today in your timetable</p>
              <p className="text-xs text-slate-500 max-w-md mx-auto">
                Classes scheduled on your timetable for today will appear here automatically. You can also view or add weekly schedule slots in the Timetable section.
              </p>
              <Link to="/schedules" className="btn-secondary text-xs inline-flex items-center gap-2 mt-2">
                <span>View Full Timetable</span>
                <ChevronRight className="h-3.5 w-3.5" />
              </Link>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {todayClasses.map((cls) => (
                <div
                  key={cls.session_id}
                  onClick={() => navigate(`/upload?session_id=${cls.session_id}`)}
                  className="glass-panel p-5 border border-slate-800/90 hover:border-indigo-500/50 transition-all duration-200 cursor-pointer group hover:bg-slate-900/90 flex flex-col justify-between"
                >
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="px-2.5 py-1 rounded-md text-[11px] font-mono font-bold bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                        {cls.subject_code}
                      </span>
                      <div className="flex items-center gap-1.5 text-xs text-slate-400">
                        <Clock className="h-3.5 w-3.5 text-indigo-400" />
                        <span className="font-mono">{cls.start_time} - {cls.end_time}</span>
                      </div>
                    </div>

                    <div>
                      <h3 className="font-bold text-white text-base group-hover:text-indigo-300 transition-colors line-clamp-1">
                        {cls.subject_name}
                      </h3>
                      <p className="text-xs text-slate-400 mt-0.5">
                        {cls.section_name}
                      </p>
                    </div>

                    <div className="flex items-center gap-2 text-xs text-slate-400">
                      <MapPin className="h-3.5 w-3.5 text-slate-500" />
                      <span>{cls.room_number || 'Room TBD'}</span>
                    </div>
                  </div>

                  <div className="mt-5 pt-4 border-t border-slate-800/80 flex items-center justify-between">
                    <div>
                      {cls.has_analysis ? (
                        <span className="badge badge-instruction text-[11px] flex items-center gap-1">
                          <CheckCircle2 className="h-3 w-3" />
                          <span>Analyzed</span>
                        </span>
                      ) : cls.videos_count > 0 ? (
                        <span className="badge badge-peer text-[11px] flex items-center gap-1">
                          <VideoIcon className="h-3 w-3" />
                          <span>Video Attached</span>
                        </span>
                      ) : (
                        <span className="badge badge-away text-[11px]">
                          Pending Video
                        </span>
                      )}
                    </div>

                    <div className="flex items-center gap-2">
                      {cls.has_analysis ? (
                        <Link
                          to={`/results?session_id=${cls.session_id}`}
                          onClick={(e) => e.stopPropagation()}
                          className="btn-secondary text-xs py-1 px-2.5 flex items-center gap-1"
                        >
                          <Activity className="h-3.5 w-3.5" />
                          <span>Results</span>
                        </Link>
                      ) : null}
                      <button
                        type="button"
                        className="btn-primary text-xs py-1 px-3 flex items-center gap-1 group-hover:bg-indigo-500 transition-colors"
                      >
                        <UploadCloud className="h-3.5 w-3.5" />
                        <span>Upload Video</span>
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DashboardPage;
