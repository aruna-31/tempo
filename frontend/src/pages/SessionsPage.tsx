import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../api/client';
import { ClassSession, Section, Subject } from '../types';
import { Calendar, Plus, Clock, UploadCloud, Activity, X, ChevronRight } from 'lucide-react';

export const SessionsPage: React.FC = () => {
  const [sessions, setSessions] = useState<ClassSession[]>([]);
  const [subjects, setSubjects] = useState<Subject[]>([]);
  const [sections, setSections] = useState<Section[]>([]);
  const [loading, setLoading] = useState(true);

  // Modal
  const [showModal, setShowModal] = useState(false);
  const [selectedSectionId, setSelectedSectionId] = useState('');
  const [sessionTitle, setSessionTitle] = useState('');
  const [sessionDate, setSessionDate] = useState(new Date().toISOString().split('T')[0]);
  const [startTime, setStartTime] = useState('09:30');
  const [endTime, setEndTime] = useState('10:30');
  const [modalError, setModalError] = useState<string | null>(null);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      const [sessList, subList] = await Promise.all([
        api.getSessions(),
        api.getSubjects(),
      ]);
      setSessions(sessList);
      setSubjects(subList);

      if (subList.length > 0) {
        const secPromises = subList.map((s) => api.getSections(s.id).catch(() => []));
        const secResults = await Promise.all(secPromises);
        const allSecs = secResults.flat();
        setSections(allSecs);
        if (allSecs.length > 0) {
          setSelectedSectionId(allSecs[0].id);
        }
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateSession = async (e: React.FormEvent) => {
    e.preventDefault();
    setModalError(null);
    if (!selectedSectionId) {
      setModalError('Please select a section. No sections exist yet - upload your timetable first (Schedules page).');
      return;
    }
    if (!sessionTitle) {
      setModalError('Please enter a session title.');
      return;
    }
    try {
      const created = await api.createSession({
        section_id: selectedSectionId,
        title: sessionTitle,
        session_date: sessionDate,
        start_time: startTime,
        end_time: endTime,
      });
      setSessions([created, ...sessions]);
      setShowModal(false);
      setSessionTitle('');
    } catch (err: any) {
      setModalError(err.message || 'Failed to create session');
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Class Sessions</h1>
          <p className="text-xs text-slate-400">Classroom lectures, lab sessions, and recorded video linkage</p>
        </div>
        <button onClick={() => setShowModal(true)} className="btn-primary">
          <Plus className="h-4 w-4" />
          <span>New Class Session</span>
        </button>
      </div>

      <div className="glass-panel p-6">
        <div className="overflow-x-auto">
          {loading ? (
            <div className="text-center py-8 text-xs text-slate-500">Loading class sessions...</div>
          ) : sessions.length === 0 ? (
            <div className="text-center py-12 space-y-3">
              <Calendar className="h-8 w-8 text-slate-600 mx-auto" />
              <p className="text-sm font-semibold text-slate-300">No class sessions found</p>
              <p className="text-xs text-slate-500 max-w-sm mx-auto">
                Create a class session to map video recordings and run individual-student behavior tracking.
              </p>
              <button onClick={() => setShowModal(true)} className="btn-secondary text-xs">
                Create First Session
              </button>
            </div>
          ) : (
            <table className="data-table">
              <thead>
                <tr>
                  <th>Session Title</th>
                  <th>Date & Time</th>
                  <th>Status</th>
                  <th className="text-right">Actions</th>
                </tr>
              </thead>
              <tbody>
                {sessions.map((sess) => (
                  <tr key={sess.id}>
                    <td className="font-semibold text-white">
                      <div>{sess.title}</div>
                      <span className="font-mono text-[10px] text-slate-500">ID: {sess.id.substring(0, 8)}...</span>
                    </td>
                    <td>
                      <div className="flex items-center gap-1.5 text-xs text-slate-300">
                        <Clock className="h-3.5 w-3.5 text-slate-500" />
                        <span>{sess.session_date} ({sess.start_time} - {sess.end_time})</span>
                      </div>
                    </td>
                    <td>
                      <span className="badge badge-instruction">ACTIVE</span>
                    </td>
                    <td className="text-right">
                      <div className="flex items-center justify-end gap-2">
                        <Link
                          to={`/upload?session_id=${sess.id}`}
                          className="btn-secondary text-xs py-1 px-2.5"
                          title="Upload video for this session"
                        >
                          <UploadCloud className="h-3.5 w-3.5" />
                          <span>Upload</span>
                        </Link>
                        <Link
                          to={`/results?session_id=${sess.id}`}
                          className="btn-primary text-xs py-1 px-3"
                        >
                          <span>Analytics</span>
                          <ChevronRight className="h-3.5 w-3.5" />
                        </Link>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>

      {/* Modal */}
      {showModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
          <div className="glass-panel p-6 max-w-md w-full border border-slate-700 shadow-2xl">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800 mb-4">
              <h3 className="text-base font-bold text-white">Schedule Class Session</h3>
              <button onClick={() => setShowModal(false)} className="text-slate-400 hover:text-white">
                <X className="h-5 w-5" />
              </button>
            </div>
            <form onSubmit={handleCreateSession} className="space-y-4">
              {modalError && (
                <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 p-3 text-xs text-rose-300">
                  {modalError}
                </div>
              )}
              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">Section Batch</label>
                {sections.length === 0 ? (
                  <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-3 text-xs text-amber-300">
                    No sections exist yet. Upload your timetable on the <span className="font-semibold">Schedules</span> page first - sections are created automatically from your confirmed timetable slots.
                  </div>
                ) : (
                  <select
                    required
                    value={selectedSectionId}
                    onChange={(e) => setSelectedSectionId(e.target.value)}
                    className="input-field"
                  >
                    <option value="">Select a section...</option>
                    {sections.map((sec) => (
                      <option key={sec.id} value={sec.id}>
                        {sec.name} ({sec.semester})
                      </option>
                    ))}
                  </select>
                )}
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">Session Title</label>
                <input
                  type="text"
                  required
                  value={sessionTitle}
                  onChange={(e) => setSessionTitle(e.target.value)}
                  placeholder="e.g. Lecture 04: Neural Networks Architecture"
                  className="input-field"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">Session Date</label>
                <input
                  type="date"
                  required
                  value={sessionDate}
                  onChange={(e) => setSessionDate(e.target.value)}
                  className="input-field"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-semibold text-slate-300 mb-1">Start Time</label>
                  <input
                    type="time"
                    required
                    value={startTime}
                    onChange={(e) => setStartTime(e.target.value)}
                    className="input-field"
                  />
                </div>
                <div>
                  <label className="block text-xs font-semibold text-slate-300 mb-1">End Time</label>
                  <input
                    type="time"
                    required
                    value={endTime}
                    onChange={(e) => setEndTime(e.target.value)}
                    className="input-field"
                  />
                </div>
              </div>

              <div className="flex justify-end gap-3 pt-2">
                <button type="button" onClick={() => setShowModal(false)} className="btn-secondary">
                  Cancel
                </button>
                <button type="submit" className="btn-primary">
                  Save Session
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};
