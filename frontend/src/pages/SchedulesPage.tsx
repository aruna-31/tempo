import React, { useEffect, useState } from 'react';
import { api } from '../api/client';
import { FacultySchedule, Section, Subject, TimetableExtractedSlot } from '../types';
import {
  Calendar,
  Clock,
  MapPin,
  Play,
  Plus,
  RefreshCw,
  Sparkles,
  UploadCloud,
  CheckCircle2,
  AlertCircle,
  Trash2
} from 'lucide-react';

const DAYS_OF_WEEK = [
  'MONDAY',
  'TUESDAY',
  'WEDNESDAY',
  'THURSDAY',
  'FRIDAY',
  'SATURDAY'
];

export const SchedulesPage: React.FC = () => {
  const [schedules, setSchedules] = useState<FacultySchedule[]>([]);
  const [subjects, setSubjects] = useState<Subject[]>([]);
  const [sections, setSections] = useState<Record<string, Section[]>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Timetable image/PDF extraction state (prototype: upload -> OCR -> review/edit -> confirm)
  const [extracting, setExtracting] = useState(false);
  const [extractMessage, setExtractMessage] = useState<string | null>(null);
  const [extractError, setExtractError] = useState<string | null>(null);
  const [extractedSlots, setExtractedSlots] = useState<TimetableExtractedSlot[]>([]);
  const [confirming, setConfirming] = useState(false);
  const [confirmResult, setConfirmResult] = useState<string | null>(null);

  // Form State
  const [showAddModal, setShowAddModal] = useState(false);
  const [showGenerateModal, setShowGenerateModal] = useState(false);
  const [selectedSubjectId, setSelectedSubjectId] = useState('');
  const [selectedSectionId, setSelectedSectionId] = useState('');
  const [dayOfWeek, setDayOfWeek] = useState('MONDAY');
  const [startTime, setStartTime] = useState('10:00');
  const [endTime, setEndTime] = useState('10:50');

  // Generator State
  const [startDate, setStartDate] = useState(new Date().toISOString().split('T')[0]);
  const [endDate, setEndDate] = useState(
    new Date(Date.now() + 14 * 24 * 60 * 60 * 1000).toISOString().split('T')[0]
  );
  const [generating, setGenerating] = useState(false);
  const [saving, setSaving] = useState(false);

  const fetchData = async () => {
    try {
      setLoading(true);
      const [schedData, subs] = await Promise.all([
        api.getSchedules(),
        api.getSubjects(),
      ]);
      setSchedules(schedData);
      setSubjects(subs);

      // Load sections for each subject
      const secMap: Record<string, Section[]> = {};
      await Promise.all(
        subs.map(async (s) => {
          try {
            const secs = await api.getSections(s.id);
            secMap[s.id] = secs;
          } catch {
            secMap[s.id] = [];
          }
        })
      );
      setSections(secMap);
      setError(null);
    } catch (err: any) {
      setError(err.message || 'Failed to load faculty schedules');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const handleCreateSchedule = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedSubjectId || !selectedSectionId) return;

    try {
      setSaving(true);
      await api.createSchedule({
        subject_id: selectedSubjectId,
        section_id: selectedSectionId,
        day_of_week: dayOfWeek,
        start_time: startTime + ':00',
        end_time: endTime + ':00',
      });
      setShowAddModal(false);
      fetchData();
    } catch (err: any) {
      alert(err.message || 'Failed to create schedule slot');
    } finally {
      setSaving(false);
    }
  };

  const handleGenerateSessions = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      setGenerating(true);
      const res = await api.generateSessionsFromSchedule({
        start_date: startDate,
        end_date: endDate,
      });
      alert(res.message);
      setShowGenerateModal(false);
    } catch (err: any) {
      alert(err.message || 'Failed to instantiate calendar sessions');
    } finally {
      setGenerating(false);
    }
  };

  // ---------------- Timetable Image/PDF Upload & Confirmation ----------------
  const handleTimetableFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setExtracting(true);
    setExtractError(null);
    setExtractMessage(null);
    setConfirmResult(null);
    try {
      const resp = await api.extractTimetable(file);
      setExtractedSlots(resp.slots.map((s: TimetableExtractedSlot, i: number) => ({ ...s, id: s.id || `slot_${i}` })));
      setExtractMessage(resp.message);
    } catch (err: any) {
      setExtractError(err.message || 'Timetable extraction failed');
    } finally {
      setExtracting(false);
      e.target.value = '';
    }
  };

  const updateSlot = (idx: number, field: keyof TimetableExtractedSlot, value: string) => {
    setExtractedSlots((prev) => prev.map((s, i) => (i === idx ? { ...s, [field]: value } : s)));
  };

  const removeSlot = (idx: number) => {
    setExtractedSlots((prev) => prev.filter((_, i) => i !== idx));
  };

  const handleConfirmTimetable = async () => {
    setConfirming(true);
    setConfirmResult(null);
    setExtractError(null);
    try {
      const resp = await api.confirmTimetable({
        slots: extractedSlots.map((s) => ({
          day_of_week: s.day_of_week,
          start_time: s.start_time,
          end_time: s.end_time,
          subject_code: s.subject_code,
          subject_name: s.subject_name,
          section_name: s.section_name,
          room_number: s.room_number || undefined,
        })),
        clear_existing: true,
      });
      setConfirmResult(resp.message);
      setExtractedSlots([]);
      setExtractMessage(null);
      fetchData();
    } catch (err: any) {
      setExtractError(err.message || 'Failed to save timetable');
    } finally {
      setConfirming(false);
    }
  };

  const subjectMap = new Map(subjects.map((s) => [s.id, s]));

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight flex items-center gap-2">
            <Calendar className="w-6 h-6 text-indigo-400" />
            Faculty Timetable
          </h1>
          <p className="text-sm text-slate-400">Your weekly class timetable slots.</p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => setShowGenerateModal(true)}
            className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg text-sm font-medium flex items-center gap-2 shadow-lg shadow-emerald-500/20 transition"
          >
            <Sparkles className="w-4 h-4" />
            Auto-Instantiate Sessions
          </button>
          <button
            onClick={() => setShowAddModal(true)}
            className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-sm font-medium flex items-center gap-2 shadow-lg shadow-indigo-500/20 transition"
          >
            <Plus className="w-4 h-4" />
            Add Timetable Slot
          </button>
        </div>
      </div>

      {/* Timetable Image/PDF Upload -> OCR Extraction -> Review/Edit -> Confirm */}
      <div className="glass-panel p-6 border border-slate-800 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h2 className="text-base font-semibold text-white flex items-center gap-2">
              <UploadCloud className="h-5 w-5 text-indigo-400" />
              <span>Upload Timetable Image / PDF</span>
            </h2>
            <p className="text-xs text-slate-400 mt-1">Upload your timetable image or PDF. Review the extracted entries, then save.</p>
          </div>
          <label className="btn-primary text-xs cursor-pointer inline-flex items-center gap-2 whitespace-nowrap">
            <UploadCloud className="h-4 w-4" />
            <span>{extracting ? 'Extracting...' : 'Choose Timetable File'}</span>
            <input
              type="file"
              accept=".png,.jpg,.jpeg,.webp,.pdf"
              className="hidden"
              onChange={handleTimetableFile}
              disabled={extracting}
            />
          </label>
        </div>

        {extractError && (
          <div className="flex items-center gap-2 rounded-xl border border-rose-500/30 bg-rose-500/10 p-3 text-xs text-rose-300">
            <AlertCircle className="h-4 w-4 flex-shrink-0" />
            <span>{extractError}</span>
          </div>
        )}
        {extractMessage && !extractError && (
          <div className="flex items-center gap-2 rounded-xl border border-indigo-500/30 bg-indigo-500/10 p-3 text-xs text-indigo-200">
            <AlertCircle className="h-4 w-4 flex-shrink-0" />
            <span>{extractMessage}</span>
          </div>
        )}
        {confirmResult && (
          <div className="flex items-center gap-2 rounded-xl border border-emerald-500/30 bg-emerald-500/10 p-3 text-xs text-emerald-300">
            <CheckCircle2 className="h-4 w-4 flex-shrink-0" />
            <span>{confirmResult}</span>
          </div>
        )}

        {extractedSlots.length > 0 && (
          <div className="space-y-3">
            <div className="overflow-x-auto">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Day</th>
                    <th>Start</th>
                    <th>End</th>
                    <th>Subject</th>
                    <th>Section / Batch</th>
                    <th>Room</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {extractedSlots.map((slot, idx) => (
                    <tr key={slot.id || idx}>
                      <td>
                        <select
                          value={slot.day_of_week}
                          onChange={(e) => updateSlot(idx, 'day_of_week', e.target.value)}
                          className="bg-slate-950 border border-slate-800 rounded px-2 py-1 text-xs text-white"
                        >
                          <option value="">-- select --</option>
                          {['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY'].map((d) => (
                            <option key={d} value={d}>{d}</option>
                          ))}
                        </select>
                      </td>
                      <td>
                        <input
                          type="time"
                          value={slot.start_time}
                          onChange={(e) => updateSlot(idx, 'start_time', e.target.value)}
                          className="bg-slate-950 border border-slate-800 rounded px-2 py-1 text-xs text-white w-28"
                        />
                      </td>
                      <td>
                        <input
                          type="time"
                          value={slot.end_time}
                          onChange={(e) => updateSlot(idx, 'end_time', e.target.value)}
                          className="bg-slate-950 border border-slate-800 rounded px-2 py-1 text-xs text-white w-28"
                        />
                      </td>
                      <td>
                        <input
                          value={slot.subject_name}
                          onChange={(e) => updateSlot(idx, 'subject_name', e.target.value)}
                          placeholder="e.g. 3-CSE-DL"
                          className="bg-slate-950 border border-slate-800 rounded px-2 py-1 text-xs text-white w-32"
                        />
                      </td>
                      <td>
                        <input
                          value={slot.section_name}
                          onChange={(e) => updateSlot(idx, 'section_name', e.target.value)}
                          placeholder="e.g. PG2 / 3-CSE-DL"
                          className="bg-slate-950 border border-slate-800 rounded px-2 py-1 text-xs text-white w-32"
                        />
                      </td>
                      <td>
                        <input
                          value={slot.room_number || ''}
                          onChange={(e) => updateSlot(idx, 'room_number', e.target.value)}
                          className="bg-slate-950 border border-slate-800 rounded px-2 py-1 text-xs text-white w-24"
                        />
                      </td>
                      <td>
                        <button
                          onClick={() => removeSlot(idx)}
                          className="text-slate-500 hover:text-rose-400"
                          title="Remove entry"
                        >
                          <Trash2 className="h-4 w-4" />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <button
              onClick={handleConfirmTimetable}
              disabled={confirming}
              className="btn-primary text-xs inline-flex items-center gap-2"
            >
              <CheckCircle2 className="h-4 w-4" />
              <span>{confirming ? 'Saving...' : 'Confirm & Save Timetable'}</span>
            </button>
          </div>
        )}
      </div>

      {/* Timetable Weekly Columns */}
      {loading ? (
        <div className="flex items-center justify-center p-12 text-slate-400">
          <RefreshCw className="w-6 h-6 animate-spin mr-2" /> Loading weekly timetable...
        </div>
      ) : error ? (
        <div className="p-4 bg-rose-500/10 border border-rose-500/30 text-rose-300 rounded-xl">
          {error}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
          {DAYS_OF_WEEK.map((day) => {
            const daySlots = schedules.filter((s) => s.day_of_week.toUpperCase() === day);
            return (
              <div
                key={day}
                className="bg-slate-900/60 backdrop-blur-md rounded-2xl border border-slate-800 p-4 space-y-3"
              >
                <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                  <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider">{day}</h3>
                  <span className="text-[10px] bg-slate-800 text-slate-400 px-2 py-0.5 rounded-full font-medium">
                    {daySlots.length}
                  </span>
                </div>

                {daySlots.length === 0 ? (
                  <div className="text-xs text-slate-600 text-center py-6">No scheduled slots</div>
                ) : (
                  <div className="space-y-2">
                    {daySlots.map((slot) => {
                      const sub = subjectMap.get(slot.subject_id);
                      return (
                        <div
                          key={slot.id}
                          className="p-3 bg-slate-950/60 rounded-xl border border-indigo-500/20 hover:border-indigo-500/40 transition space-y-2"
                        >
                          <div className="font-semibold text-white text-xs line-clamp-1">
                            {sub?.name || 'Class Session'}
                          </div>
                          <div className="flex items-center gap-1.5 text-[11px] text-indigo-300">
                            <Clock className="w-3 h-3 text-indigo-400" />
                            <span>
                              {slot.start_time.slice(0, 5)} - {slot.end_time.slice(0, 5)}
                            </span>
                          </div>
                          {slot.section_id && (
                            <div className="flex items-center gap-1.5 text-[11px] text-slate-400">
                              <MapPin className="w-3 h-3 text-slate-500" />
                              <span>{sub?.code || 'Course'}</span>
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Add Slot Modal */}
      {showAddModal && (
        <div className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-md w-full p-6 space-y-4 shadow-2xl">
            <h3 className="text-lg font-bold text-white">Add Recurring Timetable Slot</h3>
            <form onSubmit={handleCreateSchedule} className="space-y-4">
              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  Subject <span className="text-rose-400">*</span>
                </label>
                <select
                  required
                  value={selectedSubjectId}
                  onChange={(e) => {
                    setSelectedSubjectId(e.target.value);
                    setSelectedSectionId('');
                  }}
                  className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-lg text-white text-sm focus:outline-none focus:border-indigo-500"
                >
                  <option value="">Select Subject</option>
                  {subjects.map((sub) => (
                    <option key={sub.id} value={sub.id}>
                      {sub.code} - {sub.name}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  Section <span className="text-rose-400">*</span>
                </label>
                <select
                  required
                  disabled={!selectedSubjectId}
                  value={selectedSectionId}
                  onChange={(e) => setSelectedSectionId(e.target.value)}
                  className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-lg text-white text-sm focus:outline-none focus:border-indigo-500 disabled:opacity-50"
                >
                  <option value="">Select Section</option>
                  {(sections[selectedSubjectId] || []).map((sec) => (
                    <option key={sec.id} value={sec.id}>
                      {sec.name} ({sec.academic_year})
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">Day of Week</label>
                <select
                  value={dayOfWeek}
                  onChange={(e) => setDayOfWeek(e.target.value)}
                  className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-lg text-white text-sm focus:outline-none focus:border-indigo-500"
                >
                  {DAYS_OF_WEEK.map((d) => (
                    <option key={d} value={d}>
                      {d}
                    </option>
                  ))}
                </select>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1">Start Time</label>
                  <input
                    type="time"
                    required
                    value={startTime}
                    onChange={(e) => setStartTime(e.target.value)}
                    className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-lg text-white text-sm focus:outline-none focus:border-indigo-500"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1">End Time</label>
                  <input
                    type="time"
                    required
                    value={endTime}
                    onChange={(e) => setEndTime(e.target.value)}
                    className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-lg text-white text-sm focus:outline-none focus:border-indigo-500"
                  />
                </div>
              </div>

              <div className="flex items-center justify-end gap-3 pt-4 border-t border-slate-800">
                <button
                  type="button"
                  onClick={() => setShowAddModal(false)}
                  className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-sm"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={saving}
                  className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-sm font-medium"
                >
                  {saving ? 'Saving...' : 'Add Slot'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Auto-Generate Sessions Modal */}
      {showGenerateModal && (
        <div className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-md w-full p-6 space-y-4 shadow-2xl">
            <h3 className="text-lg font-bold text-white flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-emerald-400" />
              Auto-Instantiate Calendar Sessions
            </h3>
            <p className="text-xs text-slate-400">
              Populates your class session calendar by projecting your weekly recurring timetable across the selected date range.
            </p>
            <form onSubmit={handleGenerateSessions} className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1">Start Date</label>
                  <input
                    type="date"
                    required
                    value={startDate}
                    onChange={(e) => setStartDate(e.target.value)}
                    className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-lg text-white text-sm focus:outline-none focus:border-indigo-500"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1">End Date</label>
                  <input
                    type="date"
                    required
                    value={endDate}
                    onChange={(e) => setEndDate(e.target.value)}
                    className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-lg text-white text-sm focus:outline-none focus:border-indigo-500"
                  />
                </div>
              </div>

              <div className="flex items-center justify-end gap-3 pt-4 border-t border-slate-800">
                <button
                  type="button"
                  onClick={() => setShowGenerateModal(false)}
                  className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-sm"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={generating}
                  className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg text-sm font-medium flex items-center gap-2"
                >
                  {generating ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin" /> Generating...
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4" /> Instantiate Sessions
                    </>
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};
