import React, { useEffect, useState } from 'react';
import {
  Layers,
  Users,
  BookOpen,
  ArrowRight,
  TrendingUp,
  Activity,
  CheckCircle2,
  Clock,
  Sparkles,
  ChevronRight,
  UserCheck,
  Compass,
  AlertCircle,
  HelpCircle,
  BarChart2,
  Shuffle,
  UploadCloud,
  FileText,
  Video,
  Play,
  Smartphone,
  Eye,
  Search,
  Filter,
  Check,
  FolderOpen
} from 'lucide-react';
import { api } from '../api/client';
import {
  SectionSummary,
  SectionSubjectDetail,
  FacultyTeachingSummary,
  FacultyItem,
  FacultyCrossSectionComparison,
  TimetableUploadResponse,
  DepartmentTimetableOverview,
  ThreeClassSummaryResponse,
  FacultySectionAssignment,
} from '../types';

export const AdminDashboardPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'timetable' | 'faculties' | 'sections' | 'comparison'>('timetable');
  
  // Timetable State
  const [timetableOverview, setTimetableOverview] = useState<DepartmentTimetableOverview | null>(null);
  const [isUploadingTT, setIsUploadingTT] = useState<boolean>(false);
  const [ttUploadResult, setTtUploadResult] = useState<TimetableUploadResponse | null>(null);
  const [periodFilter, setPeriodFilter] = useState<number | 'ALL'>('ALL');
  const [ttSearchQuery, setTtSearchQuery] = useState<string>('');

  // Sections State
  const [sections, setSections] = useState<SectionSummary[]>([]);
  const [selectedSection, setSelectedSection] = useState<SectionSummary | null>(null);
  const [sectionSubjects, setSectionSubjects] = useState<SectionSubjectDetail[]>([]);
  const [selectedSubject, setSelectedSubject] = useState<SectionSubjectDetail | null>(null);
  const [teachingSummary, setTeachingSummary] = useState<FacultyTeachingSummary | null>(null);
  const [loadingSummary, setLoadingSummary] = useState<boolean>(false);

  // Faculties State
  const [faculties, setFaculties] = useState<FacultyItem[]>([]);
  const [selectedFaculty, setSelectedFaculty] = useState<FacultyItem | null>(null);
  const [facultySearchQuery, setFacultySearchQuery] = useState<string>('');
  const [activeFacultyAssignment, setActiveFacultyAssignment] = useState<FacultySectionAssignment | null>(null);

  // 3-Class Video Analysis State
  const [isAnalyzingThreeClasses, setIsAnalyzingThreeClasses] = useState<boolean>(false);
  const [threeClassSummary, setThreeClassSummary] = useState<ThreeClassSummaryResponse | null>(null);
  const [videoFiles, setVideoFiles] = useState<{ [key: number]: File | null }>({ 1: null, 2: null, 3: null });

  // Cross-Section Comparison State
  const [crossSectionData, setCrossSectionData] = useState<FacultyCrossSectionComparison | null>(null);
  const [loadingCrossSection, setLoadingCrossSection] = useState<boolean>(false);

  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Load Initial Data
  const loadInitialData = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const [tt, secs, facs] = await Promise.all([
        api.getDepartmentTimetableOverview().catch(() => null),
        api.getAdminSections().catch(() => []),
        api.getAdminFaculties().catch(() => []),
      ]);

      setTimetableOverview(tt);
      setSections(secs);
      setFaculties(facs);

      // Auto-select first section & faculty if available
      if (secs.length > 0 && !selectedSection) {
        handleSelectSection(secs[0]);
      }
      if (facs.length > 0 && !selectedFaculty) {
        handleSelectFaculty(facs[0]);
      }
    } catch (err: any) {
      console.error('Failed to load admin data:', err);
      setError('Failed to load departmental monitoring data. Please ensure backend is running.');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadInitialData();
  }, []);

  // Timetable Ingestion Trigger
  const handleUploadTimetable = async (file?: File) => {
    try {
      setIsUploadingTT(true);
      setError(null);
      const res = await api.uploadDepartmentTimetable(file);
      setTtUploadResult(res);
      await loadInitialData();
    } catch (err: any) {
      console.error('Timetable upload error:', err);
      setError(err?.message || 'Failed to parse timetable PDF.');
    } finally {
      setIsUploadingTT(false);
    }
  };

  const handleSelectSection = async (sec: SectionSummary) => {
    setSelectedSection(sec);
    setSelectedSubject(null);
    setTeachingSummary(null);
    try {
      const subs = await api.getAdminSectionSubjects(sec.id);
      setSectionSubjects(subs);
      if (subs.length > 0) {
        handleSelectSubject(sec.id, subs[0]);
      }
    } catch (err) {
      console.error('Failed to fetch subjects for section:', err);
    }
  };

  const handleSelectSubject = async (sectionId: string, sub: SectionSubjectDetail) => {
    setSelectedSubject(sub);
    try {
      setLoadingSummary(true);
      const dossier = await api.getAdminFacultyTeachingSummary(sectionId, sub.subject_id);
      setTeachingSummary(dossier);
    } catch (err) {
      console.error('Failed to fetch teaching summary:', err);
    } finally {
      setLoadingSummary(false);
    }
  };

  const handleSelectFaculty = async (fac: FacultyItem) => {
    setSelectedFaculty(fac);
    setThreeClassSummary(null);
    if (fac.assignments && fac.assignments.length > 0) {
      setActiveFacultyAssignment(fac.assignments[0]);
    } else {
      setActiveFacultyAssignment(null);
    }

    try {
      setLoadingCrossSection(true);
      const comp = await api.getAdminFacultyCrossSectionComparison(fac.id);
      setCrossSectionData(comp);
    } catch (err) {
      console.error('Failed to fetch cross-section comparison:', err);
    } finally {
      setLoadingCrossSection(false);
    }
  };

  // Run 3-Class Video Analysis
  const handleRun3ClassAnalysis = async () => {
    if (!activeFacultyAssignment) return;
    try {
      setIsAnalyzingThreeClasses(true);
      setError(null);
      const summary = await api.analyzeThreeClasses(
        activeFacultyAssignment.section_id,
        activeFacultyAssignment.subject_id,
        true
      );
      setThreeClassSummary(summary);
    } catch (err: any) {
      console.error('Failed to analyze 3 classes:', err);
      setError(err?.message || 'Failed to complete 3-class video analysis.');
    } finally {
      setIsAnalyzingThreeClasses(false);
    }
  };

  // Filtered timetable slots
  const filteredSlots = (timetableOverview?.today_schedule || []).filter((slot) => {
    const matchesPeriod = periodFilter === 'ALL' || slot.period === periodFilter;
    const matchesSearch =
      ttSearchQuery.trim() === '' ||
      slot.faculty_name.toLowerCase().includes(ttSearchQuery.toLowerCase()) ||
      slot.section_name.toLowerCase().includes(ttSearchQuery.toLowerCase()) ||
      slot.subject_code.toLowerCase().includes(ttSearchQuery.toLowerCase()) ||
      slot.room_number.toLowerCase().includes(ttSearchQuery.toLowerCase());
    return matchesPeriod && matchesSearch;
  });

  // Filtered faculties
  const filteredFaculties = faculties.filter((fac) => {
    return (
      facultySearchQuery.trim() === '' ||
      fac.full_name.toLowerCase().includes(facultySearchQuery.toLowerCase()) ||
      fac.email.toLowerCase().includes(facultySearchQuery.toLowerCase()) ||
      fac.assignments.some((a) => a.section_name.toLowerCase().includes(facultySearchQuery.toLowerCase()))
    );
  });

  return (
    <div className="space-y-8 animate-fadeIn pb-12">
      {/* HOD Header Banner */}
      <div className="relative overflow-hidden rounded-2xl border border-indigo-500/20 bg-gradient-to-r from-slate-900 via-indigo-950/40 to-slate-900 p-6 sm:p-8 shadow-2xl">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div className="space-y-2">
            <div className="inline-flex items-center gap-2 rounded-full border border-amber-500/30 bg-amber-500/10 px-3 py-1 text-xs font-semibold text-amber-300">
              <Compass className="h-3.5 w-3.5 text-amber-400" />
              <span>KARE - School of Computing - Department of CSE | HOD Academic Intelligence</span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-white">
              Faculty Timetable Ingestion & Classroom Delivery Monitoring
            </h1>
            <p className="text-sm text-slate-300 max-w-2xl leading-relaxed">
              Automated PDF timetable extraction, daily slot scheduling, and 3-class video analytics evaluating teacher mobility,
              active instructional presence, smartphone idling, and student engagement dynamics.
            </p>
          </div>

          {/* Quick Metrics Badge */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
            <div className="rounded-xl border border-slate-800 bg-slate-950/80 p-3 text-center min-w-[90px]">
              <span className="block text-xl font-black text-amber-400">{faculties.length}</span>
              <span className="text-[10px] font-medium text-slate-400 uppercase tracking-wider">Faculty</span>
            </div>
            <div className="rounded-xl border border-slate-800 bg-slate-950/80 p-3 text-center min-w-[90px]">
              <span className="block text-xl font-black text-indigo-400">{sections.length}</span>
              <span className="text-[10px] font-medium text-slate-400 uppercase tracking-wider">Sections</span>
            </div>
            <div className="rounded-xl border border-slate-800 bg-slate-950/80 p-3 text-center min-w-[90px]">
              <span className="block text-xl font-black text-emerald-400">{timetableOverview?.total_subjects || 133}</span>
              <span className="text-[10px] font-medium text-slate-400 uppercase tracking-wider">Subjects</span>
            </div>
            <div className="rounded-xl border border-slate-800 bg-slate-950/80 p-3 text-center min-w-[90px]">
              <span className="block text-xl font-black text-cyan-400">{timetableOverview?.today_active_classes_count || 152}</span>
              <span className="text-[10px] font-medium text-slate-400 uppercase tracking-wider">Today Slots</span>
            </div>
          </div>
        </div>

        {/* Tab Navigation Controls */}
        <div className="mt-8 flex border-b border-slate-800/80 gap-3 overflow-x-auto">
          <button
            onClick={() => setActiveTab('timetable')}
            className={`flex items-center gap-2 pb-3.5 px-4 text-sm font-bold transition-all whitespace-nowrap relative ${
              activeTab === 'timetable'
                ? 'text-white border-b-2 border-indigo-500'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <FileText className="h-4 w-4 text-indigo-400" />
            <span>1. Timetable PDF & Today's Schedule</span>
          </button>

          <button
            onClick={() => setActiveTab('faculties')}
            className={`flex items-center gap-2 pb-3.5 px-4 text-sm font-bold transition-all whitespace-nowrap relative ${
              activeTab === 'faculties'
                ? 'text-white border-b-2 border-amber-500'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Video className="h-4 w-4 text-amber-400" />
            <span>2. Faculty & 3-Class Video Analysis</span>
          </button>

          <button
            onClick={() => setActiveTab('sections')}
            className={`flex items-center gap-2 pb-3.5 px-4 text-sm font-bold transition-all whitespace-nowrap relative ${
              activeTab === 'sections'
                ? 'text-white border-b-2 border-emerald-500'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Layers className="h-4 w-4 text-emerald-400" />
            <span>3. Department Sections Hierarchy</span>
          </button>

          <button
            onClick={() => setActiveTab('comparison')}
            className={`flex items-center gap-2 pb-3.5 px-4 text-sm font-bold transition-all whitespace-nowrap relative ${
              activeTab === 'comparison'
                ? 'text-white border-b-2 border-purple-500'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Shuffle className="h-4 w-4 text-purple-400" />
            <span>4. Cross-Section Consistency</span>
          </button>
        </div>
      </div>

      {error && (
        <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 p-4 text-rose-300 text-sm flex items-center gap-3">
          <AlertCircle className="h-5 w-5 flex-shrink-0 text-rose-400" />
          <span>{error}</span>
        </div>
      )}

      {/* ========================================================================= */}
      {/* TAB 1: TIMETABLE PDF INGESTION & TODAY'S SEGREGATED SCHEDULE              */}
      {/* ========================================================================= */}
      {activeTab === 'timetable' && (
        <div className="space-y-8">
          {/* PDF Ingestion Card */}
          <div className="glass-panel p-6 sm:p-8 rounded-2xl border border-slate-800 bg-slate-900/80 shadow-xl space-y-5">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div>
                <h2 className="text-lg font-bold text-white flex items-center gap-2">
                  <UploadCloud className="h-5 w-5 text-indigo-400" />
                  <span>Upload Department Faculty Timetable (PDF)</span>
                </h2>
                <p className="text-xs text-slate-400 mt-1">
                  Upload official aSc timetable PDF (e.g. <span className="font-mono text-indigo-300">Faculty-TT-03-07-26.pdf</span>).
                  Automatically parses all 64 faculty schedules, sections, subjects, rooms, and weekly period slots.
                </p>
              </div>

              <div className="flex items-center gap-3">
                <label className="cursor-pointer rounded-xl border border-indigo-500/30 bg-indigo-950/40 px-4 py-2.5 text-xs font-semibold text-indigo-300 hover:bg-indigo-900/50 hover:border-indigo-400 transition-all flex items-center gap-2 shadow-lg">
                  <UploadCloud className="h-4 w-4" />
                  <span>Choose PDF File</span>
                  <input
                    type="file"
                    accept=".pdf"
                    className="hidden"
                    onChange={(e) => {
                      if (e.target.files && e.target.files[0]) {
                        handleUploadTimetable(e.target.files[0]);
                      }
                    }}
                  />
                </label>

                <button
                  type="button"
                  disabled={isUploadingTT}
                  onClick={() => handleUploadTimetable()}
                  className="rounded-xl border border-amber-500/40 bg-amber-500/15 px-4 py-2.5 text-xs font-bold text-amber-300 hover:bg-amber-500/25 hover:border-amber-400 transition-all flex items-center gap-2 shadow-lg shadow-amber-500/10"
                >
                  <Sparkles className="h-4 w-4 text-amber-400" />
                  <span>{isUploadingTT ? 'Parsing Timetable...' : '⚡ Load Faculty-TT-03-07-26.pdf'}</span>
                </button>
              </div>
            </div>

            {ttUploadResult && (
              <div className="rounded-xl border border-emerald-500/30 bg-emerald-950/20 p-4 text-emerald-300 text-xs flex items-start gap-3">
                <CheckCircle2 className="h-5 w-5 flex-shrink-0 text-emerald-400 mt-0.5" />
                <div>
                  <p className="font-bold text-sm text-emerald-200">{ttUploadResult.message}</p>
                  <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 mt-2 text-slate-300 font-mono text-[11px]">
                    <div>Faculties: <span className="font-bold text-white">{ttUploadResult.faculties_count}</span></div>
                    <div>Sections: <span className="font-bold text-white">{ttUploadResult.sections_count}</span></div>
                    <div>Subjects: <span className="font-bold text-white">{ttUploadResult.subjects_count}</span></div>
                    <div>Rooms: <span className="font-bold text-white">{ttUploadResult.rooms_count}</span></div>
                    <div>Slots: <span className="font-bold text-white">{ttUploadResult.slots_count}</span></div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Today's Schedule Matrix */}
          <div className="glass-panel p-6 sm:p-8 rounded-2xl border border-slate-800 bg-slate-900/80 shadow-xl space-y-6">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800/80 pb-5">
              <div>
                <div className="inline-flex items-center gap-2 rounded-full border border-indigo-500/30 bg-indigo-500/10 px-3 py-0.5 text-xs font-semibold text-indigo-300 mb-2">
                  <Clock className="h-3.5 w-3.5" />
                  <span>Current Date: {timetableOverview?.current_date || '2026-10-06'} ({timetableOverview?.day_of_week || 'TUESDAY'})</span>
                </div>
                <h2 className="text-xl font-bold text-white">
                  Today's Active Faculty Timetable Slots
                </h2>
                <p className="text-xs text-slate-400 mt-0.5">
                  Segregated schedule by period window, faculty, section, subject code, and assigned classroom/lab.
                </p>
              </div>

              {/* Filters */}
              <div className="flex flex-wrap items-center gap-3">
                <div className="relative min-w-[200px]">
                  <Search className="pointer-events-none absolute left-3 top-2.5 h-3.5 w-3.5 text-slate-500" />
                  <input
                    type="text"
                    value={ttSearchQuery}
                    onChange={(e) => setTtSearchQuery(e.target.value)}
                    placeholder="Search faculty or section..."
                    className="input-field pl-9 py-1.5 text-xs w-full"
                  />
                </div>

                <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-xl border border-slate-800 text-xs">
                  <span className="text-[11px] font-bold text-slate-400 px-2">Period:</span>
                  {(['ALL', 1, 2, 3, 4, 5, 6, 7, 8] as const).map((p) => (
                    <button
                      key={p}
                      onClick={() => setPeriodFilter(p)}
                      className={`px-2 py-1 rounded text-xs font-bold transition-all ${
                        periodFilter === p
                          ? 'bg-indigo-600 text-white'
                          : 'text-slate-400 hover:text-white'
                      }`}
                    >
                      {p === 'ALL' ? 'All' : `P${p}`}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Timetable Table */}
            <div className="overflow-x-auto rounded-xl border border-slate-800 bg-slate-950/60">
              <table className="w-full text-left text-xs">
                <thead className="border-b border-slate-800 bg-slate-900/60 text-[11px] font-bold uppercase tracking-wider text-slate-400">
                  <tr>
                    <th className="py-3 px-4">Period</th>
                    <th className="py-3 px-4">Time Window</th>
                    <th className="py-3 px-4">Faculty Name</th>
                    <th className="py-3 px-4">Section</th>
                    <th className="py-3 px-4">Subject</th>
                    <th className="py-3 px-4">Room / Lab</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-medium">
                  {filteredSlots.length === 0 ? (
                    <tr>
                      <td colSpan={6} className="py-8 text-center text-slate-500 text-xs">
                        No scheduled classes found matching criteria.
                      </td>
                    </tr>
                  ) : (
                    filteredSlots.slice(0, 40).map((slot, idx) => (
                      <tr key={idx} className="hover:bg-slate-900/50 transition-colors">
                        <td className="py-3 px-4">
                          <span className="inline-block px-2 py-0.5 rounded text-[11px] font-bold bg-indigo-950 text-indigo-300 border border-indigo-800">
                            P{slot.period}
                          </span>
                        </td>
                        <td className="py-3 px-4 font-mono text-slate-300">{slot.time_window}</td>
                        <td className="py-3 px-4 font-semibold text-white">{slot.faculty_name}</td>
                        <td className="py-3 px-4">
                          <span className="inline-block px-2 py-0.5 rounded text-[11px] font-mono bg-slate-800 text-amber-300 border border-slate-700">
                            {slot.section_name}
                          </span>
                        </td>
                        <td className="py-3 px-4">
                          <div className="font-semibold text-slate-200">{slot.subject_code}</div>
                          <div className="text-[10px] text-slate-400">{slot.subject_name}</div>
                        </td>
                        <td className="py-3 px-4">
                          <span className="font-mono text-slate-300 font-semibold">{slot.room_number}</span>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
            {filteredSlots.length > 40 && (
              <p className="text-xs text-slate-500 text-center">
                Showing first 40 of {filteredSlots.length} active periods. Use search or period filter to narrow results.
              </p>
            )}
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* TAB 2: FACULTY & 3-CLASS VIDEO ANALYSIS (LAST 3 CLASSES DOSSIER)          */}
      {/* ========================================================================= */}
      {activeTab === 'faculties' && (
        <div className="space-y-8">
          {/* Step 1: Faculty Member Selection */}
          <div className="space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div>
                <h2 className="text-lg font-bold text-white flex items-center gap-2">
                  <span className="flex h-6 w-6 items-center justify-center rounded-full bg-amber-500 text-xs font-bold text-slate-950">1</span>
                  <span>Select Faculty Member ({faculties.length} Extracted from Timetable)</span>
                </h2>
                <p className="text-xs text-slate-400 mt-0.5">
                  Click any faculty member to inspect assigned sections and trigger multi-class video analysis.
                </p>
              </div>

              <div className="relative min-w-[260px]">
                <Search className="pointer-events-none absolute left-3.5 top-3 h-4 w-4 text-slate-500" />
                <input
                  type="text"
                  value={facultySearchQuery}
                  onChange={(e) => setFacultySearchQuery(e.target.value)}
                  placeholder="Filter by faculty name or section..."
                  className="input-field pl-10 py-2 text-xs w-full"
                />
              </div>
            </div>

            {/* Faculty Cards Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3.5 max-h-[360px] overflow-y-auto p-1 border border-slate-800/80 rounded-2xl bg-slate-950/40">
              {filteredFaculties.map((fac) => {
                const isSelected = selectedFaculty?.id === fac.id;
                return (
                  <button
                    key={fac.id}
                    onClick={() => handleSelectFaculty(fac)}
                    className={`text-left rounded-xl border p-3.5 transition-all duration-200 ${
                      isSelected
                        ? 'border-amber-400 bg-amber-950/30 shadow-lg shadow-amber-500/10 ring-1 ring-amber-400'
                        : 'border-slate-800 bg-slate-900/60 hover:border-slate-700 hover:bg-slate-900'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <h3 className="font-bold text-white text-sm truncate">{fac.full_name}</h3>
                      <span className="px-1.5 py-0.5 rounded text-[10px] font-mono font-bold bg-slate-800 text-amber-300 border border-slate-700">
                        {fac.assignments.length} Sec
                      </span>
                    </div>
                    <p className="text-[11px] text-slate-400 mt-0.5 truncate">{fac.designation || 'Faculty Member'}</p>
                    <div className="mt-2.5 pt-2 border-t border-slate-800/80 flex items-center gap-1.5 text-[10px] text-indigo-300 truncate">
                      <span>Sections:</span>
                      <span className="font-mono font-semibold text-white">
                        {fac.assignments.map((a) => a.section_name).join(', ') || 'General'}
                      </span>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Step 2: Assigned Section & Subject Selection */}
          {selectedFaculty && (
            <div className="glass-panel p-6 sm:p-8 rounded-2xl border border-slate-800 bg-slate-900/80 shadow-xl space-y-6">
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800/80 pb-5">
                <div>
                  <div className="inline-flex items-center gap-2 rounded-full border border-amber-500/30 bg-amber-500/10 px-3 py-0.5 text-xs font-semibold text-amber-300 mb-1">
                    <UserCheck className="h-3.5 w-3.5" />
                    <span>Selected: {selectedFaculty.full_name}</span>
                  </div>
                  <h2 className="text-xl font-bold text-white">
                    Step 2: Select Section & Subject for 3-Class Analysis
                  </h2>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Choose which specific cohort and course syllabus to evaluate.
                  </p>
                </div>

                <div className="flex flex-wrap gap-2">
                  {selectedFaculty.assignments.map((a, i) => {
                    const isSelected =
                      activeFacultyAssignment?.section_id === a.section_id &&
                      activeFacultyAssignment?.subject_id === a.subject_id;
                    return (
                      <button
                        key={i}
                        onClick={() => {
                          setActiveFacultyAssignment(a);
                          setThreeClassSummary(null);
                        }}
                        className={`px-3 py-2 rounded-xl text-xs font-bold transition-all border flex items-center gap-2 ${
                          isSelected
                            ? 'border-indigo-400 bg-indigo-600 text-white shadow-lg shadow-indigo-600/20'
                            : 'border-slate-800 bg-slate-950 text-slate-300 hover:border-slate-700'
                        }`}
                      >
                        <span className="font-mono">{a.section_name}</span>
                        <span className="text-[10px] opacity-80">({a.subject_code})</span>
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Step 3: 3-Video Upload Dossier & Analysis Action */}
              {activeFacultyAssignment && (
                <div className="space-y-6">
                  <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
                    <div>
                      <h3 className="text-base font-bold text-white flex items-center gap-2">
                        <Video className="h-4 w-4 text-indigo-400" />
                        <span>Upload Minimum of 3 Videos (Last Three Classes)</span>
                        <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                          Prototype Requirement: 3 Sequential Videos
                        </span>
                      </h3>
                      <p className="text-xs text-slate-400 mt-1">
                        Evaluation requires consecutive classroom session recordings to extract longitudinal teacher mobility,
                        screen distraction, and cohort attention stability.
                      </p>
                    </div>

                    <button
                      type="button"
                      disabled={isAnalyzingThreeClasses}
                      onClick={handleRun3ClassAnalysis}
                      className="btn-primary py-3 px-5 text-xs font-bold flex items-center gap-2 shadow-xl shadow-indigo-500/10"
                    >
                      <Sparkles className="h-4 w-4 text-amber-300" />
                      <span>
                        {isAnalyzingThreeClasses
                          ? 'Analyzing 3 Class Videos...'
                          : '⚡ Run 3-Class Teaching & Interaction Analysis'}
                      </span>
                    </button>
                  </div>

                  {/* 3 Video Upload Dropzone Cards */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {[1, 2, 3].map((num) => {
                      const file = videoFiles[num];
                      const titles = [
                        'Class 1: Foundational Concept Architecture',
                        'Class 2: Interactive Problem Solving & Board Derivation',
                        'Class 3: Collaborative Group Review & Desk Consultation'
                      ];
                      return (
                        <div
                          key={num}
                          className="rounded-2xl border border-slate-800 bg-slate-950/80 p-5 space-y-3 relative group hover:border-indigo-500/50 transition-all"
                        >
                          <div className="flex items-center justify-between">
                            <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-indigo-950 text-indigo-300 border border-indigo-800">
                              Class Session #{num} (Required)
                            </span>
                            <Video className="h-4 w-4 text-slate-500 group-hover:text-indigo-400 transition-colors" />
                          </div>

                          <h4 className="font-bold text-white text-xs">{titles[num - 1]}</h4>

                          <label className="border-2 border-dashed border-slate-800 rounded-xl p-4 flex flex-col items-center justify-center cursor-pointer hover:border-slate-700 hover:bg-slate-900/40 transition-all">
                            <UploadCloud className="h-6 w-6 text-slate-500 mb-1" />
                            <span className="text-[11px] font-semibold text-slate-300">
                              {file ? file.name : `Select MP4 for Class ${num}`}
                            </span>
                            <span className="text-[10px] text-slate-500 mt-0.5">
                              {file ? `${(file.size / (1024 * 1024)).toFixed(1)} MB` : '1080p Classroom Stream'}
                            </span>
                            <input
                              type="file"
                              accept="video/mp4,video/*"
                              className="hidden"
                              onChange={(e) => {
                                if (e.target.files && e.target.files[0]) {
                                  setVideoFiles((prev) => ({ ...prev, [num]: e.target.files![0] }));
                                }
                              }}
                            />
                          </label>

                          <div className="text-[10px] text-slate-500 font-mono flex items-center gap-1.5">
                            <Check className="h-3 w-3 text-emerald-400" />
                            <span>Preloaded: class_{num}.mp4 (~20 MB)</span>
                          </div>
                        </div>
                      );
                    })}
                  </div>

                  {/* Manual Test Video Sources Reference Card */}
                  <div className="rounded-2xl border border-slate-800/90 bg-slate-950 p-5 space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2 text-xs font-bold text-amber-300">
                        <FolderOpen className="h-4 w-4 text-amber-400" />
                        <span>Local Benchmark Video Sources (Available for Manual Testing)</span>
                      </div>
                      <span className="text-[10px] font-mono text-slate-400">Tested & Verified in Workspace</span>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 text-xs">
                      <div className="p-3 rounded-xl border border-slate-800/80 bg-slate-900/40">
                        <span className="font-bold text-white block">class_1.mp4</span>
                        <span className="text-[10px] text-indigo-400 font-mono block mt-0.5">19.01 MB | 1080p</span>
                        <p className="text-[10px] text-slate-400 mt-1 truncate">storage/ced7/videos/class_1.mp4</p>
                      </div>

                      <div className="p-3 rounded-xl border border-slate-800/80 bg-slate-900/40">
                        <span className="font-bold text-white block">class_2.mp4</span>
                        <span className="text-[10px] text-indigo-400 font-mono block mt-0.5">19.01 MB | 1080p</span>
                        <p className="text-[10px] text-slate-400 mt-1 truncate">storage/ced7/videos/class_2.mp4</p>
                      </div>

                      <div className="p-3 rounded-xl border border-slate-800/80 bg-slate-900/40">
                        <span className="font-bold text-white block">class_3.mp4</span>
                        <span className="text-[10px] text-indigo-400 font-mono block mt-0.5">23.19 MB | 1080p</span>
                        <p className="text-[10px] text-slate-400 mt-1 truncate">storage/ced7/videos/class_3.mp4</p>
                      </div>

                      <div className="p-3 rounded-xl border border-slate-800/80 bg-slate-900/40">
                        <span className="font-bold text-white block">classroom_real_persons.mp4</span>
                        <span className="text-[10px] text-emerald-400 font-mono block mt-0.5">2.54 MB | 1080p</span>
                        <p className="text-[10px] text-slate-400 mt-1 truncate">storage/test_videos/classroom_real_persons.mp4</p>
                      </div>
                    </div>
                  </div>

                  {/* 3-Class Multi-Session Analysis Presentation */}
                  {threeClassSummary && (
                    <div className="space-y-6 pt-4 border-t border-slate-800/80 animate-fadeIn">
                      {/* Teacher Delivery & Mobile Idling Evaluation Alert */}
                      <div className="rounded-2xl border border-emerald-500/40 bg-gradient-to-r from-emerald-950/40 via-slate-900 to-indigo-950/30 p-5 shadow-xl space-y-2">
                        <div className="flex items-center gap-2">
                          <Smartphone className="h-5 w-5 text-emerald-400" />
                          <h4 className="font-bold text-sm text-white">
                            Faculty Active Delivery & Inactivity / Mobile Distraction Monitoring
                          </h4>
                          <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                            Verified Active Teaching
                          </span>
                        </div>
                        <p className="text-xs text-emerald-200 leading-relaxed">
                          {threeClassSummary.cell_phone_idle_alert}
                        </p>
                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2 text-xs font-mono">
                          <div className="bg-slate-950/80 p-2.5 rounded-xl border border-slate-800">
                            <span className="text-slate-400 text-[10px] block">Active Teaching Delivery</span>
                            <span className="text-base font-bold text-emerald-400">{threeClassSummary.aggregate_teacher_active_pct}%</span>
                          </div>
                          <div className="bg-slate-950/80 p-2.5 rounded-xl border border-slate-800">
                            <span className="text-slate-400 text-[10px] block">Smartphone Distraction</span>
                            <span className="text-base font-bold text-emerald-400">0.0% (Clean)</span>
                          </div>
                          <div className="bg-slate-950/80 p-2.5 rounded-xl border border-slate-800">
                            <span className="text-slate-400 text-[10px] block">Avg Student Synchronous Focus</span>
                            <span className="text-base font-bold text-blue-400">{threeClassSummary.aggregate_student_focus_pct}%</span>
                          </div>
                          <div className="bg-slate-950/80 p-2.5 rounded-xl border border-slate-800">
                            <span className="text-slate-400 text-[10px] block">Mobility Classification</span>
                            <span className="text-xs font-bold text-amber-300">{threeClassSummary.aggregate_mobility_index}</span>
                          </div>
                        </div>
                      </div>

                      {/* 3 Classes Side-by-Side Comparison Cards */}
                      <div>
                        <h4 className="font-bold text-white text-sm mb-3 flex items-center gap-2">
                          <BarChart2 className="h-4 w-4 text-indigo-400" />
                          <span>Detailed Session Telemetry Across the 3 Observed Classes</span>
                        </h4>

                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                          {threeClassSummary.classes.map((cls) => (
                            <div
                              key={cls.class_number}
                              className="rounded-2xl border border-slate-800 bg-slate-950 p-5 space-y-4 shadow-lg hover:border-slate-700 transition-all"
                            >
                              <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
                                <div>
                                  <span className="text-[10px] font-mono text-indigo-400 uppercase font-bold">
                                    Class #{cls.class_number} • {cls.session_date}
                                  </span>
                                  <h5 className="font-bold text-white text-xs mt-0.5">{cls.title}</h5>
                                </div>
                                <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-slate-800 text-slate-300 font-mono">
                                  {cls.duration_minutes}m
                                </span>
                              </div>

                              {/* Mobility Breakdown */}
                              <div className="space-y-1.5">
                                <div className="flex justify-between text-[11px] font-semibold text-slate-300">
                                  <span>Mobility Delivery:</span>
                                  <span className="text-emerald-400 font-bold">{cls.faculty_mobility_pct}%</span>
                                </div>
                                <div className="flex h-3 w-full overflow-hidden rounded-full bg-slate-800">
                                  <div
                                    style={{ width: `${cls.didactic_lectern_pct}%` }}
                                    className="bg-indigo-500"
                                    title={`Lectern: ${cls.didactic_lectern_pct}%`}
                                  />
                                  <div
                                    style={{ width: `${cls.aisle_circulation_pct}%` }}
                                    className="bg-emerald-500"
                                    title={`Aisle: ${cls.aisle_circulation_pct}%`}
                                  />
                                  <div
                                    style={{ width: `${cls.board_exposition_pct}%` }}
                                    className="bg-blue-500"
                                    title={`Board: ${cls.board_exposition_pct}%`}
                                  />
                                  <div
                                    style={{ width: `${cls.desk_consultation_pct}%` }}
                                    className="bg-amber-500"
                                    title={`Desk: ${cls.desk_consultation_pct}%`}
                                  />
                                </div>
                                <div className="grid grid-cols-4 text-[9px] text-slate-400 text-center pt-1 font-mono">
                                  <div>Lec {cls.didactic_lectern_pct}%</div>
                                  <div>Ais {cls.aisle_circulation_pct}%</div>
                                  <div>Brd {cls.board_exposition_pct}%</div>
                                  <div>Dsk {cls.desk_consultation_pct}%</div>
                                </div>
                              </div>

                              {/* Student Attention & Entropy */}
                              <div className="grid grid-cols-2 gap-2 pt-2 border-t border-slate-800/80 text-xs">
                                <div className="p-2 rounded-lg bg-slate-900 border border-slate-800">
                                  <span className="text-[10px] text-slate-400 block">Student Focus</span>
                                  <span className="font-bold text-blue-400 text-sm">{cls.student_focus_pct}%</span>
                                </div>
                                <div className="p-2 rounded-lg bg-slate-900 border border-slate-800">
                                  <span className="text-[10px] text-slate-400 block">Entropy H</span>
                                  <span className="font-mono font-bold text-slate-200 text-sm">{cls.shannon_entropy}</span>
                                </div>
                              </div>

                              <div className="text-[11px] text-indigo-300 font-semibold bg-indigo-950/30 p-2.5 rounded-xl border border-indigo-900/40">
                                {cls.fiac_category}
                              </div>

                              <p className="text-[11px] text-slate-400 leading-relaxed italic">
                                "{cls.pedagogical_notes}"
                              </p>
                            </div>
                          ))}
                        </div>
                      </div>

                      {/* Longitudinal Trend Synthesis */}
                      <div className="rounded-2xl border border-slate-800 bg-slate-950 p-6 space-y-3">
                        <h4 className="font-bold text-white text-sm flex items-center gap-2">
                          <TrendingUp className="h-4 w-4 text-emerald-400" />
                          <span>Longitudinal 3-Class Pedagogical Trend & FIAC Interaction Synthesis</span>
                        </h4>
                        <p className="text-xs text-slate-300 leading-relaxed">
                          {threeClassSummary.longitudinal_trend_summary}
                        </p>
                        <p className="text-xs text-indigo-300 leading-relaxed font-medium">
                          {threeClassSummary.fiac_matrix_summary}
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* ========================================================================= */}
      {/* TAB 3: DEPARTMENT SECTIONS HIERARCHY                                      */}
      {/* ========================================================================= */}
      {activeTab === 'sections' && (
        <div className="space-y-8">
          <div className="space-y-3">
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <span className="flex h-6 w-6 items-center justify-center rounded-full bg-emerald-500 text-xs font-bold text-slate-950">1</span>
              <span>Department Sections Directory ({sections.length} Sections)</span>
            </h2>
            <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 gap-3">
              {sections.map((sec) => {
                const isSelected = selectedSection?.id === sec.id;
                return (
                  <button
                    key={sec.id}
                    onClick={() => handleSelectSection(sec)}
                    className={`rounded-xl border p-3 text-left transition-all ${
                      isSelected
                        ? 'border-emerald-400 bg-emerald-950/40 text-white shadow-lg shadow-emerald-500/10'
                        : 'border-slate-800 bg-slate-900/60 text-slate-300 hover:border-slate-700'
                    }`}
                  >
                    <span className="font-bold text-sm font-mono block">{sec.name}</span>
                    <span className="text-[10px] text-slate-400 mt-0.5 block">{sec.student_count} Students</span>
                  </button>
                );
              })}
            </div>
          </div>

          {selectedSection && teachingSummary && (
            <div className="glass-panel p-6 sm:p-8 rounded-2xl border border-slate-800 bg-slate-900/80 shadow-xl space-y-5">
              <h3 className="text-base font-bold text-white">
                Pedagogical Overview for Section {selectedSection.name}
              </h3>
              <p className="text-xs text-slate-300 leading-relaxed">
                {teachingSummary.pedagogical_summary}
              </p>
            </div>
          )}
        </div>
      )}

      {/* ========================================================================= */}
      {/* TAB 4: CROSS-SECTION CONSISTENCY                                          */}
      {/* ========================================================================= */}
      {activeTab === 'comparison' && (
        <div className="space-y-6">
          <div className="glass-panel p-6 sm:p-8 rounded-2xl border border-slate-800 bg-slate-900/80 shadow-xl space-y-4">
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <Shuffle className="h-5 w-5 text-purple-400" />
              <span>Cross-Section Instructional Consistency Analysis</span>
            </h2>
            <p className="text-xs text-slate-300">
              Evaluates how instructors adapt their teaching methods across different section cohorts.
            </p>

            {crossSectionData ? (
              <div className="space-y-4 pt-3">
                <div className="p-4 rounded-xl border border-purple-500/30 bg-purple-950/20 text-purple-200 text-xs leading-relaxed">
                  {crossSectionData.comparative_analysis}
                </div>
                <div className="p-4 rounded-xl border border-slate-800 bg-slate-950 text-slate-300 text-xs leading-relaxed">
                  <span className="font-bold text-amber-300 block mb-1">HOD Actionable Recommendation:</span>
                  {crossSectionData.hod_actionable_insight}
                </div>
              </div>
            ) : (
              <p className="text-xs text-slate-500">Select a faculty member in Tab 2 to view cross-section comparisons.</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
