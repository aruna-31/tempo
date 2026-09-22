import React, { useEffect, useState } from 'react';
import { api } from '../api/client';
import { Section, Student, Subject } from '../types';
import { BookOpen, Plus, Users, FolderPlus, UserPlus, X, Check } from 'lucide-react';

export const SubjectsPage: React.FC = () => {
  const [subjects, setSubjects] = useState<Subject[]>([]);
  const [selectedSubject, setSelectedSubject] = useState<Subject | null>(null);
  const [sections, setSections] = useState<Section[]>([]);
  const [selectedSection, setSelectedSection] = useState<Section | null>(null);
  const [students, setStudents] = useState<Student[]>([]);

  // Modals state
  const [showSubjectModal, setShowSubjectModal] = useState(false);
  const [newSubCode, setNewSubCode] = useState('');
  const [newSubName, setNewSubName] = useState('');

  const [showSectionModal, setShowSectionModal] = useState(false);
  const [newSecName, setNewSecName] = useState('Sec-A');
  const [newSecYear, setNewSecYear] = useState('2025-2026');
  const [newSecSem, setNewSecSem] = useState('Semester 5');

  const [showStudentModal, setShowStudentModal] = useState(false);
  const [newStudentRoll, setNewStudentRoll] = useState('');
  const [newStudentName, setNewStudentName] = useState('');
  const [newStudentEmail, setNewStudentEmail] = useState('');

  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadSubjects();
  }, []);

  const loadSubjects = async () => {
    try {
      const list = await api.getSubjects();
      setSubjects(list);
      if (list.length > 0 && !selectedSubject) {
        selectSubject(list[0]);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const selectSubject = async (sub: Subject) => {
    setSelectedSubject(sub);
    try {
      const secList = await api.getSections(sub.id);
      setSections(secList);
      if (secList.length > 0) {
        selectSection(secList[0]);
      } else {
        setSelectedSection(null);
        setStudents([]);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const selectSection = async (sec: Section) => {
    setSelectedSection(sec);
    try {
      const stuList = await api.getStudents(sec.id);
      setStudents(stuList);
    } catch (err) {
      console.error(err);
    }
  };

  const handleCreateSubject = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newSubCode || !newSubName) return;
    try {
      const created = await api.createSubject({ code: newSubCode, name: newSubName });
      setSubjects([...subjects, created]);
      selectSubject(created);
      setShowSubjectModal(false);
      setNewSubCode('');
      setNewSubName('');
    } catch (err: any) {
      alert(err.message);
    }
  };

  const handleCreateSection = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedSubject) return;
    try {
      const created = await api.createSection(selectedSubject.id, {
        name: newSecName,
        academic_year: newSecYear,
        semester: newSecSem,
      });
      setSections([...sections, created]);
      selectSection(created);
      setShowSectionModal(false);
    } catch (err: any) {
      alert(err.message);
    }
  };

  const handleCreateStudent = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedSection) return;
    try {
      const created = await api.createStudent(selectedSection.id, {
        roll_number: newStudentRoll,
        name: newStudentName,
        email: newStudentEmail || `${newStudentRoll.toLowerCase()}@klu.ac.in`,
      });
      setStudents([...students, created]);
      setShowStudentModal(false);
      setNewStudentRoll('');
      setNewStudentName('');
      setNewStudentEmail('');
    } catch (err: any) {
      alert(err.message);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Subjects & Sections</h1>
          <p className="text-xs text-slate-400">Manage academic courses, section batches, and student rosters</p>
        </div>
        <button onClick={() => setShowSubjectModal(true)} className="btn-primary">
          <Plus className="h-4 w-4" />
          <span>Add Subject</span>
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Left Column: Subjects List */}
        <div className="glass-panel p-5">
          <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-400 mb-3">
            Subjects List
          </h3>

          <div className="space-y-2">
            {subjects.length === 0 ? (
              <p className="text-xs text-slate-500 py-4 text-center">No subjects created yet.</p>
            ) : (
              subjects.map((sub) => (
                <div
                  key={sub.id}
                  onClick={() => selectSubject(sub)}
                  className={`p-3 rounded-xl border cursor-pointer transition-all ${
                    selectedSubject?.id === sub.id
                      ? 'border-indigo-500 bg-indigo-950/40 text-white'
                      : 'border-slate-800 bg-slate-900/50 text-slate-300 hover:border-slate-700'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-xs font-bold text-indigo-400">{sub.code}</span>
                    <BookOpen className="h-4 w-4 text-slate-500" />
                  </div>
                  <p className="font-semibold text-sm mt-1">{sub.name}</p>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Center Column: Sections for Selected Subject */}
        <div className="glass-panel p-5">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-400">
              Sections ({selectedSubject?.code || 'Select'})
            </h3>
            {selectedSubject && (
              <button
                onClick={() => setShowSectionModal(true)}
                className="text-xs font-semibold text-indigo-400 hover:text-indigo-300 flex items-center gap-1"
              >
                <Plus className="h-3.5 w-3.5" />
                <span>Add Section</span>
              </button>
            )}
          </div>

          <div className="space-y-2">
            {!selectedSubject ? (
              <p className="text-xs text-slate-500 py-4 text-center">Select a subject to view sections.</p>
            ) : sections.length === 0 ? (
              <p className="text-xs text-slate-500 py-4 text-center">No sections configured for this subject.</p>
            ) : (
              sections.map((sec) => (
                <div
                  key={sec.id}
                  onClick={() => selectSection(sec)}
                  className={`p-3 rounded-xl border cursor-pointer transition-all ${
                    selectedSection?.id === sec.id
                      ? 'border-indigo-500 bg-indigo-950/40 text-white'
                      : 'border-slate-800 bg-slate-900/50 text-slate-300 hover:border-slate-700'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-sm">{sec.name}</span>
                    <span className="text-[11px] text-slate-400">{sec.semester}</span>
                  </div>
                  <p className="text-xs text-slate-400 mt-1 font-mono">AY: {sec.academic_year}</p>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Right Column: Student Roster for Selected Section */}
        <div className="glass-panel p-5">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-400">
              Student Roster ({selectedSection?.name || 'Select'})
            </h3>
            {selectedSection && (
              <button
                onClick={() => setShowStudentModal(true)}
                className="text-xs font-semibold text-indigo-400 hover:text-indigo-300 flex items-center gap-1"
              >
                <UserPlus className="h-3.5 w-3.5" />
                <span>Add Student</span>
              </button>
            )}
          </div>

          <div className="space-y-2 max-h-96 overflow-y-auto pr-1">
            {!selectedSection ? (
              <p className="text-xs text-slate-500 py-4 text-center">Select a section to view student roster.</p>
            ) : students.length === 0 ? (
              <p className="text-xs text-slate-500 py-4 text-center">No students registered in this section yet.</p>
            ) : (
              students.map((stu) => (
                <div key={stu.id} className="p-2.5 rounded-xl border border-slate-800 bg-slate-900/40 text-xs">
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-white">{stu.name}</span>
                    <span className="font-mono text-[11px] text-indigo-400">{stu.roll_number}</span>
                  </div>
                  <p className="text-slate-400 text-[11px] mt-0.5 truncate">{stu.email}</p>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* Subject Modal */}
      {showSubjectModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
          <div className="glass-panel p-6 max-w-md w-full border border-slate-700 shadow-2xl">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800 mb-4">
              <h3 className="text-base font-bold text-white">Create New Subject</h3>
              <button onClick={() => setShowSubjectModal(false)} className="text-slate-400 hover:text-white">
                <X className="h-5 w-5" />
              </button>
            </div>
            <form onSubmit={handleCreateSubject} className="space-y-4">
              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">Subject Code</label>
                <input
                  type="text"
                  required
                  value={newSubCode}
                  onChange={(e) => setNewSubCode(e.target.value)}
                  placeholder="e.g. 23CS3101"
                  className="input-field uppercase"
                />
              </div>
              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">Subject Name</label>
                <input
                  type="text"
                  required
                  value={newSubName}
                  onChange={(e) => setNewSubName(e.target.value)}
                  placeholder="e.g. Artificial Intelligence & Machine Learning"
                  className="input-field"
                />
              </div>
              <div className="flex justify-end gap-3 pt-2">
                <button type="button" onClick={() => setShowSubjectModal(false)} className="btn-secondary">
                  Cancel
                </button>
                <button type="submit" className="btn-primary">
                  Save Subject
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Section Modal */}
      {showSectionModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
          <div className="glass-panel p-6 max-w-md w-full border border-slate-700 shadow-2xl">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800 mb-4">
              <h3 className="text-base font-bold text-white">Add Section for {selectedSubject?.code}</h3>
              <button onClick={() => setShowSectionModal(false)} className="text-slate-400 hover:text-white">
                <X className="h-5 w-5" />
              </button>
            </div>
            <form onSubmit={handleCreateSection} className="space-y-4">
              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">Section Name</label>
                <input
                  type="text"
                  required
                  value={newSecName}
                  onChange={(e) => setNewSecName(e.target.value)}
                  placeholder="e.g. Section A"
                  className="input-field"
                />
              </div>
              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">Academic Year</label>
                <input
                  type="text"
                  required
                  value={newSecYear}
                  onChange={(e) => setNewSecYear(e.target.value)}
                  placeholder="2025-2026"
                  className="input-field"
                />
              </div>
              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">Semester</label>
                <input
                  type="text"
                  required
                  value={newSecSem}
                  onChange={(e) => setNewSecSem(e.target.value)}
                  placeholder="Semester 5"
                  className="input-field"
                />
              </div>
              <div className="flex justify-end gap-3 pt-2">
                <button type="button" onClick={() => setShowSectionModal(false)} className="btn-secondary">
                  Cancel
                </button>
                <button type="submit" className="btn-primary">
                  Save Section
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Student Modal */}
      {showStudentModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
          <div className="glass-panel p-6 max-w-md w-full border border-slate-700 shadow-2xl">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800 mb-4">
              <h3 className="text-base font-bold text-white">Add Student to {selectedSection?.name}</h3>
              <button onClick={() => setShowStudentModal(false)} className="text-slate-400 hover:text-white">
                <X className="h-5 w-5" />
              </button>
            </div>
            <form onSubmit={handleCreateStudent} className="space-y-4">
              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">Roll Number</label>
                <input
                  type="text"
                  required
                  value={newStudentRoll}
                  onChange={(e) => setNewStudentRoll(e.target.value)}
                  placeholder="e.g. 2300030001"
                  className="input-field"
                />
              </div>
              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">Student Full Name</label>
                <input
                  type="text"
                  required
                  value={newStudentName}
                  onChange={(e) => setNewStudentName(e.target.value)}
                  placeholder="e.g. Ananya Rao"
                  className="input-field"
                />
              </div>
              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">Student Email (Optional)</label>
                <input
                  type="email"
                  value={newStudentEmail}
                  onChange={(e) => setNewStudentEmail(e.target.value)}
                  placeholder="2300030001@klu.ac.in"
                  className="input-field"
                />
              </div>
              <div className="flex justify-end gap-3 pt-2">
                <button type="button" onClick={() => setShowStudentModal(false)} className="btn-secondary">
                  Cancel
                </button>
                <button type="submit" className="btn-primary">
                  Register Student
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};
