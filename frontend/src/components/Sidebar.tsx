import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  BookOpen,
  CalendarDays,
  UploadCloud,
  BarChart3,
  Calendar,
  HelpCircle,
  Compass,
} from 'lucide-react';
import { useAuth } from '../context/AuthContext';

export const Sidebar: React.FC = () => {
  const { user } = useAuth();
  const isHOD = user?.role === 'HOD' || user?.role === 'ADMIN';

  const navigationItems = [
    ...(isHOD
      ? [{ name: 'HOD Academic Monitoring', to: '/admin-monitoring', icon: Compass }]
      : [{ name: 'Faculty Dashboard', to: '/dashboard', icon: LayoutDashboard }]),
    { name: 'Subjects & Sections', to: '/subjects', icon: BookOpen },
    { name: 'Faculty Timetable', to: '/schedules', icon: Calendar },
    { name: 'Class Sessions', to: '/sessions', icon: CalendarDays },
    { name: 'Upload Video', to: '/upload', icon: UploadCloud },
    { name: 'Results & Analytics', to: '/results', icon: BarChart3 },
    ...(isHOD
      ? [{ name: 'Faculty Dashboard View', to: '/dashboard', icon: LayoutDashboard }]
      : []),
  ];

  return (
    <aside className="w-64 flex-shrink-0 border-r border-slate-800/80 bg-slate-950/60 p-4 min-h-[calc(100vh-4rem)] flex flex-col justify-between">
      <div className="space-y-1">
        <div className="px-3 pb-2 flex items-center justify-between">
          <p className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
            {isHOD ? 'HOD Administration' : 'Faculty Navigation'}
          </p>
          {isHOD && (
            <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-amber-500/20 text-amber-300 border border-amber-500/30">
              HOD
            </span>
          )}
        </div>

        {navigationItems.map((item) => {
          const Icon = item.icon;
          const isHodPrimary = item.to === '/admin-monitoring';
          return (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                `flex items-center gap-3 rounded-xl px-3.5 py-2.5 text-sm font-medium transition-all duration-150 ${
                  isActive
                    ? isHodPrimary
                      ? 'bg-gradient-to-r from-amber-600/90 to-amber-700/80 text-white shadow-md shadow-amber-600/20 border border-amber-500/30 font-bold'
                      : 'bg-gradient-to-r from-indigo-600/90 to-indigo-700/80 text-white shadow-md shadow-indigo-600/20 border border-indigo-500/30'
                    : 'text-slate-400 hover:bg-slate-900/80 hover:text-slate-200'
                }`
              }
            >
              <Icon className={`h-4 w-4 flex-shrink-0 ${isHodPrimary ? 'text-amber-300' : ''}`} />
              <span>{item.name}</span>
            </NavLink>
          );
        })}
      </div>

      {/* Ethical Guidance Card */}
      <div className="rounded-xl border border-indigo-900/40 bg-indigo-950/20 p-3.5 text-xs text-indigo-200/80">
        <div className="flex items-center gap-1.5 font-semibold text-indigo-300 pb-1">
          <HelpCircle className="h-3.5 w-3.5 text-indigo-400" />
          <span>Observable Analytics</span>
        </div>
        <p className="text-[11px] leading-relaxed text-slate-400">
          Temporary Session Track IDs only. Analyzes observable visual engagement without facial recognition or student grading.
        </p>
      </div>
    </aside>
  );
};
