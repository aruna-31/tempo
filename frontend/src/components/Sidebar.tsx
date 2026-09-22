import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  BookOpen,
  CalendarDays,
  UploadCloud,
  BarChart3,
  Calendar,
  HelpCircle
} from 'lucide-react';

// PROTOTYPE SCOPE: Rooms & Cameras and the separate Processing Status page are
// removed from navigation. Processing state is shown on the session/upload page.
const navigationItems = [
  { name: 'Dashboard', to: '/', icon: LayoutDashboard },
  { name: 'Subjects & Sections', to: '/subjects', icon: BookOpen },
  { name: 'Faculty Timetable', to: '/schedules', icon: Calendar },
  { name: 'Class Sessions', to: '/sessions', icon: CalendarDays },
  { name: 'Upload Video', to: '/upload', icon: UploadCloud },
  { name: 'Results & Analytics', to: '/results', icon: BarChart3 },
];

export const Sidebar: React.FC = () => {
  return (
    <aside className="w-64 flex-shrink-0 border-r border-slate-800/80 bg-slate-950/60 p-4 min-h-[calc(100vh-4rem)] flex flex-col justify-between">
      <div className="space-y-1">
        <p className="px-3 pb-2 text-[11px] font-bold uppercase tracking-wider text-slate-400">
          Analytics Navigation
        </p>

        {navigationItems.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                `flex items-center gap-3 rounded-xl px-3.5 py-2.5 text-sm font-medium transition-all duration-150 ${
                  isActive
                    ? 'bg-gradient-to-r from-indigo-600/90 to-indigo-700/80 text-white shadow-md shadow-indigo-600/20 border border-indigo-500/30'
                    : 'text-slate-400 hover:bg-slate-900/80 hover:text-slate-200'
                }`
              }
            >
              <Icon className="h-4 w-4 flex-shrink-0" />
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
