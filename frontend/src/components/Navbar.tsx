import React from 'react';
import { useAuth } from '../context/AuthContext';
import { LogOut, User, Activity, GraduationCap, ShieldCheck } from 'lucide-react';

export const Navbar: React.FC = () => {
  const { user, logout } = useAuth();

  return (
    <header className="sticky top-0 z-40 w-full border-b border-slate-800/80 bg-slate-950/80 backdrop-blur-md">
      <div className="flex h-16 items-center justify-between px-6">
        {/* Brand Logo & System Title */}
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-tr from-indigo-600 via-indigo-500 to-sky-400 shadow-lg shadow-indigo-500/25">
            <Activity className="h-5 w-5 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-heading text-lg font-bold tracking-tight text-white">TEMPO</span>
              <span className="rounded bg-indigo-500/20 px-1.5 py-0.5 text-[10px] font-semibold text-indigo-300 border border-indigo-500/30">
                v2.0 Individual Tracks
              </span>
            </div>
            <p className="text-xs text-slate-400">Classroom Behaviour Analytics System</p>
          </div>
        </div>

        {/* Faculty Profile and Institution Domain Badge */}
        {user && (
          <div className="flex items-center gap-4">
            <div className="hidden md:flex items-center gap-2 rounded-full border border-sky-500/30 bg-sky-950/40 px-3 py-1 text-xs text-sky-300">
              <ShieldCheck className="h-3.5 w-3.5 text-sky-400" />
              <span>KLU Faculty Portal</span>
            </div>

            <div className="flex items-center gap-3 border-l border-slate-800 pl-4">
              <div className="text-right hidden sm:block">
                <p className="text-sm font-semibold text-slate-200">{user.full_name}</p>
                <p className="text-xs text-slate-400">{user.department || user.email}</p>
              </div>

              <div className="flex h-9 w-9 items-center justify-center rounded-full bg-slate-800 border border-slate-700 text-indigo-400">
                <User className="h-4 w-4" />
              </div>

              <button
                onClick={logout}
                title="Log Out"
                className="flex h-9 w-9 items-center justify-center rounded-lg border border-slate-800 bg-slate-900 text-slate-400 hover:bg-rose-950/40 hover:border-rose-800 hover:text-rose-400 transition-colors"
              >
                <LogOut className="h-4 w-4" />
              </button>
            </div>
          </div>
        )}
      </div>
    </header>
  );
};
