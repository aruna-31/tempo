import React from 'react';
import { Link } from 'react-router-dom';
import { ArrowRight } from 'lucide-react';

export const LandingPage: React.FC = () => {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex items-center justify-center p-4">
      <div className="text-center space-y-6 max-w-xl">
        <h1 className="text-5xl sm:text-6xl font-extrabold tracking-tight font-heading text-white">
          TEMPO
        </h1>
        <p className="text-base sm:text-lg text-slate-300 leading-relaxed">
          Classroom Temporal Behaviour Analytics — Faculty Portal.
        </p>
        <Link
          to="/login"
          className="btn-primary inline-flex items-center gap-2 px-10 py-4 text-base font-semibold"
        >
          <span>Enter</span>
          <ArrowRight className="h-5 w-5" />
        </Link>
      </div>
    </div>
  );
};
