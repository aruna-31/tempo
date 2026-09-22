import React, { useState } from 'react';
import { FacultyInsight } from '../types';
import {
  Lightbulb,
  Clock,
  Layers,
  Sparkles,
  Compass,
  Users,
  Eye,
  ShieldCheck,
  Filter,
} from 'lucide-react';

interface FacultyInsightSectionProps {
  insights: FacultyInsight[];
}

const CATEGORY_META: Record<
  string,
  { label: string; icon: React.ComponentType<{ className?: string }>; colorClass: string; badgeClass: string }
> = {
  PACING: {
    label: 'Instructional Pacing',
    icon: Clock,
    colorClass: 'text-amber-400 border-amber-500/30 bg-amber-500/10',
    badgeClass: 'bg-amber-400/10 text-amber-300 border-amber-400/20',
  },
  ATTENTION_PATTERNS: {
    label: 'Orientation & Gaze Shifts',
    icon: Eye,
    colorClass: 'text-violet-400 border-violet-500/30 bg-violet-500/10',
    badgeClass: 'bg-violet-400/10 text-violet-300 border-violet-400/20',
  },
  INTERACTION: {
    label: 'Peer Collaboration',
    icon: Users,
    colorClass: 'text-sky-400 border-sky-500/30 bg-sky-500/10',
    badgeClass: 'bg-sky-400/10 text-sky-300 border-sky-400/20',
  },
  VARIETY: {
    label: 'Activity Diversity & Entropy',
    icon: Layers,
    colorClass: 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10',
    badgeClass: 'bg-emerald-400/10 text-emerald-300 border-emerald-400/20',
  },
  COVERAGE: {
    label: 'Observational Coverage Context',
    icon: ShieldCheck,
    colorClass: 'text-indigo-400 border-indigo-500/30 bg-indigo-500/10',
    badgeClass: 'bg-indigo-400/10 text-indigo-300 border-indigo-400/20',
  },
};

export const FacultyInsightSection: React.FC<FacultyInsightSectionProps> = ({ insights }) => {
  const [selectedCategory, setSelectedCategory] = useState<string>('ALL');

  if (!insights || insights.length === 0) {
    return null;
  }

  const filteredInsights =
    selectedCategory === 'ALL'
      ? insights
      : insights.filter((item) => item.category === selectedCategory);

  const categories = ['ALL', ...Array.from(new Set(insights.map((item) => item.category)))];

  const formatTime = (seconds?: number | null) => {
    if (seconds === undefined || seconds === null) return null;
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <section className="space-y-5">
      {/* Header with Title and Ethical Notice */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-xs font-semibold text-indigo-400 uppercase tracking-wider mb-1">
            <Lightbulb className="h-4 w-4" />
            <span>Post-Class Faculty Insight Engine</span>
          </div>
          <h2 className="text-xl font-bold text-white tracking-tight">
            Suggestions for Future Classes
          </h2>
          <p className="text-xs text-slate-400 mt-1 max-w-3xl">
            Class-level observations derived from temporal analytics (states, entropy, transitions,
            and tracking stability). Formulated as constructive guidance for subsequent lesson planning.
          </p>
        </div>

        {/* Category Filters */}
        <div className="flex items-center gap-1.5 flex-wrap">
          <Filter className="h-3.5 w-3.5 text-slate-500 mr-1" />
          {categories.map((cat) => {
            const isSelected = selectedCategory === cat;
            const meta = CATEGORY_META[cat];
            const label = cat === 'ALL' ? 'All Categories' : meta?.label || cat;
            return (
              <button
                key={cat}
                onClick={() => setSelectedCategory(cat)}
                className={`text-xs px-2.5 py-1 rounded-md transition-all font-medium ${
                  isSelected
                    ? 'bg-indigo-600 text-white shadow-sm shadow-indigo-500/20'
                    : 'bg-slate-800/80 text-slate-400 hover:text-slate-200 hover:bg-slate-800'
                }`}
              >
                {label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Ethical AI Callout Notice */}
      <div className="flex items-start gap-3 rounded-lg border border-slate-800 bg-slate-900/60 p-3.5 text-xs text-slate-400">
        <Compass className="h-4 w-4 text-indigo-400 flex-shrink-0 mt-0.5" />
        <div className="space-y-0.5">
          <span className="font-semibold text-slate-300">Ethical Pedagogical Assurance:</span>
          <span>
            {' '}
            TEMPO analyzes observable physical posture, gaze orientation, and multimodal activity cadence.
            It does not measure internal cognitive effort, emotional state, or student capability.
            All suggestions are strictly aggregate and forward-looking.
          </span>
        </div>
      </div>

      {/* Grid of Insight Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {filteredInsights.map((insight) => {
          const meta = CATEGORY_META[insight.category] || {
            label: insight.category,
            icon: Sparkles,
            colorClass: 'text-indigo-400 border-indigo-500/30 bg-indigo-500/10',
            badgeClass: 'bg-indigo-400/10 text-indigo-300 border-indigo-400/20',
          };
          const Icon = meta.icon;
          const timeRange =
            insight.start_time !== undefined &&
            insight.start_time !== null &&
            insight.end_time !== undefined &&
            insight.end_time !== null
              ? `${formatTime(insight.start_time)} - ${formatTime(insight.end_time)}`
              : 'Class Overview';

          return (
            <div
              key={insight.id}
              className="glass-panel p-5 flex flex-col justify-between space-y-4 rounded-xl border border-slate-800/90 hover:border-slate-700/80 transition-all bg-gradient-to-br from-slate-900/70 via-slate-900/40 to-slate-950/70"
            >
              <div className="space-y-3">
                {/* Header row: Category Badge & Time */}
                <div className="flex items-center justify-between gap-2">
                  <div
                    className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full border text-[11px] font-medium ${meta.badgeClass}`}
                  >
                    <Icon className="h-3 w-3" />
                    <span>{meta.label}</span>
                  </div>

                  <div className="flex items-center gap-1 text-[11px] text-slate-400 font-mono bg-slate-800/60 px-2 py-0.5 rounded">
                    <Clock className="h-3 w-3 text-slate-500" />
                    <span>{timeRange}</span>
                  </div>
                </div>

                {/* Observable Observation */}
                <div>
                  <h4 className="text-[11px] uppercase tracking-wider font-semibold text-slate-400 mb-1">
                    Observed Activity
                  </h4>
                  <p className="text-sm font-medium text-slate-200 leading-snug">
                    {insight.observation}
                  </p>
                </div>

                {/* Pedagogical Context */}
                <p className="text-xs text-slate-400 leading-relaxed italic border-l-2 border-slate-700 pl-2.5">
                  {insight.pedagogical_context}
                </p>
              </div>

              {/* Actionable Suggestion Callout */}
              <div className="rounded-lg border border-amber-500/20 bg-amber-500/5 p-3.5 space-y-1">
                <div className="flex items-center gap-1.5 text-xs font-semibold text-amber-300">
                  <Lightbulb className="h-3.5 w-3.5 text-amber-400" />
                  <span>Suggestion for Future Classes</span>
                </div>
                <p className="text-xs text-amber-100/90 leading-relaxed">
                  {insight.suggested_action}
                </p>
              </div>

              {/* Footer with Coverage Context */}
              {insight.coverage_context && (
                <div className="pt-1 flex items-center justify-between text-[11px] text-slate-500 border-t border-slate-800/60">
                  <span>{insight.coverage_context}</span>
                  <span className="font-mono text-slate-400">
                    Confidence: {(insight.confidence * 100).toFixed(0)}%
                  </span>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
};
