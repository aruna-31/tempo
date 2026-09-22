import React from 'react';
import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Legend,
  AreaChart,
  Area,
} from 'recharts';
import { StudentTrackMetrics, TemporalBehaviourBucket } from '../types';

interface BehaviorChartsProps {
  trackMetrics: StudentTrackMetrics[];
  timelineBuckets: TemporalBehaviourBucket[];
}

const COLORS = {
  instruction: '#38bdf8', // Sky blue
  reading: '#34d399',     // Emerald
  writing: '#818cf8',     // Indigo
  peer: '#fbbf24',        // Amber
  away: '#fb7185',        // Rose
};

export const BehaviorCharts: React.FC<BehaviorChartsProps> = ({
  trackMetrics,
  timelineBuckets,
}) => {
  // Aggregate class-level percentages across all tracks
  const totalTracks = trackMetrics.length || 1;
  const classTotals = trackMetrics.reduce(
    (acc, t) => ({
      instruction: acc.instruction + t.looking_toward_instruction_pct,
      reading: acc.reading + t.reading_pct,
      writing: acc.writing + t.writing_pct,
      peer: acc.peer + t.peer_interaction_pct,
      away: acc.away + t.looking_away_pct,
    }),
    { instruction: 0, reading: 0, writing: 0, peer: 0, away: 0 }
  );

  const pieData = [
    { name: 'Looking Toward Instruction', value: Math.round(classTotals.instruction / totalTracks), color: COLORS.instruction },
    { name: 'Reading', value: Math.round(classTotals.reading / totalTracks), color: COLORS.reading },
    { name: 'Writing', value: Math.round(classTotals.writing / totalTracks), color: COLORS.writing },
    { name: 'Peer Interaction', value: Math.round(classTotals.peer / totalTracks), color: COLORS.peer },
    { name: 'Looking Away', value: Math.round(classTotals.away / totalTracks), color: COLORS.away },
  ].filter(d => d.value > 0);

  // Bar data per track
  const barData = trackMetrics.map((t) => ({
    label: t.display_label,
    Instruction: t.looking_toward_instruction_pct,
    Reading: t.reading_pct,
    Writing: t.writing_pct,
    'Peer Interaction': t.peer_interaction_pct,
    'Looking Away': t.looking_away_pct,
  }));

  // Timeline area data
  const timelineData = timelineBuckets.map((b) => {
    const minStart = Math.floor(b.timestamp_start / 60);
    const minEnd = Math.ceil(b.timestamp_end / 60);
    return {
      time: `${minStart}-${minEnd}m`,
      Instruction: b.looking_toward_instruction_count,
      Reading: b.reading_count,
      Writing: b.writing_count,
      'Peer Interaction': b.peer_interaction_count,
      'Looking Away': b.looking_away_count,
      'Engagement Index': b.engagement_index,
    };
  });

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 1. Class Activity Distribution Donut */}
        <div className="glass-panel p-5 flex flex-col justify-between">
          <div>
            <h3 className="text-base font-semibold text-white">Classroom Activity Distribution</h3>
            <p className="text-xs text-slate-400">Aggregated observable behaviour breakdown</p>
          </div>

          <div className="h-64 w-full my-2">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={55}
                  outerRadius={85}
                  paddingAngle={4}
                  dataKey="value"
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(15, 23, 42, 0.95)',
                    border: '1px solid rgba(148, 163, 184, 0.2)',
                    borderRadius: '0.5rem',
                    color: '#f8fafc',
                    fontSize: '12px',
                  }}
                  formatter={(value: any) => [`${value}%`, 'Share']}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>

          <div className="grid grid-cols-2 gap-2 text-xs pt-2 border-t border-slate-800">
            {pieData.map((d) => (
              <div key={d.name} className="flex items-center gap-1.5">
                <span className="h-2.5 w-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: d.color }} />
                <span className="text-slate-300 truncate">{d.name}:</span>
                <span className="font-semibold text-white font-mono">{d.value}%</span>
              </div>
            ))}
          </div>
        </div>

        {/* 2. Individual Track Comparison Stacked Bar */}
        <div className="glass-panel p-5 lg:col-span-2">
          <div className="flex items-center justify-between pb-3">
            <div>
              <h3 className="text-base font-semibold text-white">Student Track Activity Breakdown</h3>
              <p className="text-xs text-slate-400">Comparing individual temporary Track IDs</p>
            </div>
          </div>

          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={barData} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.1)" />
                <XAxis dataKey="label" stroke="#94a3b8" fontSize={11} tickLine={false} />
                <YAxis stroke="#94a3b8" fontSize={11} unit="%" tickLine={false} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(15, 23, 42, 0.95)',
                    border: '1px solid rgba(148, 163, 184, 0.2)',
                    borderRadius: '0.5rem',
                    color: '#f8fafc',
                    fontSize: '12px',
                  }}
                  formatter={(value: any, name: string) => [`${value}%`, name]}
                />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
                <Bar dataKey="Instruction" fill={COLORS.instruction} stackId="a" radius={[0, 0, 0, 0]} />
                <Bar dataKey="Reading" fill={COLORS.reading} stackId="a" />
                <Bar dataKey="Writing" fill={COLORS.writing} stackId="a" />
                <Bar dataKey="Peer Interaction" fill={COLORS.peer} stackId="a" />
                <Bar dataKey="Looking Away" fill={COLORS.away} stackId="a" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* 3. Temporal Classroom Behaviour Timeline */}
      {timelineData.length > 0 && (
        <div className="glass-panel p-5">
          <div className="flex items-center justify-between pb-3">
            <div>
              <h3 className="text-base font-semibold text-white">Temporal Classroom Timeline</h3>
              <p className="text-xs text-slate-400">Observing engagement dynamics over class progression</p>
            </div>
          </div>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={timelineData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="colorEngagement" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#6366f1" stopOpacity={0.8} />
                    <stop offset="95%" stopColor="#6366f1" stopOpacity={0.0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.1)" />
                <XAxis dataKey="time" stroke="#94a3b8" fontSize={11} />
                <YAxis stroke="#94a3b8" fontSize={11} unit="%" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(15, 23, 42, 0.95)',
                    border: '1px solid rgba(148, 163, 184, 0.2)',
                    borderRadius: '0.5rem',
                    color: '#f8fafc',
                    fontSize: '12px',
                  }}
                />
                <Area
                  type="monotone"
                  dataKey="Engagement Index"
                  stroke="#818cf8"
                  strokeWidth={2}
                  fillOpacity={1}
                  fill="url(#colorEngagement)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
};
