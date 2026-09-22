import React from 'react';
import { BehaviourResult } from '../types';
import { Clock, CheckCircle2, ChevronRight, User } from 'lucide-react';

interface TrackTimelineProps {
  trackId: number;
  behaviours: BehaviourResult[];
  onSelectTimestamp?: (time: number) => void;
}

const BADGE_MAP: Record<string, { label: string; classStyle: string }> = {
  Looking_Toward_Instruction: { label: 'Instruction', classStyle: 'badge-instruction' },
  Reading: { label: 'Reading', classStyle: 'badge-reading' },
  Writing: { label: 'Writing', classStyle: 'badge-writing' },
  Peer_Interaction: { label: 'Peer Interaction', classStyle: 'badge-peer' },
  Looking_Away: { label: 'Looking Away', classStyle: 'badge-away' },
};

export const TrackTimeline: React.FC<TrackTimelineProps> = ({
  trackId,
  behaviours,
  onSelectTimestamp,
}) => {
  const trackBehaviours = behaviours
    .filter((b) => b.track_id === trackId)
    .sort((a, b) => a.timestamp_seconds - b.timestamp_seconds);

  const formatTimestamp = (sec: number) => {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  return (
    <div className="glass-panel p-5">
      <div className="flex items-center justify-between pb-4 border-b border-slate-800">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-indigo-600/20 text-indigo-400 border border-indigo-500/30">
            <User className="h-5 w-5" />
          </div>
          <div>
            <h4 className="text-base font-semibold text-white">
              Student {trackId.toString().padStart(2, '0')} Temporal Stream
            </h4>
            <p className="text-xs text-slate-400">
              {trackBehaviours.length} observable behaviour detections recorded
            </p>
          </div>
        </div>
      </div>

      <div className="mt-4 max-h-96 overflow-y-auto pr-2 space-y-2.5">
        {trackBehaviours.length === 0 ? (
          <p className="text-xs text-slate-500 py-6 text-center">No timeline events recorded for this track.</p>
        ) : (
          trackBehaviours.map((item, idx) => {
            const badge = BADGE_MAP[item.behaviour_type] || {
              label: item.behaviour_type,
              classStyle: 'badge-instruction',
            };

            return (
              <div
                key={item.id || idx}
                onClick={() => onSelectTimestamp && onSelectTimestamp(item.timestamp_seconds)}
                className="group flex items-center justify-between rounded-xl border border-slate-800/80 bg-slate-900/50 p-3 hover:border-indigo-500/40 hover:bg-slate-800/60 transition-all cursor-pointer"
              >
                <div className="flex items-center gap-3">
                  <div className="flex items-center gap-1.5 font-mono text-xs text-slate-400 min-w-[4rem]">
                    <Clock className="h-3.5 w-3.5 text-slate-500" />
                    <span>{formatTimestamp(item.timestamp_seconds)}</span>
                  </div>

                  <span className={`badge ${badge.classStyle}`}>
                    {badge.label}
                  </span>
                </div>

                <div className="flex items-center gap-4">
                  {/* Probability confidence bar */}
                  <div className="flex items-center gap-2">
                    <div className="h-1.5 w-16 rounded-full bg-slate-800 overflow-hidden">
                      <div
                        className="h-full bg-indigo-500 rounded-full"
                        style={{ width: `${Math.round(item.confidence * 100)}%` }}
                      />
                    </div>
                    <span className="font-mono text-xs font-semibold text-slate-300">
                      {Math.round(item.confidence * 100)}%
                    </span>
                  </div>

                  <ChevronRight className="h-4 w-4 text-slate-600 group-hover:text-indigo-400 transition-colors" />
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
