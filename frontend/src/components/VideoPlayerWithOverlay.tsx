import React, { useEffect, useRef, useState } from 'react';
import { Play, Pause, RotateCcw, Volume2, VolumeX, Eye } from 'lucide-react';
import { BehaviourResult, StudentTrackResult } from '../types';

interface VideoPlayerWithOverlayProps {
  videoUrl: string;
  tracks: StudentTrackResult[];
  behaviours: BehaviourResult[];
  selectedTrackId?: number | null;
}

const BEHAVIOUR_COLORS: Record<string, string> = {
  Looking_Toward_Instruction: '#38bdf8', // Sky blue
  Reading: '#34d399',                   // Emerald
  Writing: '#818cf8',                   // Indigo
  Peer_Interaction: '#fbbf24',          // Amber
  Looking_Away: '#fb7185',              // Rose
};

export const VideoPlayerWithOverlay: React.FC<VideoPlayerWithOverlayProps> = ({
  videoUrl,
  tracks,
  behaviours,
  selectedTrackId = null,
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isMuted, setIsMuted] = useState(true);
  const [showOverlays, setShowOverlays] = useState(true);
  const [activeFilterTrack, setActiveFilterTrack] = useState<number | null>(selectedTrackId);

  useEffect(() => {
    setActiveFilterTrack(selectedTrackId);
  }, [selectedTrackId]);

  // Synchronize canvas rendering loop with video playback
  useEffect(() => {
    let animationFrameId: number;

    const renderOverlay = () => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas) return;

      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      // Match canvas internal resolution to video display resolution
      if (canvas.width !== video.clientWidth || canvas.height !== video.clientHeight) {
        canvas.width = video.clientWidth;
        canvas.height = video.clientHeight;
      }

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      if (showOverlays && video.videoWidth > 0 && video.videoHeight > 0) {
        const curTime = video.currentTime;
        const scaleX = canvas.width / video.videoWidth;
        const scaleY = canvas.height / video.videoHeight;

        // Render each track's active bounding box
        tracks.forEach((track) => {
          if (activeFilterTrack !== null && track.track_id !== activeFilterTrack) {
            return;
          }

          // Find the bounding box closest to curTime
          let closestPoint = null;
          let minDiff = 0.6; // tolerance in seconds

          for (const pt of track.bounding_box_history) {
            const diff = Math.abs(pt.timestamp - curTime);
            if (diff < minDiff) {
              minDiff = diff;
              closestPoint = pt;
            }
          }

          if (closestPoint && closestPoint.bbox) {
            const [x1, y1, x2, y2] = closestPoint.bbox;
            const drawX = x1 * scaleX;
            const drawY = y1 * scaleY;
            const drawW = (x2 - x1) * scaleX;
            const drawH = (y2 - y1) * scaleY;

            // Find current predicted behaviour for this track
            const currentPred = behaviours.find(
              (b) => b.track_id === track.track_id && Math.abs(b.timestamp_seconds - curTime) < 1.5
            );

            const bType = currentPred?.behaviour_type || 'Looking_Toward_Instruction';
            const color = BEHAVIOUR_COLORS[bType] || '#38bdf8';
            const conf = currentPred ? Math.round(currentPred.confidence * 100) : 85;
            const label = `Student ${track.track_id.toString().padStart(2, '0')}`;

            // Draw Box Border with glowing shadow
            ctx.save();
            ctx.shadowColor = color;
            ctx.shadowBlur = 8;
            ctx.strokeStyle = color;
            ctx.lineWidth = 2.5;
            ctx.strokeRect(drawX, drawY, drawW, drawH);
            ctx.restore();

            // Draw Top Track Tag
            ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
            ctx.fillRect(drawX, Math.max(0, drawY - 24), Math.max(drawW, 110), 22);

            ctx.fillStyle = color;
            ctx.font = 'bold 11px Inter, sans-serif';
            ctx.fillText(`${label} • ${conf}%`, drawX + 6, Math.max(14, drawY - 8));

            // Draw Bottom Behavior Tag
            const bLabel = bType.replace(/_/g, ' ');
            const tagY = Math.min(canvas.height - 6, drawY + drawH + 18);
            ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
            ctx.fillRect(drawX, tagY - 14, Math.max(drawW, 120), 18);

            ctx.fillStyle = '#ffffff';
            ctx.font = '500 10px Inter, sans-serif';
            ctx.fillText(bLabel, drawX + 6, tagY);
          }
        });
      }

      animationFrameId = requestAnimationFrame(renderOverlay);
    };

    animationFrameId = requestAnimationFrame(renderOverlay);
    return () => cancelAnimationFrame(animationFrameId);
  }, [tracks, behaviours, showOverlays, activeFilterTrack]);

  const togglePlay = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const time = parseFloat(e.target.value);
    if (videoRef.current) {
      videoRef.current.currentTime = time;
      setCurrentTime(time);
    }
  };

  const formatTime = (secs: number) => {
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  return (
    <div className="glass-panel overflow-hidden border border-slate-800">
      {/* Video Container with Overlaid Canvas */}
      <div ref={containerRef} className="relative aspect-video w-full bg-black flex items-center justify-center">
        <video
          ref={videoRef}
          src={videoUrl}
          className="h-full w-full object-contain"
          playsInline
          muted={isMuted}
          onTimeUpdate={() => {
            if (videoRef.current) {
              setCurrentTime(videoRef.current.currentTime);
            }
          }}
          onLoadedMetadata={() => {
            if (videoRef.current) {
              setDuration(videoRef.current.duration || 0);
            }
          }}
          onEnded={() => setIsPlaying(false)}
        />
        <canvas
          ref={canvasRef}
          className="pointer-events-none absolute inset-0 h-full w-full"
        />

        {/* Play Overlay Button if paused */}
        {!isPlaying && (
          <button
            onClick={togglePlay}
            className="absolute inset-0 flex items-center justify-center bg-black/40 hover:bg-black/30 transition-colors"
          >
            <div className="flex h-16 w-16 items-center justify-center rounded-full bg-indigo-600/90 text-white shadow-xl shadow-indigo-600/40 backdrop-blur-sm hover:scale-105 transition-transform">
              <Play className="h-7 w-7 ml-1" />
            </div>
          </button>
        )}
      </div>

      {/* Control Bar */}
      <div className="space-y-2 p-4 bg-slate-950/90 border-t border-slate-800/80">
        {/* Scrubber Progress Bar */}
        <div className="flex items-center gap-3">
          <span className="text-xs font-mono text-slate-400 min-w-[3rem]">
            {formatTime(currentTime)}
          </span>
          <input
            type="range"
            min="0"
            max={duration || 100}
            step="0.1"
            value={currentTime}
            onChange={handleSeek}
            className="h-1.5 w-full cursor-pointer appearance-none rounded-lg bg-slate-800 accent-indigo-500"
          />
          <span className="text-xs font-mono text-slate-400 min-w-[3rem] text-right">
            {formatTime(duration)}
          </span>
        </div>

        {/* Buttons & Track Filters */}
        <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
          <div className="flex items-center gap-2">
            <button
              onClick={togglePlay}
              className="flex h-8 w-8 items-center justify-center rounded-lg bg-slate-800 text-slate-200 hover:bg-slate-700"
            >
              {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4 ml-0.5" />}
            </button>
            <button
              onClick={() => {
                if (videoRef.current) {
                  videoRef.current.currentTime = 0;
                  setCurrentTime(0);
                }
              }}
              className="flex h-8 w-8 items-center justify-center rounded-lg bg-slate-800 text-slate-200 hover:bg-slate-700"
            >
              <RotateCcw className="h-4 w-4" />
            </button>
            <button
              onClick={() => setIsMuted(!isMuted)}
              className="flex h-8 w-8 items-center justify-center rounded-lg bg-slate-800 text-slate-200 hover:bg-slate-700"
            >
              {isMuted ? <VolumeX className="h-4 w-4" /> : <Volume2 className="h-4 w-4" />}
            </button>
          </div>

          <div className="flex items-center gap-3">
            {/* Filter by track */}
            <div className="flex items-center gap-2 text-xs">
              <span className="text-slate-400">Track:</span>
              <select
                value={activeFilterTrack ?? ''}
                onChange={(e) => setActiveFilterTrack(e.target.value ? parseInt(e.target.value) : null)}
                className="rounded-lg bg-slate-800 border border-slate-700 px-2 py-1 text-xs text-slate-200 outline-none"
              >
                <option value="">All Student Tracks</option>
                {tracks.map((t) => (
                  <option key={t.track_id} value={t.track_id}>
                    Student {t.track_id.toString().padStart(2, '0')}
                  </option>
                ))}
              </select>
            </div>

            {/* Toggle Overlay */}
            <button
              onClick={() => setShowOverlays(!showOverlays)}
              className={`flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-xs font-medium transition-colors ${
                showOverlays
                  ? 'bg-indigo-600/30 text-indigo-300 border border-indigo-500/40'
                  : 'bg-slate-800 text-slate-400'
              }`}
            >
              <Eye className="h-3.5 w-3.5" />
              <span>{showOverlays ? 'Overlays On' : 'Overlays Off'}</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
