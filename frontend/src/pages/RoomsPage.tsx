import React, { useEffect, useState } from 'react';
import { api } from '../api/client';
import { Camera, Room } from '../types';
import {
  Building,
  Camera as CameraIcon,
  CheckCircle,
  HardDrive,
  Plus,
  RefreshCw,
  Video
} from 'lucide-react';

export const RoomsPage: React.FC = () => {
  const [rooms, setRooms] = useState<Room[]>([]);
  const [storageStats, setStorageStats] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // New Room Form State
  const [showRoomModal, setShowRoomModal] = useState(false);
  const [roomNumber, setRoomNumber] = useState('');
  const [building, setBuilding] = useState('Academic Block A');
  const [floor, setFloor] = useState(1);
  const [capacity, setCapacity] = useState(60);

  // New Camera Form State
  const [activeRoomId, setActiveRoomId] = useState<string | null>(null);
  const [cameraName, setCameraName] = useState('');
  const [deviceCode, setDeviceCode] = useState('');
  const [locationInRoom, setLocationInRoom] = useState('FRONT');

  const [saving, setSaving] = useState(false);

  const fetchData = async () => {
    try {
      setLoading(true);
      const [roomsData, stats] = await Promise.all([
        api.getRooms(),
        api.getStorageStats().catch(() => null)
      ]);
      setRooms(roomsData);
      setStorageStats(stats);
      setError(null);
    } catch (err: any) {
      setError(err.message || 'Failed to load campus infrastructure');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const handleCreateRoom = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!roomNumber.trim()) return;
    try {
      setSaving(true);
      await api.createRoom({
        room_number: roomNumber.trim(),
        building,
        floor: Number(floor),
        capacity: Number(capacity),
      });
      setShowRoomModal(false);
      setRoomNumber('');
      fetchData();
    } catch (err: any) {
      alert(err.message || 'Failed to create room');
    } finally {
      setSaving(false);
    }
  };

  const handleAddCamera = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!activeRoomId || !deviceCode.trim()) return;
    try {
      setSaving(true);
      await api.addCameraToRoom(activeRoomId, {
        camera_name: cameraName.trim() || `Cam-${deviceCode.trim()}`,
        device_code: deviceCode.trim(),
        location_in_room: locationInRoom,
      });
      setActiveRoomId(null);
      setCameraName('');
      setDeviceCode('');
      fetchData();
    } catch (err: any) {
      alert(err.message || 'Failed to register camera');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Top Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight flex items-center gap-2">
            <Building className="w-6 h-6 text-indigo-400" />
            Classroom Infrastructure & Cameras
          </h1>
          <p className="text-sm text-slate-400">
            Manage physical campus lecture halls, existing NVR channels, and recording sources.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={fetchData}
            className="px-3 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-sm flex items-center gap-2 border border-slate-700 transition"
          >
            <RefreshCw className="w-4 h-4" />
            Refresh
          </button>
          <button
            onClick={() => setShowRoomModal(true)}
            className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-sm font-medium flex items-center gap-2 shadow-lg shadow-indigo-500/20 transition"
          >
            <Plus className="w-4 h-4" />
            Add Classroom Room
          </button>
        </div>
      </div>

      {/* Storage & Integration Health Bar */}
      {storageStats && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-slate-900/60 backdrop-blur-md p-4 rounded-xl border border-slate-800">
            <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
              <span>Raw Video Storage</span>
              <HardDrive className="w-4 h-4 text-indigo-400" />
            </div>
            <div className="text-xl font-bold text-white">
              {(storageStats.raw_videos_bytes / (1024 * 1024)).toFixed(1)} MB
            </div>
            <div className="text-xs text-slate-500 mt-1">Retention: {storageStats.retention_days} days</div>
          </div>

          <div className="bg-slate-900/60 backdrop-blur-md p-4 rounded-xl border border-slate-800">
            <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
              <span>Ingestion Dropzone</span>
              <Video className="w-4 h-4 text-emerald-400" />
            </div>
            <div className="text-xl font-bold text-emerald-400">
              {(storageStats.dropzone_bytes / (1024 * 1024)).toFixed(1)} MB
            </div>
            <div className="text-xs text-slate-500 mt-1">NFS/SFTP watch drop</div>
          </div>

          <div className="bg-slate-900/60 backdrop-blur-md p-4 rounded-xl border border-slate-800">
            <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
              <span>Classroom Rooms</span>
              <Building className="w-4 h-4 text-cyan-400" />
            </div>
            <div className="text-xl font-bold text-cyan-400">{rooms.length} Active</div>
            <div className="text-xs text-slate-500 mt-1">Configured for matching</div>
          </div>

          <div className="bg-slate-900/60 backdrop-blur-md p-4 rounded-xl border border-slate-800">
            <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
              <span>Registered Cameras</span>
              <CameraIcon className="w-4 h-4 text-purple-400" />
            </div>
            <div className="text-xl font-bold text-purple-400">
              {rooms.reduce((acc, r) => acc + (r.cameras?.length || 0), 0)} Connected
            </div>
            <div className="text-xs text-slate-500 mt-1">NVR channels active</div>
          </div>
        </div>
      )}

      {/* Rooms Grid */}
      {loading ? (
        <div className="flex items-center justify-center p-12 text-slate-400">
          <RefreshCw className="w-6 h-6 animate-spin mr-2" /> Loading campus rooms...
        </div>
      ) : error ? (
        <div className="p-4 bg-rose-500/10 border border-rose-500/30 text-rose-300 rounded-xl">
          {error}
        </div>
      ) : rooms.length === 0 ? (
        <div className="p-12 text-center bg-slate-900/40 rounded-2xl border border-slate-800">
          <Building className="w-12 h-12 text-slate-600 mx-auto mb-3" />
          <h3 className="text-lg font-semibold text-slate-300">No Classroom Rooms Configured</h3>
          <p className="text-sm text-slate-500 max-w-md mx-auto mt-1 mb-4">
            Register lecture halls and classrooms to enable automatic NVR recording matching and schedule association.
          </p>
          <button
            onClick={() => setShowRoomModal(true)}
            className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-sm font-medium"
          >
            Add First Classroom
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {rooms.map((room) => (
            <div
              key={room.id}
              className="bg-slate-900/60 backdrop-blur-md rounded-2xl border border-slate-800 p-5 hover:border-slate-700 transition space-y-4"
            >
              <div className="flex items-start justify-between">
                <div>
                  <h3 className="text-lg font-bold text-white flex items-center gap-2">
                    <span className="px-2.5 py-1 bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 rounded-lg text-sm">
                      {room.room_number}
                    </span>
                  </h3>
                  <p className="text-xs text-slate-400 mt-1">
                    {room.building} &bull; Floor {room.floor} &bull; Capacity {room.capacity}
                  </p>
                </div>
                <span className="flex items-center gap-1 text-xs text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded-full border border-emerald-500/20">
                  <CheckCircle className="w-3 h-3" /> Active
                </span>
              </div>

              {/* Cameras List */}
              <div className="space-y-2 pt-2 border-t border-slate-800">
                <div className="flex items-center justify-between text-xs font-semibold text-slate-400 uppercase tracking-wider">
                  <span>Connected Cameras ({room.cameras?.length || 0})</span>
                  <button
                    onClick={() => setActiveRoomId(room.id)}
                    className="text-indigo-400 hover:text-indigo-300 text-xs font-medium flex items-center gap-1"
                  >
                    <Plus className="w-3 h-3" /> Add Camera
                  </button>
                </div>

                {room.cameras && room.cameras.length > 0 ? (
                  <div className="space-y-1.5">
                    {room.cameras.map((cam) => (
                      <div
                        key={cam.id}
                        className="flex items-center justify-between p-2 bg-slate-950/40 rounded-lg border border-slate-800/80 text-xs"
                      >
                        <div className="flex items-center gap-2">
                          <CameraIcon className="w-3.5 h-3.5 text-indigo-400" />
                          <div>
                            <span className="text-slate-200 font-medium">{cam.camera_name}</span>
                            <span className="text-slate-500 ml-1.5">({cam.device_code})</span>
                          </div>
                        </div>
                        <span className="px-1.5 py-0.5 bg-slate-800 text-slate-400 rounded text-[10px]">
                          {cam.location_in_room}
                        </span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-xs text-slate-500 italic py-2">
                    No camera streams mapped. Recording matching will use Room Number.
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Add Room Modal */}
      {showRoomModal && (
        <div className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-md w-full p-6 space-y-4 shadow-2xl">
            <h3 className="text-lg font-bold text-white">Add Classroom Room</h3>
            <form onSubmit={handleCreateRoom} className="space-y-4">
              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  Room Number / Code <span className="text-rose-400">*</span>
                </label>
                <input
                  type="text"
                  required
                  placeholder="e.g. 301, CS-LAB-02, LH-04"
                  value={roomNumber}
                  onChange={(e) => setRoomNumber(e.target.value)}
                  className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-lg text-white text-sm focus:outline-none focus:border-indigo-500"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">Building</label>
                <input
                  type="text"
                  value={building}
                  onChange={(e) => setBuilding(e.target.value)}
                  className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-lg text-white text-sm focus:outline-none focus:border-indigo-500"
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1">Floor</label>
                  <input
                    type="number"
                    min="0"
                    value={floor}
                    onChange={(e) => setFloor(Number(e.target.value))}
                    className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-lg text-white text-sm focus:outline-none focus:border-indigo-500"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1">Capacity</label>
                  <input
                    type="number"
                    min="1"
                    value={capacity}
                    onChange={(e) => setCapacity(Number(e.target.value))}
                    className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-lg text-white text-sm focus:outline-none focus:border-indigo-500"
                  />
                </div>
              </div>

              <div className="flex items-center justify-end gap-3 pt-4 border-t border-slate-800">
                <button
                  type="button"
                  onClick={() => setShowRoomModal(false)}
                  className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-sm"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={saving}
                  className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-sm font-medium"
                >
                  {saving ? 'Creating...' : 'Create Room'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Add Camera Modal */}
      {activeRoomId && (
        <div className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-md w-full p-6 space-y-4 shadow-2xl">
            <h3 className="text-lg font-bold text-white">Register Camera / NVR Channel</h3>
            <form onSubmit={handleAddCamera} className="space-y-4">
              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  Device Code / Channel <span className="text-rose-400">*</span>
                </label>
                <input
                  type="text"
                  required
                  placeholder="e.g. CAM-301-FRONT, NVR-CH-04"
                  value={deviceCode}
                  onChange={(e) => setDeviceCode(e.target.value)}
                  className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-lg text-white text-sm focus:outline-none focus:border-indigo-500"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">Camera Name</label>
                <input
                  type="text"
                  placeholder="e.g. Front Podium Camera"
                  value={cameraName}
                  onChange={(e) => setCameraName(e.target.value)}
                  className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-lg text-white text-sm focus:outline-none focus:border-indigo-500"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">Location in Room</label>
                <select
                  value={locationInRoom}
                  onChange={(e) => setLocationInRoom(e.target.value)}
                  className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-lg text-white text-sm focus:outline-none focus:border-indigo-500"
                >
                  <option value="FRONT">FRONT (Facing Students)</option>
                  <option value="REAR">REAR (Facing Board)</option>
                  <option value="CEILING">CEILING (Overview)</option>
                </select>
              </div>

              <div className="flex items-center justify-end gap-3 pt-4 border-t border-slate-800">
                <button
                  type="button"
                  onClick={() => setActiveRoomId(null)}
                  className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-sm"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={saving}
                  className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-sm font-medium"
                >
                  {saving ? 'Adding...' : 'Register Camera'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};
