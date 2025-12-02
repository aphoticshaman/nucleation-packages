'use client';

import { useState } from 'react';

export default function ConsumerSettingsPage() {
  const [saving, setSaving] = useState(false);

  const handleSave = async () => {
    setSaving(true);
    await new Promise((resolve) => setTimeout(resolve, 1000));
    setSaving(false);
  };

  return (
    <div className="max-w-2xl space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-white">Settings</h1>
        <p className="text-slate-400 mt-1">Manage your account and preferences</p>
      </div>

      {/* Profile */}
      <div className="bg-slate-900 rounded-xl border border-slate-800 p-6">
        <h2 className="text-lg font-medium text-white mb-6">Profile</h2>

        <div className="space-y-4">
          <div>
            <label className="block text-sm text-slate-400 mb-2">Display Name</label>
            <input
              type="text"
              className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-blue-500"
              placeholder="Your name"
            />
          </div>

          <div>
            <label className="block text-sm text-slate-400 mb-2">Email</label>
            <input
              type="email"
              className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-slate-400"
              disabled
              value="user@example.com"
            />
            <p className="text-xs text-slate-500 mt-1">Contact support to change email</p>
          </div>
        </div>
      </div>

      {/* Preferences */}
      <div className="bg-slate-900 rounded-xl border border-slate-800 p-6">
        <h2 className="text-lg font-medium text-white mb-6">Preferences</h2>

        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-white">Default Map Layer</p>
              <p className="text-sm text-slate-400">Shown when you open the app</p>
            </div>
            <select className="px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white">
              <option value="basin">Stability</option>
              <option value="risk">Risk</option>
              <option value="regime">Regimes</option>
            </select>
          </div>

          <div className="flex items-center justify-between">
            <div>
              <p className="text-white">Auto-save simulations</p>
              <p className="text-sm text-slate-400">Save after each run</p>
            </div>
            <button className="w-12 h-6 bg-slate-700 rounded-full relative" onClick={() => {}}>
              <span className="absolute left-1 top-1 w-4 h-4 bg-slate-400 rounded-full transition-transform" />
            </button>
          </div>

          <div className="flex items-center justify-between">
            <div>
              <p className="text-white">Email notifications</p>
              <p className="text-sm text-slate-400">Updates and announcements</p>
            </div>
            <button className="w-12 h-6 bg-blue-600 rounded-full relative" onClick={() => {}}>
              <span className="absolute right-1 top-1 w-4 h-4 bg-white rounded-full transition-transform" />
            </button>
          </div>
        </div>
      </div>

      {/* Usage */}
      <div className="bg-slate-900 rounded-xl border border-slate-800 p-6">
        <h2 className="text-lg font-medium text-white mb-6">Usage</h2>

        <div className="space-y-4">
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-slate-400">Daily simulations</span>
              <span className="text-white">7 / 10</span>
            </div>
            <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
              <div className="h-full w-[70%] bg-blue-500 rounded-full" />
            </div>
            <p className="text-xs text-slate-500 mt-1">Resets at midnight UTC</p>
          </div>

          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-slate-400">Saved simulations</span>
              <span className="text-white">3 / 5</span>
            </div>
            <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
              <div className="h-full w-[60%] bg-blue-500 rounded-full" />
            </div>
          </div>
        </div>
      </div>

      {/* Plan */}
      <div className="bg-gradient-to-r from-slate-900 to-slate-800 rounded-xl border border-slate-700 p-6">
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center gap-3">
              <h2 className="text-lg font-medium text-white">Free Plan</h2>
              <span className="px-2 py-0.5 bg-slate-700 rounded text-xs text-slate-300">
                Current
              </span>
            </div>
            <p className="text-slate-400 mt-1">10 simulations/day, 5 save slots</p>
          </div>
          <a
            href="/pricing"
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-500"
          >
            Upgrade
          </a>
        </div>
      </div>

      {/* Danger zone */}
      <div className="bg-slate-900 rounded-xl border border-red-900/50 p-6">
        <h2 className="text-lg font-medium text-red-400 mb-4">Danger Zone</h2>

        <div className="flex items-center justify-between">
          <div>
            <p className="text-white">Delete account</p>
            <p className="text-sm text-slate-400">Permanently delete your account and all data</p>
          </div>
          <button className="px-4 py-2 border border-red-600 text-red-400 rounded-lg hover:bg-red-900/20">
            Delete
          </button>
        </div>
      </div>

      {/* Save button */}
      <div className="flex justify-end">
        <button
          onClick={() => void handleSave()}
          disabled={saving}
          className={`px-6 py-2 rounded-lg font-medium ${
            saving ? 'bg-slate-700 text-slate-400' : 'bg-blue-600 text-white hover:bg-blue-500'
          }`}
        >
          {saving ? 'Saving...' : 'Save Changes'}
        </button>
      </div>
    </div>
  );
}
