'use client';

import { useState, useEffect } from 'react';
import axios from 'axios';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function StatsPage() {
  const [currentSeason, setCurrentSeason] = useState({ year: 2025, week: 1 });

  useEffect(() => {
    fetchSeasonData();
  }, []);

  const fetchSeasonData = async () => {
    try {
      const seasonRes = await axios.get(`${API_URL}/api/current-season`);
      setCurrentSeason(seasonRes.data);
    } catch (error) {
      console.error('Error fetching season data:', error);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">Team Statistics</h1>
          <p className="text-lg text-gray-600">
            {currentSeason.year} Season • Week {currentSeason.week}
          </p>
        </div>

        {/* Placeholder */}
        <div className="bg-white rounded-xl shadow-lg overflow-hidden">
          <div className="p-12 text-center">
            <div className="text-6xl mb-4">📊</div>
            <h2 className="text-2xl font-bold text-gray-900 mb-2">Stats Page Coming Soon</h2>
            <p className="text-gray-600">
              This page will display detailed team statistics and analytics.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}