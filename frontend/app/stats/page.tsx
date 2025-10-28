'use client';

import { useState, useEffect } from 'react';
import axios from 'axios';
import StatsTable from '@/components/StatsTable';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function StatsPage() {
  const [currentSeason, setCurrentSeason] = useState({ year: 2025, week: 1 });
  const [stats, setStats] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      const seasonRes = await axios.get(`${API_URL}/api/current-season`);
      setCurrentSeason(seasonRes.data);

      const statsRes = await axios.get(
        `${API_URL}/api/team-stats/${seasonRes.data.year}/${seasonRes.data.week}`
      );
      setStats(statsRes.data.data);

      setLoading(false);
    } catch (error) {
      console.error('Error fetching data:', error);
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-gray-900 dark:to-gray-800">
      <div className="max-w-[1600px] mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2">Team Statistics</h1>
          <p className="text-lg text-gray-600 dark:text-gray-400">
            {currentSeason.year} Season • Week {currentSeason.week}
          </p>
        </div>

        {/* Stats Table */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg overflow-hidden">
          <div className="p-6">
            {loading ? (
              <div className="text-center py-12">
                <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 dark:border-blue-400"></div>
                <p className="mt-4 text-gray-600 dark:text-gray-400">Loading statistics...</p>
              </div>
            ) : (
              <>
                <StatsTable data={stats} />
                <div className="mt-6 text-sm text-gray-600 dark:text-gray-400 space-y-1 border-t dark:border-gray-700 pt-4">
                  <p><strong className="dark:text-gray-300">Success Rate</strong> - Percentage of successful plays | <strong className="dark:text-gray-300">PPA</strong> - Predicted Points Added per play</p>
                  <p><strong className="dark:text-gray-300">Rush/Pass</strong> - Rushing and Passing efficiency metrics | <strong className="dark:text-gray-300">PPO</strong> - Points Per Opportunity | <strong className="dark:text-gray-300">Drive Q</strong> - Drive Quality rating</p>
                  <p className="mt-3 font-semibold dark:text-gray-300">All stats are opponent-adjusted</p>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}