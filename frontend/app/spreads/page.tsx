'use client';

import { useState, useEffect } from 'react';
import axios from 'axios';
import SpreadsTable from '@/components/SpreadsTable';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function SpreadsPage() {
  const [currentSeason, setCurrentSeason] = useState({ year: 2025, week: 1 });
  const [spreads, setSpreads] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      const seasonRes = await axios.get(`${API_URL}/api/current-season`);
      setCurrentSeason(seasonRes.data);

      const spreadsRes = await axios.get(
        `${API_URL}/api/spreads/${seasonRes.data.year}/${seasonRes.data.week}`
      );
      setSpreads(spreadsRes.data.data);

      setLoading(false);
    } catch (error) {
      console.error('Error fetching data:', error);
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-gray-900 dark:to-gray-800 pt-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2">Week {currentSeason.week} Games</h1>
          <p className="text-lg text-gray-600 dark:text-gray-400">
            PEAR Projections
          </p>
        </div>

        {/* Spreads Table */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg overflow-hidden">
          <div className="p-6">
            {loading ? (
              <div className="text-center py-12">
                <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-green-600 dark:border-green-400"></div>
                <p className="mt-4 text-gray-600 dark:text-gray-400">Loading spreads...</p>
              </div>
            ) : spreads.length > 0 ? (
              <>
                <SpreadsTable data={spreads} year={currentSeason.year} week={currentSeason.week} />
              </>
            ) : (
              <div className="text-center py-12 text-gray-600 dark:text-gray-400">
                <p className="text-lg">No spreads data available for this week.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}