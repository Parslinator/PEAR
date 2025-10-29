'use client';

import { useState, useEffect } from 'react';
import axios from 'axios';
import SpreadCalculator from '@/components/SpreadCalculator';

const API_URL = process.env.NEXT_PUBLIC_API_URL || '';

export default function MatchupsPage() {
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
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-gray-900 dark:to-gray-800 pt-16">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2">Matchup Predictor</h1>
          <p className="text-lg text-gray-600 dark:text-gray-400">
            Generate detailed matchup previews for any two teams
          </p>
        </div>

        {/* Calculator */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg overflow-hidden">
          <div className="p-6">
            <SpreadCalculator />
          </div>
        </div>

        {/* Info Box */}
        <div className="mt-8 bg-blue-50 dark:bg-blue-900/30 border border-blue-200 dark:border-blue-800 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-blue-900 dark:text-blue-200 mb-2">How It Works</h3>
          <ul className="text-sm text-blue-800 dark:text-blue-300 space-y-2">
            <li>• Select an away team and a home team from the dropdowns</li>
            <li>• Check "Neutral Site Game" if applicable</li>
            <li>• Click "Calculate Spread" to generate a detailed matchup preview image</li>
            <li>• The preview includes win probabilities, projected scores, and detailed stat breakdowns</li>
            <li>• Download the image to save or share your prediction</li>
          </ul>
        </div>
      </div>
    </div>
  );
}