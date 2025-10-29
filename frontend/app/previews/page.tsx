'use client';

import { useState, useEffect } from 'react';
import axios from 'axios';
import ImageGallery from '@/components/ImageGallery';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function PreviewsPage() {
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
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2">Game Previews & Team Profiles</h1>
          <p className="text-lg text-gray-600 dark:text-gray-400">
            Week {currentSeason.week} • {currentSeason.year} Season
          </p>
        </div>

        {/* Image Galleries */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <ImageGallery
            title="Game Previews"
            type="games"
            year={currentSeason.year}
            week={currentSeason.week}
          />
          <ImageGallery
            title="Stat Profiles"
            type="profiles"
            year={currentSeason.year}
            week={currentSeason.week}
          />
        </div>
      </div>
    </div>
  );
}