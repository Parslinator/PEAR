'use client';

import { useState, useEffect } from 'react';
import axios from 'axios';
import RatingsTable from '@/components/RatingsTable';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function Home() {
  const [currentSeason, setCurrentSeason] = useState({ year: 2025, week: 1 });
  const [ratings, setRatings] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      const seasonRes = await axios.get(`${API_URL}/api/current-season`);
      setCurrentSeason(seasonRes.data);

      const ratingsRes = await axios.get(
        `${API_URL}/api/ratings/${seasonRes.data.year}/${seasonRes.data.week}`
      );
      setRatings(ratingsRes.data.data);

      setLoading(false);
    } catch (error) {
      console.error('Error fetching data:', error);
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">FBS Power Ratings</h1>
          <p className="text-lg text-gray-600">
            {currentSeason.year} Season • Week {currentSeason.week}
          </p>
        </div>

        {/* Ratings Table */}
        <div className="bg-white rounded-xl shadow-lg overflow-hidden">
          <div className="p-6">
            {loading ? (
              <div className="text-center py-12">
                <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
                <p className="mt-4 text-gray-600">Loading ratings...</p>
              </div>
            ) : (
              <>
                <RatingsTable data={ratings} />
                <div className="mt-6 text-sm text-gray-600 space-y-1 border-t pt-4">
                  <p><strong>MD</strong> - Most Deserving (PEAR's 'AP' Ballot)</p>
                  <p><strong>SOS</strong> - Strength of Schedule | <strong>SOR</strong> - Strength of Record</p>
                  <p><strong>OFF</strong> - Offense | <strong>DEF</strong> - Defense</p>
                  <p><strong>PBR</strong> - Penalty Burden Ratio | <strong>DCE</strong> - Drive Control Efficiency | <strong>DDE</strong> - Drive Disruption Efficiency</p>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}