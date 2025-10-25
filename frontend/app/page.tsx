'use client';

import { useState, useEffect } from 'react';
import axios from 'axios';
import RatingsTable from '@/components/RatingsTable';
import SpreadsTable from '@/components/SpreadsTable';
import SpreadCalculator from '@/components/SpreadCalculator';
import ImageGallery from '@/components/ImageGallery';
import { Trophy, TrendingUp, Calculator, History } from 'lucide-react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function Home() {
  const [currentSeason, setCurrentSeason] = useState({ year: 2025, week: 1 });
  const [ratings, setRatings] = useState([]);
  const [spreads, setSpreads] = useState([]);
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
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-blue-900 via-blue-800 to-blue-900 text-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
          <div className="text-center">
            <div className="flex justify-center mb-6">
              <Trophy className="w-20 h-20 text-yellow-400" />
            </div>
            <h1 className="text-5xl md:text-6xl font-bold mb-4 tracking-tight">
              {currentSeason.year} CFB PEAR
            </h1>
            <p className="text-lg text-blue-300 mt-4">
              Week {currentSeason.week} • College Football Analytics
            </p>
          </div>
        </div>
      </div>

      {/* Navigation Cards */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 -mt-8 mb-12">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <a href="#ratings" className="bg-white rounded-lg shadow-lg p-6 hover:shadow-xl transition-shadow">
            <div className="flex items-center space-x-4">
              <div className="bg-blue-100 rounded-full p-3">
                <TrendingUp className="w-6 h-6 text-blue-600" />
              </div>
              <div>
                <h3 className="font-bold text-lg text-gray-900">Power Ratings</h3>
                <p className="text-gray-600 text-sm">View FBS rankings</p>
              </div>
            </div>
          </a>
          
          <a href="#calculator" className="bg-white rounded-lg shadow-lg p-6 hover:shadow-xl transition-shadow">
            <div className="flex items-center space-x-4">
              <div className="bg-green-100 rounded-full p-3">
                <Calculator className="w-6 h-6 text-green-600" />
              </div>
              <div>
                <h3 className="font-bold text-lg text-gray-900">Spread Calculator</h3>
                <p className="text-gray-600 text-sm">Predict any matchup</p>
              </div>
            </div>
          </a>
          
          <a href="#images" className="bg-white rounded-lg shadow-lg p-6 hover:shadow-xl transition-shadow">
            <div className="flex items-center space-x-4">
              <div className="bg-purple-100 rounded-full p-3">
                <History className="w-6 h-6 text-purple-600" />
              </div>
              <div>
                <h3 className="font-bold text-lg text-gray-900">Game Previews</h3>
                <p className="text-gray-600 text-sm">View matchup graphics</p>
              </div>
            </div>
          </a>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pb-16">
        {/* Power Ratings Section */}
        <section id="ratings" className="mb-16">
          <div className="bg-white rounded-xl shadow-lg overflow-hidden">
            <div className="bg-gradient-to-r from-blue-600 to-blue-700 px-6 py-4">
              <h2 className="text-2xl font-bold text-white">FBS Power Ratings</h2>
              <p className="text-blue-100 text-sm mt-1">
                Week {currentSeason.week} • {currentSeason.year} Season
              </p>
            </div>
            <div className="p-6">
              {loading ? (
                <div className="text-center py-12">
                  <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
                  <p className="mt-4 text-gray-600">Loading ratings...</p>
                </div>
              ) : (
                <>
                  <RatingsTable data={ratings} />
                  <div className="mt-4 text-sm text-gray-600 space-y-1">
                    <p><strong>MD</strong> - Most Deserving (PEAR's 'AP' Ballot)</p>
                    <p><strong>SOS</strong> - Strength of Schedule | <strong>SOR</strong> - Strength of Record</p>
                    <p><strong>OFF</strong> - Offense | <strong>DEF</strong> - Defense</p>
                    <p><strong>PBR</strong> - Penalty Burden Ratio | <strong>DCE</strong> - Drive Control Efficiency | <strong>DDE</strong> - Drive Disruption Efficiency</p>
                  </div>
                </>
              )}
            </div>
          </div>
        </section>

        {/* Spreads and Calculator Section */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-16">
          {/* Weekly Spreads */}
          <section>
            <div className="bg-white rounded-xl shadow-lg overflow-hidden h-full">
              <div className="bg-gradient-to-r from-green-600 to-green-700 px-6 py-4">
                <h2 className="text-2xl font-bold text-white">Week {currentSeason.week} Spreads</h2>
                <p className="text-green-100 text-sm mt-1">PEAR vs Vegas Lines</p>
              </div>
              <div className="p-6">
                {loading ? (
                  <div className="text-center py-12">
                    <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-green-600"></div>
                  </div>
                ) : (
                  <SpreadsTable data={spreads} />
                )}
              </div>
            </div>
          </section>

          {/* Spread Calculator */}
          <section id="calculator">
            <div className="bg-white rounded-xl shadow-lg overflow-hidden h-full">
              <div className="bg-gradient-to-r from-purple-600 to-purple-700 px-6 py-4">
                <h2 className="text-2xl font-bold text-white">Spread Calculator</h2>
                <p className="text-purple-100 text-sm mt-1">Predict any matchup</p>
              </div>
              <div className="p-6">
                <SpreadCalculator />
              </div>
            </div>
          </section>
        </div>

        {/* Image Galleries */}
        <section id="images" className="mb-16">
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
        </section>

        {/* Historical Archive Link */}
        <section id="archive" className="text-center">
          <div className="bg-gradient-to-r from-slate-700 to-slate-800 rounded-xl shadow-lg p-12">
            <History className="w-16 h-16 text-slate-300 mx-auto mb-4" />
            <h2 className="text-3xl font-bold text-white mb-4">Historical Archive</h2>
            <p className="text-slate-300 mb-6 max-w-2xl mx-auto">
              Explore ratings and data from 2014-2024. View normalized ratings across all seasons
              and analyze historical performance trends.
            </p>
            <a
              href="/archive"
              className="inline-block bg-white text-slate-800 px-8 py-3 rounded-lg font-semibold hover:bg-slate-100 transition-colors"
            >
              View Historical Data
            </a>
          </div>
        </section>
      </div>

      {/* Footer */}
      <footer className="bg-slate-900 text-white py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <p className="text-slate-400">© {currentSeason.year} PEAR Ratings. College Football Analytics.</p>
          <p className="text-slate-500 text-sm mt-2">
            <a href="https://x.com/PEARatings" target="_blank" rel="noopener noreferrer" className="hover:text-slate-300 transition-colors">
              @PEARatings
            </a>
          </p>
        </div>
      </footer>
    </div>
  );
}