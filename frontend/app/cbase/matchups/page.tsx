'use client';

import { useState, useEffect } from 'react';
import { Trophy, Calculator } from 'lucide-react';
import Image from 'next/image';
import axios from 'axios';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface CalculationResult {
  spread: number;
  formatted_spread: string;
  home_win_prob: number;
  away_win_prob: number;
  elo_win_prob: number;
  game_quality: number;
  home_pr: number;
  away_pr: number;
  home_elo: number;
  away_elo: number;
}

export default function CbaseMatchupsPage() {
  const [teams, setTeams] = useState<string[]>([]);
  const [awayTeam, setAwayTeam] = useState('');
  const [homeTeam, setHomeTeam] = useState('');
  const [neutral, setNeutral] = useState(false);
  const [result, setResult] = useState<CalculationResult | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  
  // Search/filter states
  const [awaySearchTerm, setAwaySearchTerm] = useState('');
  const [homeSearchTerm, setHomeSearchTerm] = useState('');
  const [showAwayDropdown, setShowAwayDropdown] = useState(false);
  const [showHomeDropdown, setShowHomeDropdown] = useState(false);
  const [fullscreenImage, setFullscreenImage] = useState<string | null>(null);

  useEffect(() => {
    fetchTeams();
  }, []);

  // Handle escape key for fullscreen modal
  useEffect(() => {
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && fullscreenImage) {
        setFullscreenImage(null);
      }
    };

    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [fullscreenImage]);

  const fetchTeams = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/cbase/teams`);
      setTeams(response.data.teams);
    } catch (error) {
      console.error('Error fetching teams:', error);
    }
  };

  const calculateSpread = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!awayTeam || !homeTeam) {
      setError('Please select both teams');
      return;
    }
    
    if (awayTeam === homeTeam) {
      setError('Please select different teams');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);
    setImageUrl(null);
    
    try {
      // Get calculation results
      const response = await axios.post(`${API_URL}/api/cbase/calculate-spread`, {
        away_team: awayTeam,
        home_team: homeTeam,
        neutral: neutral
      });
      setResult(response.data);
      
      // Get matchup image
      const imageResponse = await axios.post(`${API_URL}/api/cbase/matchup-image`, {
        away_team: awayTeam,
        home_team: homeTeam,
        neutral: neutral
      }, {
        responseType: 'blob'
      });
      
      const imageObjectUrl = URL.createObjectURL(imageResponse.data);
      setImageUrl(imageObjectUrl);
    } catch (error) {
      setError('Error calculating spread. Please try again.');
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  };

  // Filter teams based on search
  const filteredAwayTeams = teams.filter(team =>
    team.toLowerCase().includes(awaySearchTerm.toLowerCase())
  );

  const filteredHomeTeams = teams.filter(team =>
    team.toLowerCase().includes(homeSearchTerm.toLowerCase())
  );

  const getGQIColor = (gqi: number) => {
    if (gqi >= 8) return 'text-green-600 dark:text-green-400 bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-700';
    if (gqi >= 6) return 'text-green-500 dark:text-green-400 bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-700';
    if (gqi >= 4) return 'text-yellow-600 dark:text-yellow-400 bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-700';
    return 'text-gray-600 dark:text-gray-400 bg-gray-50 dark:bg-gray-800 border-gray-200 dark:border-gray-700';
  };

  const getWinProbColor = (prob: number) => {
    if (prob >= 70) return 'text-green-700 dark:text-green-400 font-bold';
    if (prob >= 55) return 'text-blue-700 dark:text-blue-400 font-semibold';
    return 'text-gray-700 dark:text-gray-300';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-gray-900 dark:to-gray-800 pt-16">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2">Matchup Calculator</h1>
          <p className="text-lg text-gray-600 dark:text-gray-400">Calculate spread and win probability for any matchup</p>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg overflow-hidden">
          <div className="p-6">
            <form onSubmit={calculateSpread} className="space-y-6">
              {/* Team Selectors - Side by Side */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Away Team */}
                <div>
                  <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                    Away Team
                  </label>
                  <div className="relative">
                    <input
                      type="text"
                      value={awayTeam || awaySearchTerm}
                      onChange={(e) => {
                        setAwaySearchTerm(e.target.value);
                        setAwayTeam('');
                        setShowAwayDropdown(true);
                      }}
                      onFocus={() => setShowAwayDropdown(true)}
                      onBlur={() => setTimeout(() => setShowAwayDropdown(false), 200)}
                      placeholder="Search for a team..."
                      className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400"
                    />
                    {showAwayDropdown && filteredAwayTeams.length > 0 && (
                      <div className="absolute z-10 w-full mt-1 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg shadow-lg max-h-60 overflow-y-auto">
                        {filteredAwayTeams.map(team => (
                          <div
                            key={team}
                            onClick={() => {
                              setAwayTeam(team);
                              setAwaySearchTerm('');
                              setShowAwayDropdown(false);
                            }}
                            className="px-4 py-3 hover:bg-purple-50 dark:hover:bg-purple-900/30 cursor-pointer text-gray-900 dark:text-gray-100"
                          >
                            {team}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>

                {/* Home Team */}
                <div>
                  <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                    Home Team
                  </label>
                  <div className="relative">
                    <input
                      type="text"
                      value={homeTeam || homeSearchTerm}
                      onChange={(e) => {
                        setHomeSearchTerm(e.target.value);
                        setHomeTeam('');
                        setShowHomeDropdown(true);
                      }}
                      onFocus={() => setShowHomeDropdown(true)}
                      onBlur={() => setTimeout(() => setShowHomeDropdown(false), 200)}
                      placeholder="Search for a team..."
                      className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400"
                    />
                    {showHomeDropdown && filteredHomeTeams.length > 0 && (
                      <div className="absolute z-10 w-full mt-1 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg shadow-lg max-h-60 overflow-y-auto">
                        {filteredHomeTeams.map(team => (
                          <div
                            key={team}
                            onClick={() => {
                              setHomeTeam(team);
                              setHomeSearchTerm('');
                              setShowHomeDropdown(false);
                            }}
                            className="px-4 py-3 hover:bg-purple-50 dark:hover:bg-purple-900/30 cursor-pointer text-gray-900 dark:text-gray-100"
                          >
                            {team}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Game Location */}
              <div>
                <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
                  Game Location
                </label>
                <div className="flex items-center space-x-6">
                  <label className="flex items-center cursor-pointer">
                    <input
                      type="radio"
                      checked={!neutral}
                      onChange={() => setNeutral(false)}
                      className="w-4 h-4 text-purple-600 border-gray-300 dark:border-gray-600 focus:ring-purple-500"
                    />
                    <span className="ml-2 text-gray-700 dark:text-gray-300">On Campus</span>
                  </label>
                  <label className="flex items-center cursor-pointer">
                    <input
                      type="radio"
                      checked={neutral}
                      onChange={() => setNeutral(true)}
                      className="w-4 h-4 text-purple-600 border-gray-300 dark:border-gray-600 focus:ring-purple-500"
                    />
                    <span className="ml-2 text-gray-700 dark:text-gray-300">Neutral Site</span>
                  </label>
                </div>
              </div>

              {/* Error Message */}
              {error && (
                <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-700 text-red-700 dark:text-red-300 px-4 py-3 rounded-lg">
                  {error}
                </div>
              )}

              {/* Calculate Button */}
              <button
                type="submit"
                disabled={loading}
                className="w-full bg-purple-600 dark:bg-purple-500 hover:bg-purple-700 dark:hover:bg-purple-600 text-white font-bold py-4 px-6 rounded-lg transition-colors disabled:bg-gray-400 dark:disabled:bg-gray-600 disabled:cursor-not-allowed flex items-center justify-center space-x-3 text-lg"
              >
                {loading ? (
                  <>
                    <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-white"></div>
                    <span>Calculating...</span>
                  </>
                ) : (
                  <>
                    <Calculator className="w-6 h-6" />
                    <span>Calculate Matchup</span>
                  </>
                )}
              </button>
            </form>

            {/* Results */}
            {imageUrl && (
              <div className="mt-8">
                <div className="border-t-2 border-gray-200 dark:border-gray-700 pt-6">
                  <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4 text-center">Matchup Comparison</h3>
                  
                  <div className="flex justify-center">
                    <div 
                      className="cursor-pointer"
                      onClick={() => setFullscreenImage(imageUrl)}
                    >
                      <img 
                        src={imageUrl} 
                        alt="Matchup Comparison" 
                        className="max-w-full h-auto rounded-lg shadow-lg hover:opacity-90 transition-opacity"
                      />
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Fullscreen Modal */}
        {fullscreenImage && (
          <div
            className="fixed inset-0 z-50 bg-black bg-opacity-90 flex items-center justify-center p-4"
            onClick={() => setFullscreenImage(null)}
          >
            <button
              onClick={() => setFullscreenImage(null)}
              className="absolute top-4 right-4 text-white hover:text-gray-300 text-4xl font-bold z-10"
              aria-label="Close fullscreen"
            >
              ×
            </button>
            <div className="relative max-w-full max-h-full">
              <Image
                src={fullscreenImage}
                alt="Fullscreen view"
                width={2000}
                height={2000}
                className="max-w-full max-h-[95vh] w-auto h-auto object-contain"
                unoptimized
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}