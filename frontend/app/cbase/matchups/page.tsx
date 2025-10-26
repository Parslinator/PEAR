'use client';

import { useState, useEffect } from 'react';
import { Trophy, Calculator } from 'lucide-react';
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

  useEffect(() => {
    fetchTeams();
  }, []);

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
    if (gqi >= 8) return 'text-green-600 bg-green-50 border-green-200';
    if (gqi >= 6) return 'text-green-500 bg-green-50 border-green-200';
    if (gqi >= 4) return 'text-yellow-600 bg-yellow-50 border-yellow-200';
    return 'text-gray-600 bg-gray-50 border-gray-200';
  };

  const getWinProbColor = (prob: number) => {
    if (prob >= 70) return 'text-green-700 font-bold';
    if (prob >= 55) return 'text-blue-700 font-semibold';
    return 'text-gray-700';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-900 via-blue-800 to-blue-900 text-white py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-center mb-4">
            <Calculator className="w-12 h-12 text-yellow-400 mr-3" />
            <h1 className="text-4xl font-bold">Matchup Calculator</h1>
          </div>
          <p className="text-center text-blue-200">Calculate spread and win probability for any matchup</p>
        </div>
      </div>

      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="bg-white rounded-xl shadow-lg overflow-hidden">
          <div className="bg-gradient-to-r from-purple-600 to-purple-700 px-6 py-4">
            <h2 className="text-xl font-bold text-white">Team Selection</h2>
          </div>

          <div className="p-6">
            <form onSubmit={calculateSpread} className="space-y-6">
              {/* Away Team */}
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">
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
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent text-gray-900"
                  />
                  {showAwayDropdown && filteredAwayTeams.length > 0 && (
                    <div className="absolute z-10 w-full mt-1 bg-white border border-gray-300 rounded-lg shadow-lg max-h-60 overflow-y-auto">
                      {filteredAwayTeams.map(team => (
                        <div
                          key={team}
                          onClick={() => {
                            setAwayTeam(team);
                            setAwaySearchTerm('');
                            setShowAwayDropdown(false);
                          }}
                          className="px-4 py-3 hover:bg-purple-50 cursor-pointer text-gray-900"
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
                <label className="block text-sm font-semibold text-gray-700 mb-2">
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
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent text-gray-900"
                  />
                  {showHomeDropdown && filteredHomeTeams.length > 0 && (
                    <div className="absolute z-10 w-full mt-1 bg-white border border-gray-300 rounded-lg shadow-lg max-h-60 overflow-y-auto">
                      {filteredHomeTeams.map(team => (
                        <div
                          key={team}
                          onClick={() => {
                            setHomeTeam(team);
                            setHomeSearchTerm('');
                            setShowHomeDropdown(false);
                          }}
                          className="px-4 py-3 hover:bg-purple-50 cursor-pointer text-gray-900"
                        >
                          {team}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              {/* Game Location */}
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-3">
                  Game Location
                </label>
                <div className="flex items-center space-x-6">
                  <label className="flex items-center cursor-pointer">
                    <input
                      type="radio"
                      checked={!neutral}
                      onChange={() => setNeutral(false)}
                      className="w-4 h-4 text-purple-600 border-gray-300 focus:ring-purple-500"
                    />
                    <span className="ml-2 text-gray-700">On Campus</span>
                  </label>
                  <label className="flex items-center cursor-pointer">
                    <input
                      type="radio"
                      checked={neutral}
                      onChange={() => setNeutral(true)}
                      className="w-4 h-4 text-purple-600 border-gray-300 focus:ring-purple-500"
                    />
                    <span className="ml-2 text-gray-700">Neutral Site</span>
                  </label>
                </div>
              </div>

              {/* Error Message */}
              {error && (
                <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">
                  {error}
                </div>
              )}

              {/* Calculate Button */}
              <button
                type="submit"
                disabled={loading}
                className="w-full bg-purple-600 hover:bg-purple-700 text-white font-bold py-4 px-6 rounded-lg transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center justify-center space-x-3 text-lg"
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
                <div className="border-t-2 border-gray-200 pt-6">
                  <h3 className="text-xl font-bold text-gray-900 mb-4 text-center">Matchup Comparison</h3>
                  
                  <div className="flex justify-center">
                    <img 
                      src={imageUrl} 
                      alt="Matchup Comparison" 
                      className="max-w-full h-auto rounded-lg shadow-lg"
                    />
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}