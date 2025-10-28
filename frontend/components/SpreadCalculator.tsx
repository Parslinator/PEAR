'use client';

import { useState, useEffect } from 'react';
import Image from 'next/image';
import axios from 'axios';
import { Calculator, TrendingUp, TrendingDown } from 'lucide-react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface CalculationResult {
  spread: number;
  formatted_spread: string;
  home_win_prob: number;
  away_win_prob: number;
  home_score: number;
  away_score: number;
  predicted_total: number;
  game_quality: number;
  home_pr: number;
  away_pr: number;
}

export default function SpreadCalculator() {
  const [teams, setTeams] = useState<string[]>([]);
  const [awayTeam, setAwayTeam] = useState('');
  const [homeTeam, setHomeTeam] = useState('');
  const [awaySearch, setAwaySearch] = useState('');
  const [homeSearch, setHomeSearch] = useState('');
  const [showAwayDropdown, setShowAwayDropdown] = useState(false);
  const [showHomeDropdown, setShowHomeDropdown] = useState(false);
  const [neutral, setNeutral] = useState(false);
  const [result, setResult] = useState<CalculationResult | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [fullscreenImage, setFullscreenImage] = useState<string | null>(null);

  useEffect(() => {
    fetchTeams();
  }, []);

  // Close dropdowns when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as HTMLElement;
      if (!target.closest('.relative')) {
        setShowAwayDropdown(false);
        setShowHomeDropdown(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
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
      const response = await axios.get(`${API_URL}/api/teams`);
      setTeams(response.data.teams);
    } catch (error) {
      console.error('Error fetching teams:', error);
    }
  };

  // Filter teams based on search input
  const getFilteredTeams = (searchTerm: string) => {
    if (!searchTerm) return teams;
    return teams.filter(team => 
      team.toLowerCase().includes(searchTerm.toLowerCase())
    );
  };

  const handleAwayTeamSelect = (team: string) => {
    setAwayTeam(team);
    setAwaySearch(team);
    setShowAwayDropdown(false);
  };

  const handleHomeTeamSelect = (team: string) => {
    setHomeTeam(team);
    setHomeSearch(team);
    setShowHomeDropdown(false);
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
      const response = await axios.post(`${API_URL}/api/calculate-spread`, {
        away_team: awayTeam,
        home_team: homeTeam,
        neutral: neutral
      });
      setResult(response.data);
      
      // Generate matchup image
      const imageResponse = await axios.post(
        `${API_URL}/api/generate-matchup-image`, 
        {
          away_team: awayTeam,
          home_team: homeTeam,
          neutral: neutral
        },
        { responseType: 'blob' }
      );
      
      const imageBlob = new Blob([imageResponse.data], { type: 'image/png' });
      const imageObjectUrl = URL.createObjectURL(imageBlob);
      setImageUrl(imageObjectUrl);
      
    } catch (error) {
      setError('Error calculating spread. Please try again.');
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  };

  const getGQIColor = (gqi: number) => {
    if (gqi >= 8) return 'text-green-600';
    if (gqi >= 6) return 'text-green-500';
    if (gqi >= 4) return 'text-yellow-600';
    return 'text-gray-600';
  };

  return (
    <div>
      <form onSubmit={calculateSpread} className="space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="relative">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Away Team
            </label>
            <input
              type="text"
              value={awaySearch}
              onChange={(e) => {
                setAwaySearch(e.target.value);
                setAwayTeam('');
                setShowAwayDropdown(true);
              }}
              onFocus={() => setShowAwayDropdown(true)}
              placeholder="Search for away team..."
              className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400"
            />
            {showAwayDropdown && getFilteredTeams(awaySearch).length > 0 && (
              <div className="absolute z-10 w-full mt-1 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg shadow-lg max-h-60 overflow-y-auto">
                {getFilteredTeams(awaySearch).map((team) => (
                  <button
                    key={team}
                    type="button"
                    onClick={() => handleAwayTeamSelect(team)}
                    className="w-full text-left px-4 py-2 hover:bg-purple-50 dark:hover:bg-purple-900/30 transition-colors text-gray-900 dark:text-white"
                  >
                    {team}
                  </button>
                ))}
              </div>
            )}
          </div>

          <div className="relative">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Home Team
            </label>
            <input
              type="text"
              value={homeSearch}
              onChange={(e) => {
                setHomeSearch(e.target.value);
                setHomeTeam('');
                setShowHomeDropdown(true);
              }}
              onFocus={() => setShowHomeDropdown(true)}
              placeholder="Search for home team..."
              className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400"
            />
            {showHomeDropdown && getFilteredTeams(homeSearch).length > 0 && (
              <div className="absolute z-10 w-full mt-1 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg shadow-lg max-h-60 overflow-y-auto">
                {getFilteredTeams(homeSearch).map((team) => (
                  <button
                    key={team}
                    type="button"
                    onClick={() => handleHomeTeamSelect(team)}
                    className="w-full text-left px-4 py-2 hover:bg-purple-50 dark:hover:bg-purple-900/30 transition-colors text-gray-900 dark:text-white"
                  >
                    {team}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        <div className="flex items-center space-x-2">
          <input
            type="checkbox"
            id="neutral"
            checked={neutral}
            onChange={(e) => setNeutral(e.target.checked)}
            className="w-4 h-4 text-purple-600 border-gray-300 dark:border-gray-600 rounded focus:ring-purple-500 dark:bg-gray-700"
          />
          <label htmlFor="neutral" className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Neutral Site Game
          </label>
        </div>

        {error && (
          <div className="bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-300 px-4 py-3 rounded-lg">
            {error}
          </div>
        )}

        <button
          type="submit"
          disabled={loading}
          className="w-full bg-purple-600 dark:bg-purple-500 hover:bg-purple-700 dark:hover:bg-purple-600 text-white font-semibold py-3 px-6 rounded-lg transition-colors disabled:bg-gray-400 dark:disabled:bg-gray-600 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
        >
          {loading ? (
            <>
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
              <span>Calculating...</span>
            </>
          ) : (
            <>
              <Calculator className="w-5 h-5" />
              <span>Calculate Spread</span>
            </>
          )}
        </button>
      </form>

      {imageUrl && (
        <div className="mt-6">
          <div 
            className="bg-white dark:bg-gray-700 rounded-lg border-2 border-purple-200 dark:border-purple-700 overflow-hidden cursor-pointer"
            onClick={() => setFullscreenImage(imageUrl)}
          >
            <img 
              src={imageUrl} 
              alt="Matchup Preview" 
              className="w-full h-auto hover:opacity-90 transition-opacity"
            />
          </div>
          <button
            onClick={() => {
              const a = document.createElement('a');
              a.href = imageUrl;
              a.download = `${awayTeam}_vs_${homeTeam}_matchup.png`;
              a.click();
            }}
            className="mt-3 w-full px-4 py-2 bg-purple-600 dark:bg-purple-500 text-white rounded-lg hover:bg-purple-700 dark:hover:bg-purple-600 transition-colors font-semibold"
          >
            Download Image
          </button>
        </div>
      )}

      {/* Fullscreen Modal */}
      {fullscreenImage && (
        <div
          className="fixed inset-0 z-50 bg-black bg-opacity-90 dark:bg-opacity-95 flex items-center justify-center p-4"
          onClick={() => setFullscreenImage(null)}
        >
          <button
            onClick={() => setFullscreenImage(null)}
            className="absolute top-4 right-4 text-white hover:text-gray-300 dark:hover:text-gray-400 text-4xl font-bold z-10"
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
  );
}