'use client';

import { useState, useEffect } from 'react';
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
  const [neutral, setNeutral] = useState(false);
  const [result, setResult] = useState<CalculationResult | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchTeams();
  }, []);

  const fetchTeams = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/teams`);
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
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Away Team
          </label>
          <select
            value={awayTeam}
            onChange={(e) => setAwayTeam(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
          >
            <option value="">Select Away Team</option>
            {teams.map((team) => (
              <option key={team} value={team}>
                {team}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Home Team
          </label>
          <select
            value={homeTeam}
            onChange={(e) => setHomeTeam(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
          >
            <option value="">Select Home Team</option>
            {teams.map((team) => (
              <option key={team} value={team}>
                {team}
              </option>
            ))}
          </select>
        </div>

        <div className="flex items-center space-x-2">
          <input
            type="checkbox"
            id="neutral"
            checked={neutral}
            onChange={(e) => setNeutral(e.target.checked)}
            className="w-4 h-4 text-purple-600 border-gray-300 rounded focus:ring-purple-500"
          />
          <label htmlFor="neutral" className="text-sm font-medium text-gray-700">
            Neutral Site Game
          </label>
        </div>

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">
            {error}
          </div>
        )}

        <button
          type="submit"
          disabled={loading}
          className="w-full bg-purple-600 hover:bg-purple-700 text-white font-semibold py-3 px-6 rounded-lg transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
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
          <div className="bg-white rounded-lg border-2 border-purple-200 overflow-hidden">
            <img 
              src={imageUrl} 
              alt="Matchup Preview" 
              className="w-full h-auto"
            />
          </div>
          <button
            onClick={() => {
              const a = document.createElement('a');
              a.href = imageUrl;
              a.download = `${awayTeam}_vs_${homeTeam}_matchup.png`;
              a.click();
            }}
            className="mt-3 w-full px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors font-semibold"
          >
            Download Image
          </button>
        </div>
      )}
    </div>
  );
}