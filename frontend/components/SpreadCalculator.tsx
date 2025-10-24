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
    
    try {
      const response = await axios.post(`${API_URL}/api/calculate-spread`, {
        away_team: awayTeam,
        home_team: homeTeam,
        neutral: neutral
      });
      setResult(response.data);
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

      {result && (
        <div className="mt-6 space-y-4">
          <div className="bg-gradient-to-r from-purple-50 to-blue-50 rounded-lg p-6 border-2 border-purple-200">
            <div className="text-center mb-4">
              <h3 className="text-2xl font-bold text-gray-900 mb-2">
                {result.formatted_spread}
              </h3>
              <p className="text-sm text-gray-600">
                {neutral ? `${awayTeam} (N) ${homeTeam}` : `${awayTeam} at ${homeTeam}`}
              </p>
            </div>

            <div className="grid grid-cols-2 gap-4 mb-4">
              <div className="bg-white rounded-lg p-4 text-center">
                <div className="text-xs text-gray-600 mb-1">Away Win %</div>
                <div className="text-3xl font-bold text-gray-900 flex items-center justify-center">
                  {result.away_win_prob < 50 ? (
                    <TrendingDown className="w-6 h-6 text-red-500 mr-2" />
                  ) : (
                    <TrendingUp className="w-6 h-6 text-green-500 mr-2" />
                  )}
                  {result.away_win_prob}%
                </div>
                <div className="text-sm text-gray-600 mt-2">
                  Proj: {result.away_score}
                </div>
              </div>

              <div className="bg-white rounded-lg p-4 text-center">
                <div className="text-xs text-gray-600 mb-1">Home Win %</div>
                <div className="text-3xl font-bold text-gray-900 flex items-center justify-center">
                  {result.home_win_prob < 50 ? (
                    <TrendingDown className="w-6 h-6 text-red-500 mr-2" />
                  ) : (
                    <TrendingUp className="w-6 h-6 text-green-500 mr-2" />
                  )}
                  {result.home_win_prob}%
                </div>
                <div className="text-sm text-gray-600 mt-2">
                  Proj: {result.home_score}
                </div>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-3 text-center">
              <div className="bg-white rounded-lg p-3">
                <div className="text-xs text-gray-600 mb-1">O/U</div>
                <div className="text-lg font-bold text-gray-900">
                  {result.predicted_total}
                </div>
              </div>
              <div className="bg-white rounded-lg p-3">
                <div className="text-xs text-gray-600 mb-1">GQI</div>
                <div className={`text-lg font-bold ${getGQIColor(result.game_quality)}`}>
                  {result.game_quality}
                </div>
              </div>
              <div className="bg-white rounded-lg p-3">
                <div className="text-xs text-gray-600 mb-1">Spread</div>
                <div className="text-lg font-bold text-gray-900">
                  {Math.abs(result.spread)}
                </div>
              </div>
            </div>
          </div>

          <div className="bg-gray-50 rounded-lg p-4">
            <h4 className="font-semibold text-gray-900 mb-3">Power Ratings</h4>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <div className="text-sm text-gray-600">{awayTeam}</div>
                <div className="text-xl font-bold text-gray-900">{result.away_pr}</div>
              </div>
              <div>
                <div className="text-sm text-gray-600">{homeTeam}</div>
                <div className="text-xl font-bold text-gray-900">{result.home_pr}</div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}