'use client';

import { useState, useEffect, useMemo } from 'react';
import Image from 'next/image';
import { X, Download } from 'lucide-react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || '';

interface Game {
  home_team: string;
  away_team: string;
  Location: string;
  PEAR: number;
  GQI: number;
  Date: string;
  home_win_prob: number;
}

type SortMode = 'gqi' | 'time';

export default function CbaseGamesPage() {
  const [games, setGames] = useState<Game[]>([]);
  const [date, setDate] = useState<string>('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedGame, setSelectedGame] = useState<Game | null>(null);
  const [matchupImageUrl, setMatchupImageUrl] = useState<string | null>(null);
  const [loadingImage, setLoadingImage] = useState(false);
  const [sortMode, setSortMode] = useState<SortMode>('gqi');

  useEffect(() => {
    fetchGames();
  }, []);

  const fetchGames = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${API_URL}/api/cbase/schedule/today`);
      if (!response.ok) throw new Error('Failed to fetch games');
      
      const data = await response.json();
      setGames(data.games || []);
      setDate(data.date || '');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const sortedGames = useMemo(() => {
    const sorted = [...games];
    if (sortMode === 'gqi') {
      return sorted.sort((a, b) => b.GQI - a.GQI);
    }
    return sorted; // time sorting would go here if we had time data
  }, [games, sortMode]);

  const handleGameClick = async (game: Game) => {
    setSelectedGame(game);
    setLoadingImage(true);
    setMatchupImageUrl(null);
    
    try {
      const response = await fetch(`${API_URL}/api/cbase/matchup-image`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          away_team: game.away_team,
          home_team: game.home_team,
          neutral: game.Location === 'Neutral'
        }),
      });
      
      if (!response.ok) throw new Error('Failed to fetch matchup image');
      
      const blob = await response.blob();
      const imageObjectUrl = URL.createObjectURL(blob);
      setMatchupImageUrl(imageObjectUrl);
    } catch (err) {
      console.error('Error fetching matchup image:', err);
    } finally {
      setLoadingImage(false);
    }
  };

  const closeModal = () => {
    if (matchupImageUrl) {
      URL.revokeObjectURL(matchupImageUrl);
    }
    setSelectedGame(null);
    setMatchupImageUrl(null);
  };

  const getGQIColor = (gqi: number) => {
    if (gqi >= 9) return 'bg-green-700 dark:bg-green-800';
    if (gqi >= 8) return 'bg-green-500 dark:bg-green-500';
    if (gqi >= 7) return 'bg-green-400 dark:bg-green-400';
    if (gqi >= 6) return 'bg-yellow-500 dark:bg-yellow-600';
    if (gqi >= 4) return 'bg-orange-500 dark:bg-orange-600';
    return 'bg-red-600 dark:bg-red-600';
  };

  const downloadCSV = () => {
    const headers = ['Date', 'Away Team', 'Home Team', 'Location', 'Away Win %', 'Home Win %', 'PEAR', 'GQI'];
    const csvData = games.map(game => [
      date,
      game.away_team,
      game.home_team,
      game.Location,
      `${((1 - game.home_win_prob) * 100).toFixed(1)}%`,
      `${(game.home_win_prob * 100).toFixed(1)}%`,
      game.PEAR.toFixed(1),
      game.GQI.toFixed(1)
    ]);
    
    const csvContent = [
      headers.join(','),
      ...csvData.map(row => row.join(','))
    ].join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'cbase_games.csv';
    a.click();
    window.URL.revokeObjectURL(url);
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 pt-20 px-4">
        <div className="max-w-7xl mx-auto">
          <div className="text-center py-12">
            <div className="text-xl text-gray-600 dark:text-gray-400">Loading games...</div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 pt-20 px-4">
        <div className="max-w-7xl mx-auto">
          <div className="text-center py-12">
            <div className="text-xl text-red-600 dark:text-red-400">Error: {error}</div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 pt-20 px-4 pb-8">
      <div className="max-w-7xl mx-auto">
        <div className="mb-6">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2">
            Today's Games
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-400">{date}</p>
        </div>

        <div className="mb-4 flex justify-between items-center">
          <div className="flex gap-2">
            <button
              onClick={() => setSortMode('gqi')}
              className={`px-4 py-2 rounded-lg font-semibold transition-colors ${
                sortMode === 'gqi'
                  ? 'bg-[#CECEB2] text-gray-900'
                  : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
              }`}
            >
              GQI
            </button>
            <button
              onClick={() => setSortMode('time')}
              className={`px-4 py-2 rounded-lg font-semibold transition-colors ${
                sortMode === 'time'
                  ? 'bg-[#CECEB2] text-gray-900'
                  : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
              }`}
            >
              TIME
            </button>
          </div>
          <button
            onClick={downloadCSV}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 dark:bg-blue-500 text-white rounded-lg hover:bg-blue-700 dark:hover:bg-blue-600 font-semibold transition-colors"
          >
            <Download className="w-4 h-4" />
            Export CSV
          </button>
        </div>

        {games.length === 0 ? (
          <div className="text-center py-12">
            <p className="text-xl text-gray-600 dark:text-gray-400">
              No games scheduled for today
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
            {sortedGames.map((game, index) => (
              <GameCard
                key={index}
                game={game}
                onClick={() => handleGameClick(game)}
                getGQIColor={getGQIColor}
              />
            ))}
          </div>
        )}

        <div className="mt-6 text-xs text-gray-600 dark:text-gray-400 space-y-1">
          <p><strong className="dark:text-gray-300">GQI</strong> - Game Quality Index (1-10 scale, higher is better). Click any game to view matchup!</p>
          <p><strong className="dark:text-gray-300">Win %</strong> - PEAR's projected win probability for each team</p>
          <p><strong className="dark:text-gray-300">PROJECTION</strong> - PEAR's predicted spread</p>
        </div>
      </div>

      {/* Matchup Image Modal */}
      {selectedGame && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-[9999] p-4"
          onClick={closeModal}
        >
          <div 
            className="relative max-w-6xl max-h-[90vh] bg-white dark:bg-gray-800 rounded-lg shadow-2xl overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              onClick={closeModal}
              className="absolute top-4 right-4 p-2 bg-gray-900 bg-opacity-50 hover:bg-opacity-75 rounded-full text-white transition-all z-10"
            >
              <X className="w-6 h-6" />
            </button>
            
            {loadingImage ? (
              <div className="flex items-center justify-center p-12">
                <div className="text-lg text-gray-600 dark:text-gray-400">Loading matchup...</div>
              </div>
            ) : matchupImageUrl ? (
              <img
                src={matchupImageUrl}
                alt={`${selectedGame.away_team} vs ${selectedGame.home_team} matchup`}
                className="max-w-full max-h-[90vh] object-contain"
                onError={(e) => {
                  console.error('Failed to load matchup image');
                  e.currentTarget.src = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="400" height="300"><rect width="400" height="300" fill="%23f3f4f6"/><text x="50%" y="50%" text-anchor="middle" fill="%23374151" font-size="18">Matchup not available</text></svg>';
                }}
              />
            ) : (
              <div className="flex items-center justify-center p-12">
                <div className="text-lg text-red-600 dark:text-red-400">Failed to load matchup</div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function GameCard({ 
  game, 
  onClick, 
  getGQIColor 
}: { 
  game: Game; 
  onClick: () => void;
  getGQIColor: (gqi: number) => string;
}) {
  const awayWinProb = ((1 - game.home_win_prob) * 100).toFixed(1);
  const homeWinProb = (game.home_win_prob * 100).toFixed(1);

  return (
    <div
      onClick={onClick}
      className="bg-white dark:bg-gray-800 rounded-lg p-3 shadow-md border-2 border-gray-200 dark:border-gray-700 hover:shadow-lg hover:border-blue-400 dark:hover:border-blue-500 transition-all cursor-pointer"
    >
      {/* Header: GQI */}
      <div className="flex justify-end items-center mb-2 pb-2 border-b border-gray-200 dark:border-gray-700">
        <span className={`px-2 py-0.5 rounded-full text-white text-xs font-bold ${getGQIColor(game.GQI)}`}>
          {game.GQI.toFixed(1)}
        </span>
      </div>

      {/* Away Team */}
      <div className="flex justify-between items-center mb-2">
        <div className="flex items-center gap-2 flex-1 min-w-0">
          <img 
            src={`${API_URL}/api/baseball-logo/${encodeURIComponent(game.away_team)}`}
            alt={`${game.away_team} logo`}
            className="w-6 h-6 object-contain flex-shrink-0"
            onError={(e) => {
              e.currentTarget.style.display = 'none';
            }}
          />
          <span className="text-sm font-semibold text-gray-900 dark:text-white truncate">
            {game.away_team}
          </span>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          <span className="text-xs font-medium text-gray-600 dark:text-gray-400">
            {awayWinProb}%
          </span>
        </div>
      </div>

      {/* Home Team */}
      <div className="flex justify-between items-center mb-2">
        <div className="flex items-center gap-2 flex-1 min-w-0">
          <img 
            src={`${API_URL}/api/baseball-logo/${encodeURIComponent(game.home_team)}`}
            alt={`${game.home_team} logo`}
            className="w-6 h-6 object-contain flex-shrink-0"
            onError={(e) => {
              e.currentTarget.style.display = 'none';
            }}
          />
          <span className="text-sm font-semibold text-gray-900 dark:text-white truncate">
            {game.home_team}
          </span>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          <span className="text-xs font-medium text-gray-600 dark:text-gray-400">
            {homeWinProb}%
          </span>
        </div>
      </div>

      {/* Projection */}
      <div className="pt-2 border-t border-gray-200 dark:border-gray-700">
        <div className="flex justify-between items-center">
          <span className="text-xs text-gray-600 dark:text-gray-400">PROJECTION</span>
          <span className="px-2 py-0.5 bg-blue-100 dark:bg-blue-900/50 text-blue-800 dark:text-blue-200 rounded font-medium text-xs">
            {game.PEAR.toFixed(1)}
          </span>
        </div>
      </div>
    </div>
  );
}