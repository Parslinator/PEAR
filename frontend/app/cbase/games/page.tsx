'use client';

import { useState, useEffect } from 'react';
import Image from 'next/image';
import { X } from 'lucide-react';

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

interface MatchupData {
  spread: number;
  formatted_spread: string;
  home_win_prob: number;
  away_win_prob: number;
  game_quality: number;
  home_pr: number;
  away_pr: number;
  home_elo: number;
  away_elo: number;
}

export default function CbaseGamesPage() {
  const [games, setGames] = useState<Game[]>([]);
  const [date, setDate] = useState<string>('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedGame, setSelectedGame] = useState<Game | null>(null);
  const [matchupData, setMatchupData] = useState<MatchupData | null>(null);
  const [loadingMatchup, setLoadingMatchup] = useState(false);

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

  const handleGameClick = async (game: Game) => {
    setSelectedGame(game);
    setLoadingMatchup(true);
    
    try {
      const response = await fetch(`${API_URL}/api/cbase/calculate-spread`, {
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
      
      if (!response.ok) throw new Error('Failed to fetch matchup data');
      
      const data = await response.json();
      setMatchupData(data);
    } catch (err) {
      console.error('Error fetching matchup:', err);
    } finally {
      setLoadingMatchup(false);
    }
  };

  const closeModal = () => {
    setSelectedGame(null);
    setMatchupData(null);
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
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2">
            Today's Games
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-400">{date}</p>
          <p className="text-sm text-gray-500 dark:text-gray-500 mt-1">
            {games.length} {games.length === 1 ? 'game' : 'games'} scheduled
          </p>
        </div>

        {games.length === 0 ? (
          <div className="text-center py-12">
            <p className="text-xl text-gray-600 dark:text-gray-400">
              No games scheduled for today
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {games.map((game, index) => (
              <GameCard
                key={index}
                game={game}
                onClick={() => handleGameClick(game)}
              />
            ))}
          </div>
        )}
      </div>

      {/* Matchup Modal */}
      {selectedGame && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
          onClick={closeModal}
        >
          <div 
            className="bg-white dark:bg-gray-800 rounded-lg max-w-4xl w-full max-h-[90vh] overflow-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="sticky top-0 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 p-4 flex justify-between items-center">
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
                {selectedGame.away_team} @ {selectedGame.home_team}
              </h2>
              <button
                onClick={closeModal}
                className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
              >
                <X className="w-6 h-6 text-gray-600 dark:text-gray-400" />
              </button>
            </div>

            <div className="p-6">
              {loadingMatchup ? (
                <div className="text-center py-12">
                  <div className="text-lg text-gray-600 dark:text-gray-400">Loading matchup details...</div>
                </div>
              ) : matchupData ? (
                <div className="space-y-6">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="text-center">
                      <div className="text-sm text-gray-500 dark:text-gray-400 mb-2">Away Team</div>
                      <div className="text-xl font-bold text-gray-900 dark:text-white mb-2">
                        {selectedGame.away_team}
                      </div>
                      <div className="text-3xl font-bold text-red-600 dark:text-red-400">
                        {matchupData.away_win_prob.toFixed(1)}%
                      </div>
                      <div className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                        Rating: {matchupData.away_pr}
                      </div>
                    </div>

                    <div className="text-center">
                      <div className="text-sm text-gray-500 dark:text-gray-400 mb-2">Home Team</div>
                      <div className="text-xl font-bold text-gray-900 dark:text-white mb-2">
                        {selectedGame.home_team}
                      </div>
                      <div className="text-3xl font-bold text-green-600 dark:text-green-400">
                        {matchupData.home_win_prob.toFixed(1)}%
                      </div>
                      <div className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                        Rating: {matchupData.home_pr}
                      </div>
                    </div>
                  </div>

                  <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-center">
                      <div>
                        <div className="text-sm text-gray-500 dark:text-gray-400">Spread</div>
                        <div className="text-lg font-semibold text-gray-900 dark:text-white">
                          {matchupData.formatted_spread}
                        </div>
                      </div>
                      <div>
                        <div className="text-sm text-gray-500 dark:text-gray-400">Game Quality</div>
                        <div className="text-lg font-semibold text-gray-900 dark:text-white">
                          {matchupData.game_quality.toFixed(1)}
                        </div>
                      </div>
                      <div className="col-span-2 md:col-span-1">
                        <div className="text-sm text-gray-500 dark:text-gray-400">Location</div>
                        <div className="text-lg font-semibold text-gray-900 dark:text-white">
                          {selectedGame.Location}
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="text-center text-xs text-gray-500 dark:text-gray-400 pt-4 border-t border-gray-200 dark:border-gray-700">
                    Projection by PEAR
                  </div>
                </div>
              ) : (
                <div className="text-center py-12">
                  <div className="text-lg text-red-600 dark:text-red-400">Failed to load matchup data</div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function GameCard({ game, onClick }: { game: Game; onClick: () => void }) {
  const awayWinProb = ((1 - game.home_win_prob) * 100).toFixed(1);
  const homeWinProb = (game.home_win_prob * 100).toFixed(1);

  return (
    <div
      onClick={onClick}
      className="bg-white dark:bg-gray-800 rounded-lg shadow-md hover:shadow-xl transition-all cursor-pointer border border-gray-200 dark:border-gray-700 overflow-hidden"
    >
      <div className="p-4">
        {/* GQI Badge */}
        <div className="text-center mb-4">
          <span className="inline-block px-3 py-1 bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 rounded-full text-sm font-semibold">
            GQI: {game.GQI.toFixed(1)}
          </span>
        </div>

        {/* Away Team */}
        <div className="flex items-center justify-between mb-4 pb-4 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center space-x-3 flex-1">
            <div className="relative w-12 h-12 flex-shrink-0">
              <Image
                src={`${API_URL}/api/baseball-logo/${encodeURIComponent(game.away_team)}`}
                alt={`${game.away_team} logo`}
                width={48}
                height={48}
                className="object-contain"
                unoptimized
              />
            </div>
            <div className="flex-1 min-w-0">
              <div className="text-sm font-medium text-gray-900 dark:text-white truncate">
                {game.away_team}
              </div>
              <div className="text-xs text-gray-500 dark:text-gray-400">Away</div>
            </div>
          </div>
          <div className="text-lg font-bold text-red-600 dark:text-red-400 ml-2">
            {awayWinProb}%
          </div>
        </div>

        {/* Home Team */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3 flex-1">
            <div className="relative w-12 h-12 flex-shrink-0">
              <Image
                src={`${API_URL}/api/baseball-logo/${encodeURIComponent(game.home_team)}`}
                alt={`${game.home_team} logo`}
                width={48}
                height={48}
                className="object-contain"
                unoptimized
              />
            </div>
            <div className="flex-1 min-w-0">
              <div className="text-sm font-medium text-gray-900 dark:text-white truncate">
                {game.home_team}
              </div>
              <div className="text-xs text-gray-500 dark:text-gray-400">Home</div>
            </div>
          </div>
          <div className="text-lg font-bold text-green-600 dark:text-green-400 ml-2">
            {homeWinProb}%
          </div>
        </div>

        {/* Footer */}
        <div className="pt-3 border-t border-gray-200 dark:border-gray-700 text-center">
          <div className="text-xs text-gray-500 dark:text-gray-400 font-medium">
            PROJECTION (PEAR)
          </div>
        </div>
      </div>
    </div>
  );
}