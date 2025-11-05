'use client';

import { useState, useEffect, Suspense } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import { ArrowLeft, Trophy, TrendingUp, Calendar } from 'lucide-react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || '';

interface GameData {
  date: string;
  opponent: string;
  location: string;
  home_team: string;
  away_team: string;
  home_score: number | null;
  away_score: number | null;
  result: string | null;
  home_win_prob: number | null;
  resume_quality: number | null;
  home_net: number | null;
  away_net: number | null;
  opponent_net: number | null;
  team_win_prob: number | null;
  gqi: number | null;
  pear: string | null;
}

interface TeamMetrics {
  conference: string | null;
  rating: number | null;
  net: number | null;
  net_score: number | null;
  rpi: number | null;
  elo: number | null;
  elo_rank: number | null;
  resume_quality: number | null;
  q1: string | null;
  q2: string | null;
  q3: string | null;
  q4: string | null;
}

interface TeamProfileData {
  team_name: string;
  schedule: GameData[];
  metrics: TeamMetrics;
  data_date: string;
}

const formatDate = (dateStr: string) => {
  const date = new Date(dateStr);
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
};

const formatWinProbability = (prob: number | null) => {
  if (prob === null) return 'N/A';
  return `${(prob * 100).toFixed(1)}%`;
};

const getResultBadgeColor = (result: string | null) => {
  if (result === 'W') return 'bg-green-500 text-white';
  if (result === 'L') return 'bg-red-500 text-white';
  return 'bg-gray-300 dark:bg-gray-600 text-gray-700 dark:text-gray-300';
};

const getLocationBadge = (location: string) => {
  if (location === 'Home') return 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300';
  if (location === 'Away') return 'bg-orange-100 dark:bg-orange-900/30 text-orange-700 dark:text-orange-300';
  return 'bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300';
};

function CompletedGame({ game }: { game: GameData }) {
  return (
    <div className="bg-white dark:bg-gray-700 rounded-lg p-4 border border-gray-200 dark:border-gray-600 hover:shadow-md transition-shadow">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <span className={`px-2 py-1 rounded text-xs font-bold ${getResultBadgeColor(game.result)}`}>
            {game.result}
          </span>
          <span className="text-sm text-gray-500 dark:text-gray-400">
            {formatDate(game.date)}
          </span>
          <span className={`px-2 py-1 rounded text-xs font-semibold ${getLocationBadge(game.location)}`}>
            {game.location}
          </span>
        </div>
      </div>

      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3 flex-1">
          <img
            src={`${API_URL}/api/baseball-logo/${encodeURIComponent(game.opponent)}`}
            alt={`${game.opponent} logo`}
            className="w-10 h-10 object-contain"
            onError={(e) => {
              e.currentTarget.style.display = 'none';
            }}
          />
          <div>
            <p className="font-semibold text-gray-900 dark:text-white">
              {game.opponent}
            </p>
            {game.opponent_net && (
              <p className="text-xs text-gray-500 dark:text-gray-400">
                NET: #{game.opponent_net}
              </p>
            )}
          </div>
        </div>

        <div className="text-center px-4">
          <p className="text-2xl font-bold text-gray-900 dark:text-white">
            {game.home_score} - {game.away_score}
          </p>
        </div>

        <div className="flex flex-col items-end gap-1 flex-1">
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-600 dark:text-gray-400">Win Prob:</span>
            <span className="text-sm font-semibold text-gray-900 dark:text-white">
              {formatWinProbability(game.team_win_prob)}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-600 dark:text-gray-400">RQI:</span>
            <span className="text-sm font-semibold text-gray-900 dark:text-white">
              {game.resume_quality?.toFixed(2) || 'N/A'}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-600 dark:text-gray-400">GQI:</span>
            <span className="text-sm font-semibold text-gray-900 dark:text-white">
              {game.gqi?.toFixed(3) || 'N/A'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

function UpcomingGame({ game }: { game: GameData }) {
  return (
    <div className="bg-white dark:bg-gray-700 rounded-lg p-4 border border-gray-200 dark:border-gray-600 hover:shadow-md transition-shadow">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <span className="text-sm text-gray-500 dark:text-gray-400">
            {formatDate(game.date)}
          </span>
          <span className={`px-2 py-1 rounded text-xs font-semibold ${getLocationBadge(game.location)}`}>
            {game.location}
          </span>
        </div>
        {game.pear && (
          <span className="px-2 py-1 bg-indigo-100 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300 rounded text-xs font-semibold">
            {game.pear}
          </span>
        )}
      </div>

      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3 flex-1">
          <img
            src={`${API_URL}/api/baseball-logo/${encodeURIComponent(game.opponent)}`}
            alt={`${game.opponent} logo`}
            className="w-10 h-10 object-contain"
            onError={(e) => {
              e.currentTarget.style.display = 'none';
            }}
          />
          <div>
            <p className="font-semibold text-gray-900 dark:text-white">
              {game.opponent}
            </p>
            {game.opponent_net && (
              <p className="text-xs text-gray-500 dark:text-gray-400">
                NET: #{game.opponent_net}
              </p>
            )}
          </div>
        </div>

        <div className="flex flex-col items-end gap-1">
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-600 dark:text-gray-400">Win Prob:</span>
            <span className="text-sm font-semibold text-gray-900 dark:text-white">
              {formatWinProbability(game.team_win_prob)}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-600 dark:text-gray-400">RQI:</span>
            <span className="text-sm font-semibold text-gray-900 dark:text-white">
              {game.resume_quality?.toFixed(2) || 'N/A'}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-600 dark:text-gray-400">GQI:</span>
            <span className="text-sm font-semibold text-gray-900 dark:text-white">
              {game.gqi?.toFixed(3) || 'N/A'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

function TeamProfileContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const teamName = searchParams.get('team');
  
  const [profileData, setProfileData] = useState<TeamProfileData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (teamName) {
      fetchTeamProfile(teamName);
    }
  }, [teamName]);

  const fetchTeamProfile = async (team: string) => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch(`${API_URL}/api/cbase/profile-page`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          team_name: team,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to load team profile');
      }

      const data = await response.json();
      setProfileData(data);
      
    } catch (err) {
      console.error('Error fetching team profile:', err);
      setError('Failed to load team profile');
    } finally {
      setLoading(false);
    }
  };

  if (!teamName) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800 flex items-center justify-center">
        <div className="text-center">
          <p className="text-xl text-gray-600 dark:text-gray-400">No team selected</p>
          <button
            onClick={() => router.back()}
            className="mt-4 px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Go Back
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Back Button */}
        <button
          onClick={() => router.back()}
          className="flex items-center gap-2 text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white transition-colors mb-6"
        >
          <ArrowLeft size={20} />
          <span className="font-medium">Back to Rankings</span>
        </button>

        {/* Header */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 mb-6">
          <div className="flex items-center gap-4">
            <img 
              src={`${API_URL}/api/baseball-logo/${encodeURIComponent(teamName)}`}
              alt={`${teamName} logo`}
              className="w-20 h-20 object-contain"
              onError={(e) => {
                e.currentTarget.style.display = 'none';
              }}
            />
            <div>
              <h1 className="text-4xl font-bold text-gray-900 dark:text-white">
                {teamName}
              </h1>
              <p className="text-lg text-gray-600 dark:text-gray-400 mt-1">
                Team Profile
              </p>
            </div>
          </div>
        </div>

        {/* Content Area */}
        {loading ? (
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-12">
            <div className="flex flex-col items-center justify-center">
              <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600 dark:border-blue-400"></div>
              <p className="mt-4 text-gray-600 dark:text-gray-400 text-lg">
                Loading team profile...
              </p>
            </div>
          </div>
        ) : error ? (
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-12">
            <div className="text-center">
              <p className="text-red-600 dark:text-red-400 text-lg">{error}</p>
            </div>
          </div>
        ) : (
          <div className="space-y-6">
            {/* Stats Overview */}
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
                Season Overview
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 rounded-lg p-4">
                  <div className="flex items-center gap-3 mb-2">
                    <Trophy className="text-blue-600 dark:text-blue-400" size={20} />
                    <h3 className="font-semibold text-sm text-gray-700 dark:text-gray-300">
                      NET Rank
                    </h3>
                  </div>
                  <p className="text-2xl font-bold text-gray-900 dark:text-white">
                    #{profileData?.metrics?.net || 'N/A'}
                  </p>
                  {profileData?.metrics?.net_score && (
                    <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                      Score: {profileData.metrics.net_score.toFixed(3)}
                    </p>
                  )}
                </div>

                <div className="bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 rounded-lg p-4">
                  <div className="flex items-center gap-3 mb-2">
                    <TrendingUp className="text-purple-600 dark:text-purple-400" size={20} />
                    <h3 className="font-semibold text-sm text-gray-700 dark:text-gray-300">
                      TSR
                    </h3>
                  </div>
                  <p className="text-2xl font-bold text-gray-900 dark:text-white">
                    {profileData?.metrics?.rating?.toFixed(2) || 'N/A'}
                  </p>
                  {profileData?.metrics?.resume_quality && (
                    <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                      RQI: {profileData.metrics.resume_quality.toFixed(3)}
                    </p>
                  )}
                </div>

                <div className="bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 rounded-lg p-4">
                  <div className="flex items-center gap-3 mb-2">
                    <Calendar className="text-green-600 dark:text-green-400" size={20} />
                    <h3 className="font-semibold text-sm text-gray-700 dark:text-gray-300">
                      Conference
                    </h3>
                  </div>
                  <p className="text-2xl font-bold text-gray-900 dark:text-white">
                    {profileData?.metrics?.conference || 'N/A'}
                  </p>
                </div>

                <div className="bg-gradient-to-br from-orange-50 to-orange-100 dark:from-orange-900/20 dark:to-orange-800/20 rounded-lg p-4">
                  <div className="flex items-center gap-3 mb-2">
                    <Trophy className="text-orange-600 dark:text-orange-400" size={20} />
                    <h3 className="font-semibold text-sm text-gray-700 dark:text-gray-300">
                      Quadrant Record
                    </h3>
                  </div>
                  <div className="text-xs text-gray-900 dark:text-white space-y-1">
                    <p>Q1: {profileData?.metrics?.q1 || 'N/A'}</p>
                    <p>Q2: {profileData?.metrics?.q2 || 'N/A'}</p>
                    <p>Q3: {profileData?.metrics?.q3 || 'N/A'}</p>
                    <p>Q4: {profileData?.metrics?.q4 || 'N/A'}</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Schedule Section */}
            {profileData?.schedule && profileData.schedule.length > 0 ? (
              <>
                {(() => {
                  const completedGames = profileData.schedule.filter(game => game.result !== null);
                  const upcomingGames = profileData.schedule.filter(game => game.result === null);

                  return (
                    <>
                      {/* Completed Games */}
                      {completedGames.length > 0 && (
                        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
                          <div className="flex items-center gap-3 mb-4">
                            <Trophy className="text-blue-600 dark:text-blue-400" size={24} />
                            <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
                              Completed Games ({completedGames.length})
                            </h2>
                          </div>
                          <div className="space-y-3">
                            {completedGames.map((game, index) => (
                              <CompletedGame key={index} game={game} />
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Upcoming Games */}
                      {upcomingGames.length > 0 && (
                        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
                          <div className="flex items-center gap-3 mb-4">
                            <Calendar className="text-green-600 dark:text-green-400" size={24} />
                            <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
                              Upcoming Games ({upcomingGames.length})
                            </h2>
                          </div>
                          <div className="space-y-3">
                            {upcomingGames.map((game, index) => (
                              <UpcomingGame key={index} game={game} />
                            ))}
                          </div>
                        </div>
                      )}
                    </>
                  );
                })()}
              </>
            ) : (
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-12 text-center">
                <p className="text-gray-600 dark:text-gray-400">No schedule data available</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default function TeamProfilePage() {
  return (
    <Suspense fallback={
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800 flex items-center justify-center">
        <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600 dark:border-blue-400"></div>
      </div>
    }>
      <TeamProfileContent />
    </Suspense>
  );
}