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
  tsr: number | null;
  resume_quality: number | null;
  q1: string | null;
  q2: string | null;
  q3: string | null;
  q4: string | null;
  result: string | null;
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

const getRatingColor = (value: number, min: number, max: number, higherIsBetter: boolean = true) => {
  const range = max - min;
  let normalized = (value - min) / range;
  
  if (!higherIsBetter) {
    normalized = 1 - normalized;
  }
  
  if (normalized < 0.25) {
    const t = normalized / 0.25;
    const r = Math.round(139 + (255 - 139) * t);
    const g = Math.round(0 + (165 - 0) * t);
    const b = 0;
    return `rgb(${r}, ${g}, ${b})`;
  } else if (normalized < 0.5) {
    const t = (normalized - 0.25) / 0.25;
    const r = Math.round(255 + (211 - 255) * t);
    const g = Math.round(165 + (211 - 165) * t);
    const b = Math.round(0 + (211 - 0) * t);
    return `rgb(${r}, ${g}, ${b})`;
  } else if (normalized < 0.75) {
    const t = (normalized - 0.5) / 0.25;
    const r = Math.round(211 + (0 - 211) * t);
    const g = Math.round(211 + (255 - 211) * t);
    const b = Math.round(211 + (255 - 211) * t);
    return `rgb(${r}, ${g}, ${b})`;
  } else {
    const t = (normalized - 0.75) / 0.25;
    const r = 0;
    const g = Math.round(255 + (0 - 255) * t);
    const b = Math.round(255 + (139 - 255) * t);
    return `rgb(${r}, ${g}, ${b})`;
  }
};

const getWinProbColor = (prob: number | null) => {
  if (prob === null) return { color: 'rgb(107, 114, 128)' }; // gray
  return { color: getRatingColor(prob * 100, 0, 100, true) };
};

const getRQIColor = (rqi: number | null) => {
  if (rqi === null) return { color: 'rgb(107, 114, 128)' }; // gray
  return { color: getRatingColor(rqi, -1, 0.75, true) };
};

const getGQIColor = (gqi: number | null) => {
  if (gqi === null) return { color: 'rgb(107, 114, 128)' }; // gray
  return { color: getRatingColor(gqi, 1, 10, true) };
};

const getResultBadgeColor = (result: string | null) => {
  if (result === 'W') return 'text-gray-900';
  if (result === 'L') return 'text-white';
  return 'text-gray-700 dark:text-gray-300';
};

const getResultBadgeBackground = (result: string | null) => {
  if (result === 'W') return { backgroundColor: '#98fb98' }; // palegreen
  if (result === 'L') return { backgroundColor: '#ff7f50' }; // coral
  return { backgroundColor: '#d1d5db' }; // gray
};

const getLocationBadge = (location: string) => {
  if (location === 'Home') return 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300';
  if (location === 'Away') return 'bg-orange-100 dark:bg-orange-900/30 text-orange-700 dark:text-orange-300';
  return 'bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300';
};

function CompletedGame({ game, onGameClick }: { game: GameData; onGameClick: (game: GameData) => void }) {
  const router = useRouter();
  
  const handleOpponentClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    router.push(`/cbase/team-profile?team=${encodeURIComponent(game.opponent)}`);
  };

  return (
    <div 
      className="bg-white dark:bg-gray-700 rounded-lg p-4 border border-gray-200 dark:border-gray-600 hover:shadow-md transition-shadow cursor-pointer"
      onClick={() => onGameClick(game)}
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <span 
            className={`px-3 py-1 rounded text-sm font-bold ${getResultBadgeColor(game.result)}`}
            style={getResultBadgeBackground(game.result)}
          >
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

      <div className="flex items-center justify-between gap-6">
        {/* Opponent Info */}
        <div className="flex items-center gap-3 flex-shrink-0 w-64">
          <img
            src={`${API_URL}/api/baseball-logo/${encodeURIComponent(game.opponent)}`}
            alt={`${game.opponent} logo`}
            className="w-12 h-12 object-contain flex-shrink-0 cursor-pointer hover:opacity-80 transition-opacity"
            onClick={handleOpponentClick}
            onError={(e) => {
              e.currentTarget.style.display = 'none';
            }}
          />
          <div className="flex-1 min-w-0">
            <p className="font-semibold text-lg text-gray-900 dark:text-white truncate cursor-pointer hover:text-blue-600 dark:hover:text-blue-400 transition-colors" onClick={handleOpponentClick}>
              {game.opponent_net ? `#${game.opponent_net} ` : ''}{game.opponent}
            </p>
          </div>
        </div>

        {/* Metrics - Evenly Spaced */}
        <div className="flex items-center flex-1 justify-evenly">
          <div className="text-center">
            <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">Win Prob</p>
            <p className="text-2xl font-bold" style={getWinProbColor(game.team_win_prob)}>
              {formatWinProbability(game.team_win_prob)}
            </p>
          </div>
          <div className="text-center">
            <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">RQI</p>
            <p className="text-2xl font-bold" style={getRQIColor(game.resume_quality)}>
              {game.resume_quality?.toFixed(2) || 'N/A'}
            </p>
          </div>
          <div className="text-center">
            <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">GQI</p>
            <p className="text-2xl font-bold" style={getGQIColor(game.gqi)}>
              {game.gqi?.toFixed(1) || 'N/A'}
            </p>
          </div>
        </div>

        {/* Score */}
        <div className="text-center flex-shrink-0 w-32">
          <p className="text-3xl font-bold text-gray-900 dark:text-white whitespace-nowrap">
            {game.home_score} - {game.away_score}
          </p>
        </div>
      </div>
    </div>
  );
}

function UpcomingGame({ game, onGameClick }: { game: GameData; onGameClick: (game: GameData) => void }) {
  const router = useRouter();
  
  const handleOpponentClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    router.push(`/cbase/team-profile?team=${encodeURIComponent(game.opponent)}`);
  };

  return (
    <div 
      className="bg-white dark:bg-gray-700 rounded-lg p-4 border border-gray-200 dark:border-gray-600 hover:shadow-md transition-shadow cursor-pointer"
      onClick={() => onGameClick(game)}
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <span className="text-sm text-gray-500 dark:text-gray-400">
            {formatDate(game.date)}
          </span>
          <span className={`px-2 py-1 rounded text-xs font-semibold ${getLocationBadge(game.location)}`}>
            {game.location}
          </span>
        </div>
      </div>

      <div className="flex items-center justify-between gap-6">
        {/* Opponent Info */}
        <div className="flex items-center gap-3 flex-shrink-0 w-64">
          <img
            src={`${API_URL}/api/baseball-logo/${encodeURIComponent(game.opponent)}`}
            alt={`${game.opponent} logo`}
            className="w-12 h-12 object-contain flex-shrink-0 cursor-pointer hover:opacity-80 transition-opacity"
            onClick={handleOpponentClick}
            onError={(e) => {
              e.currentTarget.style.display = 'none';
            }}
          />
          <div className="flex-1 min-w-0">
            <p className="font-semibold text-lg text-gray-900 dark:text-white truncate cursor-pointer hover:text-blue-600 dark:hover:text-blue-400 transition-colors" onClick={handleOpponentClick}>
              {game.opponent_net ? `#${game.opponent_net} ` : ''}{game.opponent}
            </p>
          </div>
        </div>

        {/* Metrics - Evenly Spaced */}
        <div className="flex items-center flex-1 justify-evenly">
          <div className="text-center">
            <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">Win Prob</p>
            <p className="text-2xl font-bold" style={getWinProbColor(game.team_win_prob)}>
              {formatWinProbability(game.team_win_prob)}
            </p>
          </div>
          <div className="text-center">
            <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">RQI</p>
            <p className="text-2xl font-bold" style={getRQIColor(game.resume_quality)}>
              {game.resume_quality?.toFixed(2) || 'N/A'}
            </p>
          </div>
          <div className="text-center">
            <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">GQI</p>
            <p className="text-2xl font-bold" style={getGQIColor(game.gqi)}>
              {game.gqi?.toFixed(1) || 'N/A'}
            </p>
          </div>
        </div>

        {/* PEAR */}
        <div className="flex-shrink-0 w-32 flex justify-center">
          {game.pear ? (
            <span className="px-4 py-2 bg-indigo-100 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300 rounded text-lg font-bold">
              {game.pear}
            </span>
          ) : (
            <span className="text-gray-400">-</span>
          )}
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
  const [selectedGame, setSelectedGame] = useState<GameData | null>(null);
  const [matchupImageUrl, setMatchupImageUrl] = useState<string | null>(null);
  const [matchupLoading, setMatchupLoading] = useState(false);
  const [teamImageUrl, setTeamImageUrl] = useState<string | null>(null);
  const [teamImageLoading, setTeamImageLoading] = useState(false);
  const [showTeamImage, setShowTeamImage] = useState(false);

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

  const fetchMatchupImage = async (game: GameData) => {
    try {
      setMatchupLoading(true);
      const response = await fetch(`${API_URL}/api/cbase/matchup-image`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          home_team: game.home_team,
          away_team: game.away_team,
          location: game.location,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to generate matchup image');
      }

      const blob = await response.blob();
      const imageUrl = URL.createObjectURL(blob);
      setMatchupImageUrl(imageUrl);
    } catch (err) {
      console.error('Error fetching matchup image:', err);
      setMatchupImageUrl(null);
    } finally {
      setMatchupLoading(false);
    }
  };

  const fetchTeamImage = async (team: string) => {
    try {
      setTeamImageLoading(true);
      const response = await fetch(`${API_URL}/api/cbase/team-profile`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          team_name: team,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to generate team image');
      }

      const blob = await response.blob();
      const imageUrl = URL.createObjectURL(blob);
      setTeamImageUrl(imageUrl);
    } catch (err) {
      console.error('Error fetching team image:', err);
      setTeamImageUrl(null);
    } finally {
      setTeamImageLoading(false);
    }
  };

  const handleGameClick = (game: GameData) => {
    setSelectedGame(game);
    fetchMatchupImage(game);
  };

  const handleTeamLogoClick = () => {
    if (teamName) {
      setShowTeamImage(true);
      fetchTeamImage(teamName);
    }
  };

  const closeMatchupModal = () => {
    if (matchupImageUrl) {
      URL.revokeObjectURL(matchupImageUrl);
    }
    setSelectedGame(null);
    setMatchupImageUrl(null);
  };

  const closeTeamImageModal = () => {
    if (teamImageUrl) {
      URL.revokeObjectURL(teamImageUrl);
    }
    setShowTeamImage(false);
    setTeamImageUrl(null);
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
              className="w-20 h-20 object-contain cursor-pointer hover:opacity-80 transition-opacity"
              onClick={handleTeamLogoClick}
              onError={(e) => {
                e.currentTarget.style.display = 'none';
              }}
            />
            <div>
              <div className="flex items-center gap-3">
                <h1 className="text-4xl font-bold text-gray-900 dark:text-white">
                  {teamName}
                </h1>
                {!loading && profileData?.metrics?.result && (
                  <span className="text-2xl text-gray-600 dark:text-gray-400">
                    {profileData.metrics.result}
                  </span>
                )}
              </div>
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
                {/* NET Box */}
                <div className="bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 rounded-lg p-4">
                  <div className="flex items-center gap-3 mb-2">
                    <Trophy className="text-blue-600 dark:text-blue-400" size={20} />
                    <h3 className="font-semibold text-sm text-gray-700 dark:text-gray-300">
                      NET
                    </h3>
                  </div>
                  <p className="text-3xl font-bold text-gray-900 dark:text-white">
                    #{profileData?.metrics?.net || 'N/A'}
                  </p>
                  {profileData?.metrics?.net_score && (
                    <p className="text-lg text-gray-600 dark:text-gray-400 mt-1">
                      {profileData.metrics.net_score.toFixed(3)}
                    </p>
                  )}
                </div>

                {/* TSR Box */}
                <div className="bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 rounded-lg p-4">
                  <div className="flex items-center gap-3 mb-2">
                    <TrendingUp className="text-purple-600 dark:text-purple-400" size={20} />
                    <h3 className="font-semibold text-sm text-gray-700 dark:text-gray-300">
                      TSR
                    </h3>
                  </div>
                  <p className="text-3xl font-bold text-gray-900 dark:text-white">
                    #{profileData?.metrics?.tsr || 'N/A'}
                  </p>
                  {profileData?.metrics?.rating && (
                    <p className="text-lg text-gray-600 dark:text-gray-400 mt-1">
                      {profileData.metrics.rating.toFixed(2)}
                    </p>
                  )}
                </div>

                {/* Rankings Box */}
                <div className="bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 rounded-lg p-4">
                  <div className="flex items-center gap-3 mb-3">
                    <Calendar className="text-green-600 dark:text-green-400" size={20} />
                    <h3 className="font-semibold text-sm text-gray-700 dark:text-gray-300">
                      Rankings
                    </h3>
                  </div>
                  <div className="space-y-1">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-600 dark:text-gray-400">ELO:</span>
                      <span className="text-lg font-bold text-gray-900 dark:text-white">
                        #{profileData?.metrics?.elo_rank || 'N/A'}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-600 dark:text-gray-400">RPI:</span>
                      <span className="text-lg font-bold text-gray-900 dark:text-white">
                        #{profileData?.metrics?.rpi || 'N/A'}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Quadrant Record Box */}
                <div className="bg-gradient-to-br from-orange-50 to-orange-100 dark:from-orange-900/20 dark:to-orange-800/20 rounded-lg p-4">
                  <div className="flex items-center gap-3 mb-3">
                    <Trophy className="text-orange-600 dark:text-orange-400" size={20} />
                    <h3 className="font-semibold text-sm text-gray-700 dark:text-gray-300">
                      Quadrant Record
                    </h3>
                  </div>
                  <div className="space-y-1">
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Q1:</span>
                      <span className="text-sm font-bold text-gray-900 dark:text-white">
                        {profileData?.metrics?.q1 || 'N/A'}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Q2:</span>
                      <span className="text-sm font-bold text-gray-900 dark:text-white">
                        {profileData?.metrics?.q2 || 'N/A'}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Q3:</span>
                      <span className="text-sm font-bold text-gray-900 dark:text-white">
                        {profileData?.metrics?.q3 || 'N/A'}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Q4:</span>
                      <span className="text-sm font-bold text-gray-900 dark:text-white">
                        {profileData?.metrics?.q4 || 'N/A'}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Schedule Section with Scroll */}
            {profileData?.schedule && profileData.schedule.length > 0 ? (
              <>
                {(() => {
                  const completedGames = profileData.schedule.filter(game => game.result !== null);
                  const upcomingGames = profileData.schedule.filter(game => game.result === null);

                  return (
                    <>
                      {/* Completed Games */}
                      {completedGames.length > 0 && (
                        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg overflow-hidden">
                          <div className="flex items-center gap-3 p-6 pb-4">
                            <Trophy className="text-blue-600 dark:text-blue-400" size={24} />
                            <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
                              Completed Games ({completedGames.length})
                            </h2>
                          </div>
                          <div className="overflow-y-auto max-h-[600px] px-6 pb-6">
                            <div className="space-y-3">
                              {completedGames.map((game, index) => (
                                <CompletedGame key={index} game={game} onGameClick={handleGameClick} />
                              ))}
                            </div>
                          </div>
                        </div>
                      )}

                      {/* Upcoming Games */}
                      {upcomingGames.length > 0 && (
                        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg overflow-hidden">
                          <div className="flex items-center gap-3 p-6 pb-4">
                            <Calendar className="text-green-600 dark:text-green-400" size={24} />
                            <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
                              Upcoming Games ({upcomingGames.length})
                            </h2>
                          </div>
                          <div className="overflow-y-auto max-h-[600px] px-6 pb-6">
                            <div className="space-y-3">
                              {upcomingGames.map((game, index) => (
                                <UpcomingGame key={index} game={game} onGameClick={handleGameClick} />
                              ))}
                            </div>
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

        {/* Matchup Image Modal */}
        {selectedGame && (
          <div 
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
            onClick={closeMatchupModal}
          >
            <div 
              className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-5xl w-full max-h-[90vh] overflow-y-auto"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="sticky top-0 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 p-4 flex justify-between items-center">
                <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
                  {selectedGame.home_team} vs {selectedGame.away_team}
                </h2>
                <button
                  onClick={closeMatchupModal}
                  className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 text-2xl font-bold"
                >
                  ×
                </button>
              </div>
              
              <div className="p-6">
                {matchupLoading ? (
                  <div className="text-center py-12">
                    <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 dark:border-blue-400"></div>
                    <p className="mt-4 text-gray-600 dark:text-gray-400">Loading matchup image...</p>
                  </div>
                ) : matchupImageUrl ? (
                  <div className="flex justify-center">
                    <img 
                      src={matchupImageUrl} 
                      alt="Matchup analysis"
                      className="max-w-full h-auto rounded-lg"
                    />
                  </div>
                ) : (
                  <p className="text-center text-gray-600 dark:text-gray-400">Failed to load matchup image</p>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Team Image Modal */}
        {showTeamImage && (
          <div 
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
            onClick={closeTeamImageModal}
          >
            <div 
              className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-5xl w-full max-h-[90vh] overflow-y-auto"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="sticky top-0 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 p-4 flex justify-between items-center">
                <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
                  {teamName} Profile
                </h2>
                <button
                  onClick={closeTeamImageModal}
                  className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 text-2xl font-bold"
                >
                  ×
                </button>
              </div>
              
              <div className="p-6">
                {teamImageLoading ? (
                  <div className="text-center py-12">
                    <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 dark:border-blue-400"></div>
                    <p className="mt-4 text-gray-600 dark:text-gray-400">Loading team profile...</p>
                  </div>
                ) : teamImageUrl ? (
                  <div className="flex justify-center">
                    <img 
                      src={teamImageUrl} 
                      alt={`${teamName} profile`}
                      className="max-w-full h-auto rounded-lg"
                    />
                  </div>
                ) : (
                  <p className="text-center text-gray-600 dark:text-gray-400">Failed to load team profile image</p>
                )}
              </div>
            </div>
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