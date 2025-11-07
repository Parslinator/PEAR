'use client';

import { useState, useEffect, Suspense } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import { Search } from 'lucide-react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || '';

interface GameData {
  week: number;
  start_date: string;
  location: string;
  opponent_team: string;
  is_completed: boolean;
  team_score: number | null;
  opponent_score: number | null;
  home_points: number | null;
  away_points: number | null;
  neutral: boolean;
  conference_game: boolean;
  opponent_pr: number | null;
  opponent_pr_rank: number | null;
  opponent_or: number | null;
  opponent_or_rank: number | null;
  opponent_dr: number | null;
  opponent_dr_rank: number | null;
  team_win_prob: number | null;
  PEAR: string | null;
  total: number | null;
  home_score: number | null;
  away_score: number | null;
}

interface ProfileData {
  team_name: string;
  schedule: GameData[];
}

const formatDate = (dateStr: string) => {
  const date = new Date(dateStr);
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
};

const formatWinProbability = (prob: number | null) => {
  if (prob === null) return 'N/A';
  return `${(prob).toFixed(1)}%`;
};

const getLocationBadge = (location: string) => {
  if (location === 'Home') return { bg: 'bg-blue-500', text: 'text-white' };
  if (location === 'Away') return { bg: 'bg-orange-500', text: 'text-white' };
  return { bg: 'bg-purple-500', text: 'text-white' };
};

function CompletedGame({ game, onGameClick }: { game: GameData; onGameClick: (game: GameData) => void }) {
  const router = useRouter();
  
  const handleOpponentClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    router.push(`/team-profile?team=${encodeURIComponent(game.opponent_team)}`);
  };

  const locationStyle = getLocationBadge(game.location);
  const didWin = game.team_score !== null && game.opponent_score !== null && game.team_score > game.opponent_score;

  return (
    <div 
      className="bg-white dark:bg-gray-700 rounded-lg p-3 border-2 border-gray-200 dark:border-gray-600 hover:shadow-md transition-shadow cursor-pointer"
      style={{ borderLeftWidth: '6px', borderLeftColor: didWin ? '#10b981' : '#ef4444' }}
      onClick={() => onGameClick(game)}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold text-gray-500 dark:text-gray-400">
            Week {game.week}
          </span>
          <span className="text-xs text-gray-500 dark:text-gray-400">
            {formatDate(game.start_date)}
          </span>
          <span className={`px-2 py-0.5 rounded text-xs font-semibold ${locationStyle.bg} ${locationStyle.text}`}>
            {game.location}
          </span>
          {game.conference_game && (
            <span className="px-2 py-0.5 rounded text-xs font-semibold bg-yellow-500 text-black">
              CONF
            </span>
          )}
        </div>
      </div>

      <div className="flex items-center justify-between gap-3">
        {/* Opponent Info */}
        <div className="flex items-center gap-3 min-w-0 flex-1">
          <img
            src={`${API_URL}/api/football-logo/${encodeURIComponent(game.opponent_team)}`}
            alt={`${game.opponent_team} logo`}
            className="w-10 h-10 object-contain cursor-pointer hover:opacity-80 transition-opacity flex-shrink-0"
            onClick={handleOpponentClick}
            onError={(e) => {
              e.currentTarget.style.display = 'none';
            }}
          />
          <div className="flex flex-col min-w-0 flex-1">
            <div className="flex items-center gap-2 flex-wrap">
              {game.opponent_pr_rank && (
                <span className="text-xs font-bold text-blue-600 dark:text-blue-400">
                  #{game.opponent_pr_rank}
                </span>
              )}
              <p className="font-medium text-sm text-gray-900 dark:text-white truncate cursor-pointer hover:text-blue-600 dark:hover:text-blue-400 transition-colors" onClick={handleOpponentClick}>
                {game.opponent_team}
              </p>
            </div>
            <div className="flex items-center gap-2 text-xs text-gray-600 dark:text-gray-400">
              {game.opponent_or_rank && <span>OR: #{game.opponent_or_rank}</span>}
              {game.opponent_dr_rank && <span>DR: #{game.opponent_dr_rank}</span>}
            </div>
          </div>
        </div>

        {/* Final Score */}
        <div className="text-right flex-shrink-0">
          <div className="text-2xl font-bold text-gray-900 dark:text-white">
            {game.team_score} - {game.opponent_score}
          </div>
          <div className="text-xs font-semibold" style={{ color: didWin ? '#10b981' : '#ef4444' }}>
            {didWin ? 'W' : 'L'}
          </div>
        </div>
      </div>
    </div>
  );
}

function UpcomingGame({ game, onGameClick }: { game: GameData; onGameClick: (game: GameData) => void }) {
  const router = useRouter();
  
  const handleOpponentClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    router.push(`/team-profile?team=${encodeURIComponent(game.opponent_team)}`);
  };

  const locationStyle = getLocationBadge(game.location);

  return (
    <div 
      className="bg-white dark:bg-gray-700 rounded-lg p-3 border-2 border-gray-200 dark:border-gray-600 hover:shadow-md transition-shadow cursor-pointer"
      onClick={() => onGameClick(game)}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold text-gray-500 dark:text-gray-400">
            Week {game.week}
          </span>
          <span className="text-xs text-gray-500 dark:text-gray-400">
            {formatDate(game.start_date)}
          </span>
          <span className={`px-2 py-0.5 rounded text-xs font-semibold ${locationStyle.bg} ${locationStyle.text}`}>
            {game.location}
          </span>
          {game.conference_game && (
            <span className="px-2 py-0.5 rounded text-xs font-semibold bg-yellow-500 text-black">
              CONF
            </span>
          )}
        </div>
        {game.PEAR && (
          <span className="px-2 py-1 bg-indigo-500 text-white rounded text-xs font-bold">
            {game.PEAR}
          </span>
        )}
      </div>

      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 sm:gap-3">
        {/* Opponent Info */}
        <div className="flex items-center gap-3 min-w-0 flex-1">
          <img
            src={`${API_URL}/api/football-logo/${encodeURIComponent(game.opponent_team)}`}
            alt={`${game.opponent_team} logo`}
            className="w-10 h-10 object-contain cursor-pointer hover:opacity-80 transition-opacity flex-shrink-0"
            onClick={handleOpponentClick}
            onError={(e) => {
              e.currentTarget.style.display = 'none';
            }}
          />
          <div className="flex flex-col min-w-0 flex-1">
            <div className="flex items-center gap-2 flex-wrap">
              {game.opponent_pr_rank && (
                <span className="text-xs font-bold text-blue-600 dark:text-blue-400">
                  #{game.opponent_pr_rank}
                </span>
              )}
              <p className="font-medium text-sm text-gray-900 dark:text-white truncate cursor-pointer hover:text-blue-600 dark:hover:text-blue-400 transition-colors" onClick={handleOpponentClick}>
                {game.opponent_team}
              </p>
            </div>
            <div className="flex items-center gap-2 text-xs text-gray-600 dark:text-gray-400">
              {game.opponent_or_rank && <span>OR: #{game.opponent_or_rank}</span>}
              {game.opponent_dr_rank && <span>DR: #{game.opponent_dr_rank}</span>}
            </div>
          </div>
        </div>

        {/* Prediction Stats */}
        <div className="flex items-center gap-2 flex-shrink-0 flex-wrap">
          <div className="flex flex-col items-center px-3 py-1 bg-gray-100 dark:bg-gray-600 rounded">
            <span className="text-xs text-gray-600 dark:text-gray-400">Win %</span>
            <span className="text-sm font-bold text-gray-900 dark:text-white">
              {formatWinProbability(game.team_win_prob)}
            </span>
          </div>
          {game.total !== null && (
            <div className="flex flex-col items-center px-3 py-1 bg-gray-100 dark:bg-gray-600 rounded">
              <span className="text-xs text-gray-600 dark:text-gray-400">O/U</span>
              <span className="text-sm font-bold text-gray-900 dark:text-white">
                {game.total.toFixed(1)}
              </span>
            </div>
          )}
          <div className="flex flex-col items-center px-3 py-1 bg-gray-100 dark:bg-gray-600 rounded">
            <span className="text-xs text-gray-600 dark:text-gray-400">Score</span>
            <span className="text-sm font-bold text-gray-900 dark:text-white">
              {game.home_score?.toFixed(1)}-{game.away_score?.toFixed(1)}
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
  
  const [profileData, setProfileData] = useState<ProfileData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [allTeams, setAllTeams] = useState<string[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [showSearchResults, setShowSearchResults] = useState(false);
  const [selectedGame, setSelectedGame] = useState<GameData | null>(null);

  useEffect(() => {
    if (teamName) {
      fetchTeamProfile(teamName);
    }
    fetchAllTeams();
  }, [teamName]);

  const fetchAllTeams = async () => {
    try {
      const response = await fetch(`${API_URL}/api/teams`);
      if (response.ok) {
        const data = await response.json();
        setAllTeams(data.teams || []);
      }
    } catch (err) {
      console.error('Error fetching teams:', err);
    }
  };

  const fetchTeamProfile = async (team: string) => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch(`${API_URL}/api/profile-page/${encodeURIComponent(team)}`);

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

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchQuery(e.target.value);
    setShowSearchResults(e.target.value.length > 0);
  };

  const handleTeamSelect = (team: string) => {
    setSearchQuery('');
    setShowSearchResults(false);
    router.push(`/team-profile?team=${encodeURIComponent(team)}`);
  };

  const handleGameClick = (game: GameData) => {
    setSelectedGame(game);
  };

  const closeModal = () => {
    setSelectedGame(null);
  };

  const filteredTeams = allTeams.filter(team => 
    team.toLowerCase().includes(searchQuery.toLowerCase())
  ).slice(0, 10);

  if (!teamName) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800 flex items-center justify-center">
        <div className="text-center">
          <p className="text-xl text-gray-600 dark:text-gray-400">No team selected</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800 pt-20">
      <div className="max-w-[1200px] mx-auto px-4 py-6">
        {/* Header with Search - Sticky */}
        <div className="sticky top-16 bg-white dark:bg-gray-800 rounded-lg shadow-lg p-4 mb-4 z-20">
          <div className="flex flex-col lg:flex-row items-start lg:items-center gap-4 justify-between">
            {/* Left side - Team Info */}
            <div className="flex items-start gap-4 flex-1 min-w-0 w-full lg:w-auto">
              <img 
                src={`${API_URL}/api/football-logo/${encodeURIComponent(teamName)}`}
                alt={`${teamName} logo`}
                className="w-12 h-12 sm:w-16 sm:h-16 object-contain flex-shrink-0"
                onError={(e) => {
                  e.currentTarget.style.display = 'none';
                }}
              />
              <div className="flex-1 min-w-0">
                <h1 className="text-2xl sm:text-3xl font-bold text-gray-900 dark:text-white">
                  {teamName}
                </h1>
              </div>
            </div>

            {/* Right side - Search Box */}
            <div className="relative w-full lg:w-64 flex-shrink-0 lg:self-center">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={18} />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={handleSearchChange}
                  onFocus={() => setShowSearchResults(searchQuery.length > 0)}
                  placeholder="Search teams..."
                  className="w-full pl-10 pr-4 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
              
              {showSearchResults && filteredTeams.length > 0 && (
                <div className="absolute top-full mt-1 w-full bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 max-h-64 overflow-y-auto z-50">
                  {filteredTeams.map((team) => (
                    <div
                      key={team}
                      onClick={() => handleTeamSelect(team)}
                      className="px-4 py-2 hover:bg-gray-100 dark:hover:bg-gray-700 cursor-pointer text-gray-900 dark:text-white text-sm"
                    >
                      {team}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Schedule */}
        {loading ? (
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-12">
            <div className="flex flex-col items-center justify-center">
              <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600 dark:border-blue-400"></div>
              <p className="mt-4 text-gray-600 dark:text-gray-400 text-lg">
                Loading schedule...
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
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg overflow-hidden">
            {profileData?.schedule && profileData.schedule.length > 0 ? (
              <>
                {(() => {
                  const completedGames = profileData.schedule.filter(game => game.is_completed);
                  const upcomingGames = profileData.schedule.filter(game => !game.is_completed);

                  return (
                    <div className="max-h-[800px] overflow-y-auto">
                      {/* Completed Games */}
                      {completedGames.length > 0 && (
                        <div>
                          <div className="sticky top-0 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 p-4 pb-2 z-10">
                            <h3 className="text-lg font-bold text-gray-900 dark:text-white">
                              Completed Games ({completedGames.length})
                            </h3>
                          </div>
                          
                          <div className="p-4 pt-2">
                            <div className="space-y-2">
                              {completedGames.map((game, index) => (
                                <CompletedGame key={index} game={game} onGameClick={handleGameClick} />
                              ))}
                            </div>
                          </div>
                        </div>
                      )}

                      {/* Upcoming Games */}
                      {upcomingGames.length > 0 && (
                        <div>
                          <div className={`sticky ${completedGames.length > 0 ? 'top-[56px]' : 'top-0'} bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 p-4 pb-2 z-10`}>
                            <h3 className="text-lg font-bold text-gray-900 dark:text-white">
                              Upcoming Games ({upcomingGames.length})
                            </h3>
                          </div>
                          
                          <div className="p-4 pt-2">
                            <div className="space-y-2">
                              {upcomingGames.map((game, index) => (
                                <UpcomingGame key={index} game={game} onGameClick={handleGameClick} />
                              ))}
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })()}
              </>
            ) : (
              <div className="p-12 text-center">
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