'use client';

import { useState, useEffect, Suspense } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import { Search, X } from 'lucide-react';

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

interface StatWithRank {
  value: number;
  rank: number;
}

interface TeamRecords {
  overall: string;
  conference: string;
  home: string;
  away: string;
  neutral: string;
}

interface TeamStats {
  power_rating: StatWithRank;
  offensive_rating: StatWithRank;
  defensive_rating: StatWithRank;
  most_deserving_wins: StatWithRank;
  sos: StatWithRank;
  kford_rating: StatWithRank;
  sp_rating: StatWithRank;
  fpi: StatWithRank;
  offense: {
    success_rate: StatWithRank;
    ppa: StatWithRank;
    rushing: StatWithRank;
    passing: StatWithRank;
    ppo: StatWithRank;
    drive_quality: StatWithRank;
  };
  defense: {
    success_rate: StatWithRank;
    ppa: StatWithRank;
    rushing: StatWithRank;
    passing: StatWithRank;
    ppo: StatWithRank;
    drive_quality: StatWithRank;
  };
}

interface ProfileData {
  team_name: string;
  schedule: GameData[];
  records: TeamRecords;
  stats: TeamStats;
}

const formatDate = (dateStr: string) => {
  const date = new Date(dateStr);
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
};

const formatWinProbability = (prob: number | null) => {
  if (prob === null) return 'N/A';
  return `${(prob).toFixed(1)}%`;
};

const getRatingColor = (value: number, allValues: number[], higherIsBetter: boolean = true) => {
  const max = Math.max(...allValues);
  const min = Math.min(...allValues);
  const range = max - min;
  let normalized = (value - min) / range;
  
  if (!higherIsBetter) {
    normalized = 1 - normalized;
  }
  
  // Colors: Dark Blue #00008B (0, 0, 139), Light Gray #D3D3D3 (211, 211, 211), Dark Red #8B0000 (139, 0, 0)
  if (normalized >= 0.5) {
    const t = (normalized - 0.5) * 2;
    const r = Math.round(211 + (0 - 211) * t);
    const g = Math.round(211 + (0 - 211) * t);
    const b = Math.round(211 + (139 - 211) * t);
    return `rgb(${r}, ${g}, ${b})`;
  } else {
    const t = normalized * 2;
    const r = Math.round(139 + (211 - 139) * t);
    const g = Math.round(0 + (211 - 0) * t);
    const b = Math.round(0 + (211 - 0) * t);
    return `rgb(${r}, ${g}, ${b})`;
  }
};

const getRankColor = (rank: number, totalTeams: number = 133) => {
  // Create array of all possible ranks for color gradient
  const allRanks = Array.from({length: totalTeams}, (_, i) => i + 1);
  return getRatingColor(rank, allRanks, false); // Lower rank is better
};

const getTextColor = (bgColor: string) => {
  const match = bgColor.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
  if (!match) return 'white';
  
  const r = parseInt(match[1]);
  const g = parseInt(match[2]);
  const b = parseInt(match[3]);
  
  const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
  
  return luminance > 0.5 ? 'black' : 'white';
};

const getStatDecimals = (statKey: string): number => {
  if (statKey.includes('rating')) return 1;
  if (statKey === 'most_deserving_wins' || statKey === 'sos' || statKey === 'fpi') return 2;
  return 2; // default
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
      className="bg-white dark:bg-gray-700 rounded-lg p-4 border-2 border-gray-200 dark:border-gray-600 hover:shadow-md transition-shadow cursor-pointer"
      style={{ borderLeftWidth: '6px', borderLeftColor: didWin ? '#10b981' : '#ef4444' }}
      onClick={() => onGameClick(game)}
    >
      {/* Header Row */}
      <div className="flex items-center justify-between mb-3 pb-2 border-b border-gray-200 dark:border-gray-600">
        <div className="flex items-center gap-2">
          <span className="text-sm font-bold text-gray-900 dark:text-white">Week {game.week}</span>
          <span className="text-xs text-gray-500 dark:text-gray-400">•</span>
          <span className="text-xs text-gray-600 dark:text-gray-400">{formatDate(game.start_date)}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className={`px-2.5 py-1 rounded-full text-xs font-bold ${locationStyle.bg} ${locationStyle.text}`}>
            {game.location === 'Home' ? 'H' : game.location === 'Away' ? 'A' : 'N'}
          </span>
          {game.conference_game && (
            <span className="px-2.5 py-1 rounded-full text-xs font-bold bg-yellow-500 text-black">C</span>
          )}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex items-center justify-between gap-4 mb-3">
        {/* Opponent Info */}
        <div className="flex items-center gap-3 min-w-0 flex-1">
          <img
            src={`${API_URL}/api/football-logo/${encodeURIComponent(game.opponent_team)}`}
            alt={`${game.opponent_team} logo`}
            className="w-12 h-12 object-contain cursor-pointer hover:opacity-80 transition-opacity flex-shrink-0"
            onClick={handleOpponentClick}
            onError={(e) => e.currentTarget.style.display = 'none'}
          />
          <div className="flex flex-col min-w-0">
            <div className="flex items-center gap-2 mb-1">
              {game.opponent_pr_rank && (
                <span className="text-xs font-bold text-blue-600 dark:text-blue-400">#{game.opponent_pr_rank}</span>
              )}
              <p className="font-bold text-base text-gray-900 dark:text-white cursor-pointer hover:text-blue-600 dark:hover:text-blue-400 transition-colors" onClick={handleOpponentClick}>
                {game.opponent_team}
              </p>
            </div>
          </div>
        </div>

        {/* Score */}
        <div className="text-right">
          <div className="text-3xl font-bold text-gray-900 dark:text-white mb-1">
            {game.team_score} - {game.opponent_score}
          </div>
          <div className="text-sm font-bold px-3 py-1 rounded-full inline-block" style={{ backgroundColor: didWin ? '#10b981' : '#ef4444', color: 'white' }}>
            {didWin ? 'WIN' : 'LOSS'}
          </div>
        </div>
      </div>

      {/* Stats Row */}
      <div className="flex items-center gap-2 flex-wrap">
        {game.opponent_or_rank && (
          <div className="flex items-center gap-1 px-2.5 py-1 bg-blue-50 dark:bg-blue-900/30 rounded-md">
            <span className="text-xs font-semibold text-blue-700 dark:text-blue-300">OFF</span>
            <span className="text-xs font-bold text-blue-900 dark:text-blue-100">#{game.opponent_or_rank}</span>
          </div>
        )}
        {game.opponent_dr_rank && (
          <div className="flex items-center gap-1 px-2.5 py-1 bg-red-50 dark:bg-red-900/30 rounded-md">
            <span className="text-xs font-semibold text-red-700 dark:text-red-300">DEF</span>
            <span className="text-xs font-bold text-red-900 dark:text-red-100">#{game.opponent_dr_rank}</span>
          </div>
        )}
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
      className="bg-white dark:bg-gray-700 rounded-lg p-4 border-2 border-blue-200 dark:border-blue-800 hover:shadow-md transition-shadow cursor-pointer"
      onClick={() => onGameClick(game)}
    >
      {/* Header Row */}
      <div className="flex items-center justify-between mb-3 pb-2 border-b border-gray-200 dark:border-gray-600">
        <div className="flex items-center gap-2">
          <span className="text-sm font-bold text-gray-900 dark:text-white">Week {game.week}</span>
          <span className="text-xs text-gray-500 dark:text-gray-400">•</span>
          <span className="text-xs text-gray-600 dark:text-gray-400">{formatDate(game.start_date)}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className={`px-2.5 py-1 rounded-full text-xs font-bold ${locationStyle.bg} ${locationStyle.text}`}>
            {game.location === 'Home' ? 'H' : game.location === 'Away' ? 'A' : 'N'}
          </span>
          {game.conference_game && (
            <span className="px-2.5 py-1 rounded-full text-xs font-bold bg-yellow-500 text-black">C</span>
          )}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex items-start justify-between gap-4 mb-3">
        {/* Opponent Info */}
        <div className="flex items-center gap-3 min-w-0 flex-1">
          <img
            src={`${API_URL}/api/football-logo/${encodeURIComponent(game.opponent_team)}`}
            alt={`${game.opponent_team} logo`}
            className="w-12 h-12 object-contain cursor-pointer hover:opacity-80 transition-opacity flex-shrink-0"
            onClick={handleOpponentClick}
            onError={(e) => e.currentTarget.style.display = 'none'}
          />
          <div className="flex flex-col min-w-0">
            <div className="flex items-center gap-2 mb-1">
              {game.opponent_pr_rank && (
                <span className="text-xs font-bold text-blue-600 dark:text-blue-400">#{game.opponent_pr_rank}</span>
              )}
              <p className="font-bold text-base text-gray-900 dark:text-white cursor-pointer hover:text-blue-600 dark:hover:text-blue-400 transition-colors" onClick={handleOpponentClick}>
                {game.opponent_team}
              </p>
            </div>
            {/* Opponent Ratings */}
            <div className="flex items-center gap-2 flex-wrap">
              {game.opponent_or_rank && (
                <div className="flex items-center gap-1 px-2 py-0.5 bg-blue-50 dark:bg-blue-900/30 rounded">
                  <span className="text-[10px] font-semibold text-blue-700 dark:text-blue-300">OFF</span>
                  <span className="text-[10px] font-bold text-blue-900 dark:text-blue-100">#{game.opponent_or_rank}</span>
                </div>
              )}
              {game.opponent_dr_rank && (
                <div className="flex items-center gap-1 px-2 py-0.5 bg-red-50 dark:bg-red-900/30 rounded">
                  <span className="text-[10px] font-semibold text-red-700 dark:text-red-300">DEF</span>
                  <span className="text-[10px] font-bold text-red-900 dark:text-red-100">#{game.opponent_dr_rank}</span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Spread */}
        {game.PEAR && (
          <div className="flex-shrink-0">
            <div className="px-3 py-2 bg-indigo-100 dark:bg-indigo-900 rounded-lg text-center">
              <div className="text-[10px] font-semibold text-indigo-700 dark:text-indigo-300 mb-0.5">SPREAD</div>
              <div className="text-lg font-bold text-indigo-900 dark:text-indigo-100">{game.PEAR}</div>
            </div>
          </div>
        )}
      </div>

      {/* Prediction Stats Row */}
      <div className="flex items-center gap-2 flex-wrap">
        {game.team_win_prob !== null && (
          <div className="flex flex-col items-center px-3 py-1.5 bg-green-50 dark:bg-green-900/20 rounded-md border border-green-200 dark:border-green-800">
            <span className="text-[10px] font-semibold text-green-700 dark:text-green-300">WIN %</span>
            <span className="text-sm font-bold text-green-900 dark:text-green-100">{formatWinProbability(game.team_win_prob)}</span>
          </div>
        )}
        {game.total !== null && (
          <div className="flex flex-col items-center px-3 py-1.5 bg-purple-50 dark:bg-purple-900/20 rounded-md border border-purple-200 dark:border-purple-800">
            <span className="text-[10px] font-semibold text-purple-700 dark:text-purple-300">O/U</span>
            <span className="text-sm font-bold text-purple-900 dark:text-purple-100">{game.total.toFixed(1)}</span>
          </div>
        )}
        {game.home_score !== null && game.away_score !== null && (
          <div className="flex flex-col items-center px-3 py-1.5 bg-gray-100 dark:bg-gray-600 rounded-md border border-gray-300 dark:border-gray-500">
            <span className="text-[10px] font-semibold text-gray-700 dark:text-gray-300">PROJ SCORE</span>
            <span className="text-sm font-bold text-gray-900 dark:text-white">{game.home_score.toFixed(1)}-{game.away_score.toFixed(1)}</span>
          </div>
        )}
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
  const [showTeamProfile, setShowTeamProfile] = useState(false);
  const [currentYear, setCurrentYear] = useState<number | null>(null);
  const [currentWeek, setCurrentWeek] = useState<number | null>(null);

  useEffect(() => {
    fetchCurrentSeason();
  }, []);

  useEffect(() => {
    if (teamName) {
      fetchTeamProfile(teamName);
    }
    fetchAllTeams();
  }, [teamName]);

  const fetchCurrentSeason = async () => {
    try {
      const response = await fetch(`${API_URL}/api/current-season`);
      if (response.ok) {
        const data = await response.json();
        setCurrentYear(data.year);
        setCurrentWeek(data.week);
      }
    } catch (err) {
      console.error('Error fetching current season:', err);
      // Fallback to defaults if API fails
      setCurrentYear(2024);
      setCurrentWeek(15);
    }
  };

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
                className="w-12 h-12 sm:w-16 sm:h-16 object-contain flex-shrink-0 cursor-pointer hover:opacity-80 transition-opacity"
                onClick={() => setShowTeamProfile(true)}
                onError={(e) => {
                  e.currentTarget.style.display = 'none';
                }}
              />
              <div className="flex-1 min-w-0">
                <div className="flex flex-col sm:flex-row sm:items-baseline gap-1 sm:gap-3">
                  <h1 className="text-2xl sm:text-3xl font-bold text-gray-900 dark:text-white">
                    {teamName}
                  </h1>
                  {!loading && profileData?.records && (
                    <span className="text-lg sm:text-xl font-semibold text-gray-600 dark:text-gray-400">
                      {profileData.records.overall} ({profileData.records.conference})
                    </span>
                  )}
                </div>
                {!loading && profileData?.records && (
                  <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs sm:text-sm">
                    <div className="flex items-center gap-1">
                      <span className="text-gray-600 dark:text-gray-400">Home:</span>
                      <span className="font-semibold text-blue-600 dark:text-blue-400">{profileData.records.home}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <span className="text-gray-600 dark:text-gray-400">Away:</span>
                      <span className="font-semibold text-orange-600 dark:text-orange-400">{profileData.records.away}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <span className="text-gray-600 dark:text-gray-400">Neutral:</span>
                      <span className="font-semibold text-purple-600 dark:text-purple-400">{profileData.records.neutral}</span>
                    </div>
                  </div>
                )}
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

        {/* Content Area - Two Column Layout */}
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
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Left Side - Team Metrics */}
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg overflow-hidden flex flex-col">
              <div className="p-4 border-b border-gray-200 dark:border-gray-700">
                <h2 className="text-lg font-bold text-gray-900 dark:text-white">Team Metrics</h2>
              </div>
              <div className="max-h-[640px] overflow-y-auto p-4">
                {profileData?.stats && (
                  <div className="space-y-6">
                    {/* Primary Ratings */}
                    <div>
                      <h3 className="text-sm font-bold text-gray-700 dark:text-gray-300 mb-3 uppercase tracking-wide">Primary Ratings</h3>
                      <div className="grid grid-cols-1 gap-2">
                        {[
                          { key: 'power_rating', label: 'PEAR', stat: profileData.stats.power_rating },
                          { key: 'kford_rating', label: 'KFord', stat: profileData.stats.kford_rating },
                          { key: 'sp_rating', label: 'SP+ Rating', stat: profileData.stats.sp_rating },
                          { key: 'fpi', label: 'FPI', stat: profileData.stats.fpi },
                          { key: 'offensive_rating', label: 'OFF', stat: profileData.stats.offensive_rating },
                          { key: 'defensive_rating', label: 'DEF', stat: profileData.stats.defensive_rating },
                          { key: 'most_deserving_wins', label: 'MD', stat: profileData.stats.most_deserving_wins },
                          { key: 'sos', label: 'SOS', stat: profileData.stats.sos }
                        ].map(({ key, label, stat }) => {
                          const rankBg = getRankColor(stat.rank);
                          const textColor = getTextColor(rankBg);
                          return (
                            <div key={key} className="flex items-center justify-between py-2 px-3 bg-gray-50 dark:bg-gray-700/30 rounded">
                              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">{label}</span>
                              <div className="flex items-center gap-2">
                                <span className="text-sm font-semibold text-gray-900 dark:text-white">{stat.value.toFixed(getStatDecimals(key))}</span>
                                <div className="px-3 py-1 rounded min-w-[3.5rem] text-center" style={{ backgroundColor: rankBg, color: textColor }}>
                                  <span className="text-xs font-bold">#{stat.rank}</span>
                                </div>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>

                    {/* Offensive & Defensive Stats - Side by Side */}
                    <div className="grid grid-cols-2 gap-4">
                      {/* Offense */}
                      <div>
                        <h3 className="text-sm font-bold text-blue-700 dark:text-blue-400 mb-3 uppercase tracking-wide">Offense</h3>
                        <div className="space-y-2">
                          {[
                            { key: 'success_rate', label: 'Success Rate', stat: profileData.stats.offense.success_rate },
                            { key: 'ppa', label: 'PPA', stat: profileData.stats.offense.ppa },
                            { key: 'rushing', label: 'Rushing', stat: profileData.stats.offense.rushing },
                            { key: 'passing', label: 'Passing', stat: profileData.stats.offense.passing },
                            { key: 'ppo', label: 'PPO', stat: profileData.stats.offense.ppo },
                            { key: 'drive_quality', label: 'Drive Quality', stat: profileData.stats.offense.drive_quality }
                          ].map(({ key, label, stat }) => {
                            const rankBg = getRankColor(stat.rank);
                            const textColor = getTextColor(rankBg);
                            return (
                              <div key={key} className="bg-gray-50 dark:bg-gray-700/30 rounded p-2">
                                <div className="text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">{label}</div>
                                <div className="flex items-center justify-between">
                                  <span className="text-sm font-semibold text-gray-900 dark:text-white">{stat.value.toFixed(2)}</span>
                                  <div className="px-2 py-0.5 rounded text-center min-w-[2.5rem]" style={{ backgroundColor: rankBg, color: textColor }}>
                                    <span className="text-xs font-bold">#{stat.rank}</span>
                                  </div>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </div>

                      {/* Defense */}
                      <div>
                        <h3 className="text-sm font-bold text-red-700 dark:text-red-400 mb-3 uppercase tracking-wide">Defense</h3>
                        <div className="space-y-2">
                          {[
                            { key: 'success_rate', label: 'Success Rate', stat: profileData.stats.defense.success_rate },
                            { key: 'ppa', label: 'PPA', stat: profileData.stats.defense.ppa },
                            { key: 'rushing', label: 'Rushing', stat: profileData.stats.defense.rushing },
                            { key: 'passing', label: 'Passing', stat: profileData.stats.defense.passing },
                            { key: 'ppo', label: 'PPO', stat: profileData.stats.defense.ppo },
                            { key: 'drive_quality', label: 'Drive Quality', stat: profileData.stats.defense.drive_quality }
                          ].map(({ key, label, stat }) => {
                            const rankBg = getRankColor(stat.rank);
                            const textColor = getTextColor(rankBg);
                            return (
                              <div key={key} className="bg-gray-50 dark:bg-gray-700/30 rounded p-2">
                                <div className="text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">{label}</div>
                                <div className="flex items-center justify-between">
                                  <span className="text-sm font-semibold text-gray-900 dark:text-white">{stat.value.toFixed(2)}</span>
                                  <div className="px-2 py-0.5 rounded text-center min-w-[2.5rem]" style={{ backgroundColor: rankBg, color: textColor }}>
                                    <span className="text-xs font-bold">#{stat.rank}</span>
                                  </div>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Right Side - Schedule */}
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg overflow-hidden">
              {profileData?.schedule && profileData.schedule.length > 0 ? (
                <>
                  {(() => {
                    const completedGames = profileData.schedule.filter(game => game.is_completed);
                    const upcomingGames = profileData.schedule.filter(game => !game.is_completed);

                    return (
                      <div className="max-h-[640px] overflow-y-auto">
                        {/* Upcoming Games */}
                        {upcomingGames.length > 0 && (
                          <div>
                            <div className="sticky top-0 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 p-4 pb-2 z-10">
                              <h3 className="text-lg font-bold text-gray-900 dark:text-white">
                                Upcoming Games ({upcomingGames.length})
                              </h3>
                            </div>
                            
                            <div className="p-4 pt-2">
                              <div className="space-y-3">
                                {upcomingGames.map((game, index) => (
                                  <UpcomingGame key={index} game={game} onGameClick={handleGameClick} />
                                ))}
                              </div>
                            </div>
                          </div>
                        )}

                        {/* Completed Games */}
                        {completedGames.length > 0 && (
                          <div>
                            <div className="sticky top-0 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 p-4 pb-2 z-10">
                              <h3 className="text-lg font-bold text-gray-900 dark:text-white">
                                Completed Games ({completedGames.length})
                              </h3>
                            </div>
                            
                            <div className="p-4 pt-2">
                              <div className="space-y-3">
                                {completedGames.map((game, index) => (
                                  <CompletedGame key={index} game={game} onGameClick={handleGameClick} />
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
          </div>
        )}

        {/* Team Profile Image Modal */}
        {showTeamProfile && currentYear && currentWeek && (
          <div 
            className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4"
            onClick={() => setShowTeamProfile(false)}
          >
            <div 
              className="relative w-full max-w-6xl max-h-[90vh] bg-white dark:bg-gray-800 rounded-lg shadow-2xl overflow-auto"
              onClick={(e) => e.stopPropagation()}
            >
              <button
                onClick={() => setShowTeamProfile(false)}
                className="absolute top-2 right-2 sm:top-4 sm:right-4 z-10 bg-white dark:bg-gray-700 rounded-full p-2 shadow-lg hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors"
              >
                <X className="w-5 h-5 sm:w-6 sm:h-6 text-gray-700 dark:text-gray-200" />
              </button>
              <div className="p-4">
                <h2 className="text-xl sm:text-2xl font-bold mb-4 text-gray-900 dark:text-white pr-10">{teamName} Profile</h2>
                <img
                  src={`${API_URL}/api/team-profile/${currentYear}/${currentWeek}/${encodeURIComponent(teamName)}`}
                  alt={`${teamName} profile`}
                  className="w-full h-auto"
                  onError={(e) => {
                    e.currentTarget.src = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="400" height="300"><rect width="400" height="300" fill="%23f3f4f6"/><text x="50%" y="50%" text-anchor="middle" fill="%23374151" font-size="18">Profile image not available</text></svg>';
                  }}
                />
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