'use client';

import { useState, useEffect, Suspense } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import { Search } from 'lucide-react';

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

interface StatWithRank {
  value: number;
  rank: number;
}

interface OffenseStats {
  rpg: StatWithRank;
  ba: StatWithRank;
  obp: StatWithRank;
  slg: StatWithRank;
  ops: StatWithRank;
  iso: StatWithRank;
  woba: StatWithRank;
}

interface PitchingStats {
  era: StatWithRank;
  whip: StatWithRank;
  kp7: StatWithRank;
  lob: StatWithRank;
  kbb: StatWithRank;
  fip: StatWithRank;
  pct: StatWithRank;
}

interface TeamStats {
  net: StatWithRank;
  tsr: StatWithRank;
  rqi: StatWithRank;
  sos: StatWithRank;
  war: StatWithRank;
  wpoe: StatWithRank;
  pythag: StatWithRank;
}

interface LocationRecords {
  home: string;
  away: string;
  neutral: string;
}

interface TeamMetrics {
  conference: string | null;
  rating: number | null;
  tsr: number | null;
  net: number | null;
  net_score: number | null;
  rpi: number | null;
  elo: number | null;
  elo_rank: number | null;
  resume_quality: number | null;
  record: string | null;
  q1: string | null;
  q2: string | null;
  q3: string | null;
  q4: string | null;
  offense?: OffenseStats;
  pitching?: PitchingStats;
  team_stats?: TeamStats;
  location_records?: LocationRecords;
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

const getTextColor = (bgColor: string) => {
  const match = bgColor.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
  if (!match) return 'white';
  
  const r = parseInt(match[1]);
  const g = parseInt(match[2]);
  const b = parseInt(match[3]);
  
  const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
  return luminance > 0.5 ? 'black' : 'white';
};

const getWinProbColor = (prob: number | null) => {
  if (prob === null) return 'rgb(107, 114, 128)';
  return getRatingColor(prob * 100, 0, 100, true);
};

const getRQIColor = (rqi: number | null) => {
  if (rqi === null) return 'rgb(107, 114, 128)';
  return getRatingColor(rqi, -1, 0.75, true);
};

const getGQIColor = (gqi: number | null) => {
  if (gqi === null) return 'rgb(107, 114, 128)';
  return getRatingColor(gqi, 1, 10, true);
};

const getRankColor = (rank: number) => {
  return getRatingColor(rank, 1, 300, false);
};
const getStatDecimals = (statKey: string): number => {
  const key = statKey.toLowerCase();
  
  // 1 decimal
  if (key === 'kp7') return 1;
  
  // 2 decimals
  if (['rating', 'rpg', 'era', 'whip', 'kbb'].includes(key)) return 2;
  
  // 3 decimals (default)
  return 3;
};


const getResultBadgeColor = (result: string | null) => {
  if (!result) return { backgroundColor: '#d1d5db', color: '#374151' };
  if (result.startsWith('W')) return { backgroundColor: '#98fb98', color: '#000000' };
  if (result.startsWith('L')) return { backgroundColor: '#ff7f50', color: '#ffffff' };
  return { backgroundColor: '#d1d5db', color: '#374151' };
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
    router.push(`/csoft/team-profile?team=${encodeURIComponent(game.opponent)}`);
  };

  const winProbBg = getWinProbColor(game.team_win_prob);
  const rqiBg = getRQIColor(game.resume_quality);
  const gqiBg = getGQIColor(game.gqi);
  const locationStyle = getLocationBadge(game.location);
  const resultStyle = getResultBadgeColor(game.result);

  return (
    <div 
      className="bg-white dark:bg-gray-700 rounded-lg p-3 border border-gray-200 dark:border-gray-600 hover:shadow-md transition-shadow cursor-pointer"
      onClick={() => onGameClick(game)}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span 
            className="px-2 py-1 rounded text-xs font-bold"
            style={resultStyle}
          >
            {game.result}
          </span>
          <span className="text-xs text-gray-500 dark:text-gray-400">
            {formatDate(game.date)}
          </span>
          <span className={`px-2 py-0.5 rounded text-xs font-semibold ${locationStyle.bg} ${locationStyle.text}`}>
            {game.location}
          </span>
        </div>
      </div>

      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 sm:gap-3">
        {/* Opponent Info */}
        <div className="flex items-center gap-2 min-w-0 flex-shrink-0">
          <img
            src={`${API_URL}/api/softball-logo/${encodeURIComponent(game.opponent)}`}
            alt={`${game.opponent} logo`}
            className="w-8 h-8 object-contain cursor-pointer hover:opacity-80 transition-opacity"
            onClick={handleOpponentClick}
            onError={(e) => {
              e.currentTarget.style.display = 'none';
            }}
          />
          <p className="font-medium text-sm text-gray-900 dark:text-white truncate cursor-pointer hover:text-blue-600 dark:hover:text-blue-400 transition-colors" onClick={handleOpponentClick}>
            {game.opponent_net ? `#${game.opponent_net} ` : ''}{game.opponent}
          </p>
        </div>

        {/* Metrics */}
        <div className="flex items-center gap-2 flex-shrink-0">
          <div className="px-2 py-1 rounded w-16" style={{ backgroundColor: winProbBg, color: getTextColor(winProbBg) }}>
            <p className="text-sm font-bold whitespace-nowrap text-center">
              {formatWinProbability(game.team_win_prob)}
            </p>
          </div>
          <div className="px-2 py-1 rounded w-16" style={{ backgroundColor: rqiBg, color: getTextColor(rqiBg) }}>
            <p className="text-sm font-bold whitespace-nowrap text-center">
              {game.resume_quality?.toFixed(2) || 'N/A'}
            </p>
          </div>
          <div className="px-2 py-1 rounded w-16" style={{ backgroundColor: gqiBg, color: getTextColor(gqiBg) }}>
            <p className="text-sm font-bold whitespace-nowrap text-center">
              {game.gqi?.toFixed(1) || 'N/A'}
            </p>
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
    router.push(`/csoft/team-profile?team=${encodeURIComponent(game.opponent)}`);
  };

  const winProbBg = getWinProbColor(game.team_win_prob);
  const rqiBg = getRQIColor(game.resume_quality);
  const gqiBg = getGQIColor(game.gqi);
  const locationStyle = getLocationBadge(game.location);

  return (
    <div 
      className="bg-white dark:bg-gray-700 rounded-lg p-3 border border-gray-200 dark:border-gray-600 hover:shadow-md transition-shadow cursor-pointer"
      onClick={() => onGameClick(game)}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-500 dark:text-gray-400">
            {formatDate(game.date)}
          </span>
          <span className={`px-2 py-0.5 rounded text-xs font-semibold ${locationStyle.bg} ${locationStyle.text}`}>
            {game.location}
          </span>
        </div>
        {game.pear && (
          <span className="px-2 py-1 bg-indigo-500 text-white rounded text-xs font-bold">
            {game.pear}
          </span>
        )}
      </div>

      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 sm:gap-3">
        {/* Opponent Info */}
        <div className="flex items-center gap-2 min-w-0 flex-shrink-0">
          <img
            src={`${API_URL}/api/softball-logo/${encodeURIComponent(game.opponent)}`}
            alt={`${game.opponent} logo`}
            className="w-8 h-8 object-contain cursor-pointer hover:opacity-80 transition-opacity"
            onClick={handleOpponentClick}
            onError={(e) => {
              e.currentTarget.style.display = 'none';
            }}
          />
          <p className="font-medium text-sm text-gray-900 dark:text-white truncate cursor-pointer hover:text-blue-600 dark:hover:text-blue-400 transition-colors" onClick={handleOpponentClick}>
            {game.opponent_net ? `#${game.opponent_net} ` : ''}{game.opponent}
          </p>
        </div>

        {/* Metrics */}
        <div className="flex items-center gap-2 flex-shrink-0">
          <div className="px-2 py-1 rounded w-16" style={{ backgroundColor: winProbBg, color: getTextColor(winProbBg) }}>
            <p className="text-sm font-bold whitespace-nowrap text-center">
              {formatWinProbability(game.team_win_prob)}
            </p>
          </div>
          <div className="px-2 py-1 rounded w-16" style={{ backgroundColor: rqiBg, color: getTextColor(rqiBg) }}>
            <p className="text-sm font-bold whitespace-nowrap text-center">
              {game.resume_quality?.toFixed(2) || 'N/A'}
            </p>
          </div>
          <div className="px-2 py-1 rounded w-16" style={{ backgroundColor: gqiBg, color: getTextColor(gqiBg) }}>
            <p className="text-sm font-bold whitespace-nowrap text-center">
              {game.gqi?.toFixed(1) || 'N/A'}
            </p>
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
  const [selectedGame, setSelectedGame] = useState<GameData | null>(null);
  const [matchupImageUrl, setMatchupImageUrl] = useState<string | null>(null);
  const [matchupLoading, setMatchupLoading] = useState(false);
  const [teamImageUrl, setTeamImageUrl] = useState<string | null>(null);
  const [teamImageLoading, setTeamImageLoading] = useState(false);
  const [showTeamImage, setShowTeamImage] = useState(false);
  const [allTeams, setAllTeams] = useState<string[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [showSearchResults, setShowSearchResults] = useState(false);

  useEffect(() => {
    if (teamName) {
      fetchTeamProfile(teamName);
    }
    fetchAllTeams();
  }, [teamName]);

  const fetchAllTeams = async () => {
    try {
      const response = await fetch(`${API_URL}/api/softball/teams`);
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
      
      const response = await fetch(`${API_URL}/api/softball/profile-page`, {
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
      const response = await fetch(`${API_URL}/api/softball/matchup-image`, {
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
      const response = await fetch(`${API_URL}/api/softball/team-profile`, {
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

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchQuery(e.target.value);
    setShowSearchResults(e.target.value.length > 0);
  };

  const handleTeamSelect = (team: string) => {
    setSearchQuery('');
    setShowSearchResults(false);
    router.push(`/csoft/team-profile?team=${encodeURIComponent(team)}`);
  };

  const filteredTeams = allTeams.filter(team => 
    team.toLowerCase().includes(searchQuery.toLowerCase())
  ).slice(0, 10);

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
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800 pt-14">
      <div className="max-w-[1600px] mx-auto px-4 py-6">

        {/* Header with Search */}
        <div className="sticky top-16 bg-white dark:bg-gray-800 rounded-lg shadow-lg p-4 mb-4 z-20">
          <div className="flex flex-col lg:flex-row items-start lg:items-center gap-4 justify-between">
            {/* Left side - Team Info (50%) */}
            <div className="flex items-start gap-4 flex-1 min-w-0 w-full lg:w-auto">
              <img 
                src={`${API_URL}/api/softball-logo/${encodeURIComponent(teamName)}`}
                alt={`${teamName} logo`}
                className="w-12 h-12 sm:w-16 sm:h-16 object-contain cursor-pointer hover:opacity-80 transition-opacity flex-shrink-0"
                onClick={handleTeamLogoClick}
                onError={(e) => {
                  e.currentTarget.style.display = 'none';
                }}
              />
              <div className="flex-1 min-w-0">
                <div className="flex flex-col sm:flex-row sm:items-baseline gap-1 sm:gap-3 flex-wrap">
                  <h1 className="text-2xl sm:text-3xl font-bold text-gray-900 dark:text-white">
                    {teamName}
                  </h1>
                  {!loading && profileData?.metrics?.record && (
                    <span className="text-lg sm:text-xl font-semibold text-gray-600 dark:text-gray-400">
                      {profileData.metrics.record}
                    </span>
                  )}
                </div>
                
                {!loading && profileData?.metrics && (
                  <div className="mt-2 space-y-2">
                    <div className="grid grid-cols-2 sm:flex sm:flex-wrap gap-x-3 sm:gap-x-6 gap-y-1 text-xs sm:text-sm">
                      <div>
                        <span className="text-gray-600 dark:text-gray-400">NET: </span>
                        <span className="font-bold text-gray-900 dark:text-white">#{profileData.metrics.net || 'N/A'}</span>
                      </div>
                      <div>
                        <span className="text-gray-600 dark:text-gray-400">ELO: </span>
                        <span className="font-bold text-gray-900 dark:text-white">#{profileData.metrics.elo_rank || 'N/A'}</span>
                      </div>
                      <div>
                        <span className="text-gray-600 dark:text-gray-400">RPI: </span>
                        <span className="font-bold text-gray-900 dark:text-white">#{profileData.metrics.rpi || 'N/A'}</span>
                      </div>
                      <div>
                        <span className="text-gray-600 dark:text-gray-400">Q1: </span>
                        <span className="font-semibold text-gray-900 dark:text-white">{profileData.metrics.q1 || 'N/A'}</span>
                      </div>
                      <div>
                        <span className="text-gray-600 dark:text-gray-400">Q2: </span>
                        <span className="font-semibold text-gray-900 dark:text-white">{profileData.metrics.q2 || 'N/A'}</span>
                      </div>
                      <div>
                        <span className="text-gray-600 dark:text-gray-400">Q3: </span>
                        <span className="font-semibold text-gray-900 dark:text-white">{profileData.metrics.q3 || 'N/A'}</span>
                      </div>
                      <div>
                        <span className="text-gray-600 dark:text-gray-400">Q4: </span>
                        <span className="font-semibold text-gray-900 dark:text-white">{profileData.metrics.q4 || 'N/A'}</span>
                      </div>
                    </div>
                    {profileData.metrics.location_records && (
                      <div className="flex flex-wrap gap-2 text-xs sm:text-sm">
                        <div className="flex items-center gap-1">
                          <span className="text-gray-600 dark:text-gray-400">Home: </span>
                          <span className="font-semibold text-blue-600 dark:text-blue-400">{profileData.metrics.location_records.home}</span>
                        </div>
                        <div className="flex items-center gap-1">
                          <span className="text-gray-600 dark:text-gray-400">Away: </span>
                          <span className="font-semibold text-orange-600 dark:text-orange-400">{profileData.metrics.location_records.away}</span>
                        </div>
                        <div className="flex items-center gap-1">
                          <span className="text-gray-600 dark:text-gray-400">Neutral: </span>
                          <span className="font-semibold text-purple-600 dark:text-purple-400">{profileData.metrics.location_records.neutral}</span>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>

            {/* Right side - Search Box (50%) */}
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
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Left Side - Team Metrics */}
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 h-[585px] flex flex-col overflow-hidden">
              <h2 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
                Team Metrics
              </h2>
              <div className="flex-1 overflow-y-auto">
                {profileData?.metrics && (
                  <div className="space-y-6">
                    {/* Team Stats Section */}
                    {profileData.metrics.team_stats && (
                      <div>
                        <h3 className="text-sm font-bold text-gray-700 dark:text-gray-300 mb-3 uppercase tracking-wide">
                          Team Statistics
                        </h3>
                        <div className="grid grid-cols-1 gap-2">
                          {Object.entries(profileData.metrics.team_stats).map(([key, stat]) => {
                            const statName = key.toUpperCase().replaceAll('_', ' ');
                            const rankBg = getRankColor(stat.rank);
                            return (
                              <div key={key} className="flex items-center justify-between py-1.5 px-3 bg-gray-50 dark:bg-gray-700/30 rounded">
                                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">{statName}</span>
                                <div className="flex items-center gap-2">
                                  <span className="text-sm font-semibold text-gray-900 dark:text-white">{stat.value.toFixed(getStatDecimals(key))}</span>
                                  <div className="px-2 py-1 rounded min-w-[3rem] text-center" style={{ backgroundColor: rankBg, color: getTextColor(rankBg) }}>
                                    <span className="text-xs font-bold">#{stat.rank}</span>
                                  </div>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}

                    {/* Offense and Pitching Stats - Side by Side */}
                    <div className="grid grid-cols-2 gap-4">
                      {/* Offense Stats */}
                      {profileData.metrics.offense && (
                        <div>
                          <h3 className="text-sm font-bold text-gray-700 dark:text-gray-300 mb-3 uppercase tracking-wide">
                            Offense
                          </h3>
                          <div className="space-y-2">
                            {Object.entries(profileData.metrics.offense).map(([key, stat]) => {
                              const statName = key.toUpperCase();
                              const rankBg = getRankColor(stat.rank);
                              return (
                                <div key={key} className="flex flex-col gap-1 py-1.5 px-2 bg-gray-50 dark:bg-gray-700/30 rounded">
                                  <div className="flex items-center justify-between">
                                    <span className="text-xs font-medium text-gray-700 dark:text-gray-300">{statName}</span>
                                    <span className="text-xs font-semibold text-gray-900 dark:text-white">{stat.value.toFixed(getStatDecimals(key))}</span>
                                  </div>
                                  <div className="px-1.5 py-0.5 rounded text-center" style={{ backgroundColor: rankBg, color: getTextColor(rankBg) }}>
                                    <span className="text-xs font-bold">#{stat.rank}</span>
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      )}

                      {/* Pitching Stats */}
                      {profileData.metrics.pitching && (
                        <div>
                          <h3 className="text-sm font-bold text-gray-700 dark:text-gray-300 mb-3 uppercase tracking-wide">
                            Pitching
                          </h3>
                          <div className="space-y-2">
                            {Object.entries(profileData.metrics.pitching).map(([key, stat]) => {
                              const statName = key.toUpperCase();
                              const rankBg = getRankColor(stat.rank);
                              return (
                                <div key={key} className="flex flex-col gap-1 py-1.5 px-2 bg-gray-50 dark:bg-gray-700/30 rounded">
                                  <div className="flex items-center justify-between">
                                    <span className="text-xs font-medium text-gray-700 dark:text-gray-300">{statName}</span>
                                    <span className="text-xs font-semibold text-gray-900 dark:text-white">{stat.value.toFixed(getStatDecimals(key))}</span>
                                  </div>
                                  <div className="px-1.5 py-0.5 rounded text-center" style={{ backgroundColor: rankBg, color: getTextColor(rankBg) }}>
                                    <span className="text-xs font-bold">#{stat.rank}</span>
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      )}
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
                    const completedGames = profileData.schedule.filter(game => game.result !== null);
                    const upcomingGames = profileData.schedule.filter(game => game.result === null);

                    return (
                      <div className="h-[585px] overflow-y-auto">
                        {/* Completed Games */}
                        {completedGames.length > 0 && (
                          <div>
                            {/* Sticky Header */}
                            <div className="sticky top-0 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 p-4 pb-2 z-10">
                              <div className="flex items-center justify-between">
                                <h3 className="text-lg font-bold text-gray-900 dark:text-white">
                                  Completed Games (Win Prob, Win Quality, Game Quality)
                                </h3>
                              </div>
                            </div>
                            
                            {/* Games List */}
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
                            {/* Sticky Header */}
                            <div className={`sticky ${completedGames.length > 0 ? 'top-[72px]' : 'top-0'} bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 p-4 pb-2 z-10`}>
                              <div className="flex items-center justify-between">
                                <h3 className="text-lg font-bold text-gray-900 dark:text-white">
                                  Upcoming Games ({upcomingGames.length})
                                </h3>
                                <div className="flex items-center gap-2">
                                  <span className="text-xs font-semibold text-gray-600 dark:text-gray-400 w-16 text-center px-2">Win Prob</span>
                                  <span className="text-xs font-semibold text-gray-600 dark:text-gray-400 w-16 text-center px-2">RQI</span>
                                  <span className="text-xs font-semibold text-gray-600 dark:text-gray-400 w-16 text-center px-2">GQI</span>
                                </div>
                              </div>
                            </div>
                            
                            {/* Games List */}
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