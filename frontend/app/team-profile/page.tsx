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
  return `${prob}%`;
};

const getLocationBadge = (location: string) => {
  if (location === 'Home') return { bg: 'bg-blue-500', text: 'text-white' };
  if (location === 'Away') return { bg: 'bg-orange-500', text: 'text-white' };
  return { bg: 'bg-purple-500', text: 'text-white' };
};

function GameCard({ game, onGameClick }: { game: GameData; onGameClick: (game: GameData) => void }) {
  const router = useRouter();
  
  const handleOpponentClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    router.push(`/team-profile?team=${encodeURIComponent(game.opponent_team)}`);
  };

  const locationStyle = getLocationBadge(game.location);
  const didWin = game.team_score !== null && game.opponent_score !== null && game.team_score > game.opponent_score;
  const isFavored = game.team_win_prob !== null && game.team_win_prob > 50;

  return (
    <div 
      className="bg-white dark:bg-gray-700 rounded-lg p-4 border-2 border-gray-200 dark:border-gray-600 hover:shadow-md transition-shadow cursor-pointer"
      onClick={() => onGameClick(game)}
    >
      <div className="flex items-center justify-between mb-3">
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

      <div className="flex items-center justify-between gap-4">
        {/* Opponent Info */}
        <div className="flex items-center gap-3 min-w-0 flex-1">
          <img
            src={`${API_URL}/api/football-logo/${encodeURIComponent(game.opponent_team)}`}
            alt={`${game.opponent_team} logo`}
            className="w-12 h-12 object-contain cursor-pointer hover:opacity-80 transition-opacity flex-shrink-0"
            onClick={handleOpponentClick}
            onError={(e) => {
              e.currentTarget.style.display = 'none';
            }}
          />
          <div className="flex flex-col min-w-0 flex-1">
            <div className="flex items-center gap-2 flex-wrap mb-1">
              {game.opponent_pr_rank && (
                <span className="text-base font-extrabold text-blue-600 dark:text-blue-400">
                  #{game.opponent_pr_rank}
                </span>
              )}
              <p className="font-bold text-lg text-gray-900 dark:text-white truncate cursor-pointer hover:text-blue-600 dark:hover:text-blue-400 transition-colors" onClick={handleOpponentClick}>
                {game.opponent_team}
              </p>
            </div>
            <div className="flex items-center gap-3 text-sm font-semibold text-gray-600 dark:text-gray-400">
              {game.opponent_or_rank && <span className="text-orange-600 dark:text-orange-400">OR: #{game.opponent_or_rank}</span>}
              {game.opponent_dr_rank && <span className="text-red-600 dark:text-red-400">DR: #{game.opponent_dr_rank}</span>}
            </div>
          </div>
        </div>

        {/* Score/Prediction */}
        <div className="text-right flex-shrink-0">
          {game.is_completed ? (
            <>
              <div className="text-2xl font-bold text-gray-900 dark:text-white mb-1">
                {game.team_score} - {game.opponent_score}
              </div>
              <div className={`inline-block px-3 py-1 rounded font-bold text-sm ${didWin ? 'bg-green-500 text-white' : 'bg-red-500 text-white'}`}>
                {didWin ? 'W' : 'L'}
              </div>
            </>
          ) : (
            <>
              {game.home_score !== null && game.away_score !== null ? (
                <div className="text-lg font-semibold text-gray-700 dark:text-gray-300 mb-1">
                  {game.location === 'Home' ? game.home_score : game.away_score} - {game.location === 'Home' ? game.away_score : game.home_score}
                </div>
              ) : null}
              {game.team_win_prob !== null && (
                <div className={`inline-block px-3 py-1 rounded font-bold text-sm ${isFavored ? 'bg-green-500 text-white' : 'bg-red-500 text-white'}`}>
                  {formatWinProbability(game.team_win_prob)}
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function TeamProfileContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const teamName = searchParams?.get('team');
  
  const [profileData, setProfileData] = useState<ProfileData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedGame, setSelectedGame] = useState<GameData | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [showSearchResults, setShowSearchResults] = useState(false);

  const allTeams = [
    "Air Force", "Akron", "Alabama", "Appalachian State", "Arizona", "Arizona State",
    "Arkansas", "Arkansas State", "Army", "Auburn", "Ball State", "Baylor",
    "Boise State", "Boston College", "Bowling Green", "Buffalo", "BYU", "California",
    "Central Michigan", "Charlotte", "Cincinnati", "Clemson", "Coastal Carolina",
    "Colorado", "Colorado State", "Connecticut", "Duke", "East Carolina",
    "Eastern Michigan", "Florida", "Florida Atlantic", "Florida International",
    "Florida State", "Fresno State", "Georgia", "Georgia Southern", "Georgia State",
    "Georgia Tech", "Hawai'i", "Houston", "Illinois", "Indiana", "Iowa",
    "Iowa State", "James Madison", "Kansas", "Kansas State", "Kent State",
    "Kentucky", "Liberty", "Louisiana", "Louisiana Monroe", "Louisiana Tech",
    "Louisville", "LSU", "Marshall", "Maryland", "Memphis", "Miami", "Miami (OH)",
    "Michigan", "Michigan State", "Middle Tennessee", "Minnesota", "Mississippi State",
    "Missouri", "Navy", "Nebraska", "Nevada", "UNLV", "New Mexico", "New Mexico State",
    "North Carolina", "North Carolina State", "North Texas", "Northern Illinois",
    "Northwestern", "Notre Dame", "Ohio", "Ohio State", "Oklahoma", "Oklahoma State",
    "Old Dominion", "Ole Miss", "Oregon", "Oregon State", "Penn State", "Pittsburgh",
    "Purdue", "Rice", "Rutgers", "Sam Houston", "San Diego State", "San José State",
    "SMU", "South Alabama", "South Carolina", "South Florida", "Southern Mississippi",
    "Stanford", "Syracuse", "TCU", "Temple", "Tennessee", "Texas", "Texas A&M",
    "Texas State", "Texas Tech", "Toledo", "Troy", "Tulane", "Tulsa", "UAB",
    "UCF", "UCLA", "UMass", "USC", "Utah", "Utah State", "UTEP", "UTSA",
    "Vanderbilt", "Virginia", "Virginia Tech", "Wake Forest", "Washington",
    "Washington State", "West Virginia", "Western Kentucky", "Western Michigan",
    "Wisconsin", "Wyoming"
  ];

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as HTMLElement;
      if (!target.closest('.search-container')) {
        setShowSearchResults(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  useEffect(() => {
    if (teamName) {
      loadTeamProfile(teamName);
    }
  }, [teamName]);

  const loadTeamProfile = async (team: string) => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(
        `${API_URL}/api/team-profile?team=${encodeURIComponent(team)}`
      );

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
            <div className="relative w-full lg:w-64 flex-shrink-0 lg:self-center search-container">
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
              <div className="max-h-[800px] overflow-y-auto p-4">
                <div className="space-y-3">
                  {profileData.schedule.map((game, index) => (
                    <GameCard key={index} game={game} onGameClick={handleGameClick} />
                  ))}
                </div>
              </div>
            ) : (
              <div className="p-12 text-center">
                <p className="text-gray-600 dark:text-gray-400">No schedule data available</p>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Modal */}
      {selectedGame && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
          onClick={closeModal}
        >
          <div 
            className="bg-white dark:bg-gray-800 rounded-lg p-6 max-w-2xl w-full max-h-[90vh] overflow-y-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex justify-between items-start mb-4">
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
                Week {selectedGame.week} - {selectedGame.opponent_team}
              </h2>
              <button
                onClick={closeModal}
                className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 text-2xl"
              >
                ×
              </button>
            </div>

            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Date</p>
                  <p className="font-semibold text-gray-900 dark:text-white">{formatDate(selectedGame.start_date)}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Location</p>
                  <p className="font-semibold text-gray-900 dark:text-white">{selectedGame.location}</p>
                </div>
              </div>

              {selectedGame.is_completed ? (
                <div className="bg-gray-100 dark:bg-gray-700 rounded-lg p-4">
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Final Score</p>
                  <p className="text-3xl font-bold text-gray-900 dark:text-white">
                    {selectedGame.team_score} - {selectedGame.opponent_score}
                  </p>
                  <p className={`text-lg font-bold mt-2 ${selectedGame.team_score! > selectedGame.opponent_score! ? 'text-green-600' : 'text-red-600'}`}>
                    {selectedGame.team_score! > selectedGame.opponent_score! ? 'Win' : 'Loss'}
                  </p>
                </div>
              ) : (
                <>
                  {selectedGame.home_score !== null && selectedGame.away_score !== null && (
                    <div className="bg-gray-100 dark:bg-gray-700 rounded-lg p-4">
                      <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Projected Score</p>
                      <p className="text-2xl font-bold text-gray-900 dark:text-white">
                        {selectedGame.location === 'Home' ? selectedGame.home_score : selectedGame.away_score} - {selectedGame.location === 'Home' ? selectedGame.away_score : selectedGame.home_score}
                      </p>
                    </div>
                  )}
                  {selectedGame.team_win_prob !== null && (
                    <div className="bg-gray-100 dark:bg-gray-700 rounded-lg p-4">
                      <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Win Probability</p>
                      <p className="text-2xl font-bold text-gray-900 dark:text-white">
                        {formatWinProbability(selectedGame.team_win_prob)}
                      </p>
                    </div>
                  )}
                </>
              )}

              <div className="border-t border-gray-200 dark:border-gray-600 pt-4">
                <h3 className="font-bold text-lg mb-3 text-gray-900 dark:text-white">Opponent Rankings</h3>
                <div className="grid grid-cols-3 gap-4">
                  {selectedGame.opponent_pr_rank && (
                    <div>
                      <p className="text-sm text-gray-600 dark:text-gray-400">Power Rating</p>
                      <p className="text-xl font-bold text-blue-600 dark:text-blue-400">#{selectedGame.opponent_pr_rank}</p>
                    </div>
                  )}
                  {selectedGame.opponent_or_rank && (
                    <div>
                      <p className="text-sm text-gray-600 dark:text-gray-400">Offensive</p>
                      <p className="text-xl font-bold text-orange-600 dark:text-orange-400">#{selectedGame.opponent_or_rank}</p>
                    </div>
                  )}
                  {selectedGame.opponent_dr_rank && (
                    <div>
                      <p className="text-sm text-gray-600 dark:text-gray-400">Defensive</p>
                      <p className="text-xl font-bold text-red-600 dark:text-red-400">#{selectedGame.opponent_dr_rank}</p>
                    </div>
                  )}
                </div>
              </div>

              {selectedGame.PEAR && (
                <div className="bg-indigo-50 dark:bg-indigo-900 rounded-lg p-4">
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">PEAR Rating</p>
                  <p className="text-xl font-bold text-indigo-600 dark:text-indigo-400">{selectedGame.PEAR}</p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
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