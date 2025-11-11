'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { Search, Shuffle } from 'lucide-react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || '';

interface TeamData {
  team: string;
  conference: string;
}

export default function FootballTeamsPage() {
  const router = useRouter();
  const [teams, setTeams] = useState<TeamData[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [conferenceFilter, setConferenceFilter] = useState('All');
  const [currentYear, setCurrentYear] = useState<number | null>(null);
  const [currentWeek, setCurrentWeek] = useState<number | null>(null);

  // Get unique conferences from teams
  const conferences = ['All', ...Array.from(new Set(teams.map(t => t.conference))).sort()];

  // Map conference names to shorthand
  const getConferenceShorthand = (conf: string): string => {
    const shorthandMap: { [key: string]: string } = {
      'American Athletic': 'AAC',
      'Conference USA': 'CUSA',
      'FBS Independents': 'IND',
      'Mid-American': 'MAC',
      'Mountain West': 'MW',
      'Atlantic Coast': 'ACC',
      'Big 12': 'Big 12',
      'Big Ten': 'Big Ten',
      'Pac-12': 'Pac-12',
      'Southeastern': 'SEC',
      'Sun Belt': 'Sun Belt',
    };
    return shorthandMap[conf] || conf;
  };

  useEffect(() => {
    fetchCurrentSeason();
  }, []);

  useEffect(() => {
    if (currentYear && currentWeek) {
      fetchTeams();
    }
  }, [currentYear, currentWeek]);

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

  const fetchTeams = async () => {
    if (!currentYear || !currentWeek) return;
    
    try {
      setLoading(true);
      // Fetch from ratings endpoint to get conference data
      const response = await fetch(`${API_URL}/api/ratings/${currentYear}/${currentWeek}`);
      if (response.ok) {
        const data = await response.json();
        const teamData: TeamData[] = (data.ratings || []).map((item: any) => ({
          team: item.Team,
          conference: item.CONF
        })).sort((a: TeamData, b: TeamData) => a.team.localeCompare(b.team));
        setTeams(teamData);
      }
    } catch (err) {
      console.error('Error fetching teams:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleTeamClick = (teamName: string) => {
    router.push(`/team-profile?team=${encodeURIComponent(teamName)}`);
  };

  const handleRandomTeam = () => {
    if (teams.length === 0) return;
    const randomIndex = Math.floor(Math.random() * teams.length);
    const randomTeam = teams[randomIndex];
    router.push(`/team-profile?team=${encodeURIComponent(randomTeam.team)}`);
  };

  const filteredTeams = teams.filter(teamData => {
    const matchesSearch = teamData.team.toLowerCase().includes(searchQuery.toLowerCase()) ||
      teamData.conference.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesConference = conferenceFilter === 'All' || teamData.conference === conferenceFilter;
    return matchesSearch && matchesConference;
  });

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800 pt-20">
      <div className="max-w-7xl mx-auto px-4 py-6">
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-3xl sm:text-4xl font-bold text-gray-900 dark:text-white mb-2">
            College Football Teams
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            Browse all {teams.length} FBS college football teams
          </p>
        </div>

        {/* Search and Random Bar */}
        <div className="mb-6 flex flex-col sm:flex-row gap-4">
          {/* Search Bar */}
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={20} />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search teams..."
              className="w-full pl-10 pr-4 py-3 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          {/* Random Team Button */}
          <button
            onClick={handleRandomTeam}
            disabled={teams.length === 0}
            className="flex items-center justify-center gap-2 px-4 py-3 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 text-white rounded-lg transition-all duration-200 shadow-md hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed whitespace-nowrap sm:w-auto"
          >
            <Shuffle size={20} />
            <span className="hidden sm:inline">Random</span>
          </button>
        </div>

        {/* Conference Filter Buttons */}
        <div className="mb-6 flex flex-wrap gap-2 justify-center">
          {conferences.map(conf => (
            <button
              key={conf}
              onClick={() => setConferenceFilter(conf)}
              className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                conferenceFilter === conf
                  ? 'bg-blue-600 dark:bg-blue-500 text-white'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
              }`}
            >
              {conf === 'All' ? 'All' : getConferenceShorthand(conf)}
            </button>
          ))}
        </div>

        {/* Teams Grid */}
        {loading ? (
          <div className="flex flex-col items-center justify-center py-20">
            <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600 dark:border-blue-400"></div>
            <p className="mt-4 text-gray-600 dark:text-gray-400 text-lg">
              Loading teams...
            </p>
          </div>
        ) : (
          <>
            {filteredTeams.length > 0 ? (
              <>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                  Showing {filteredTeams.length} {filteredTeams.length === 1 ? 'team' : 'teams'}
                </p>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {filteredTeams.map((teamData) => (
                    <div
                      key={teamData.team}
                      onClick={() => handleTeamClick(teamData.team)}
                      className="bg-white dark:bg-gray-800 rounded-lg shadow-md hover:shadow-xl transition-all duration-200 cursor-pointer border border-gray-200 dark:border-gray-700 hover:border-blue-500 dark:hover:border-blue-400 group"
                    >
                      <div className="p-6 flex items-center gap-4">
                        {/* Team Logo */}
                        <div className="flex-shrink-0">
                          <img
                            src={`${API_URL}/api/football-logo/${encodeURIComponent(teamData.team)}`}
                            alt={`${teamData.team} logo`}
                            className="w-16 h-16 object-contain group-hover:scale-110 transition-transform duration-200"
                            onError={(e) => {
                              e.currentTarget.style.display = 'none';
                            }}
                          />
                        </div>

                        {/* Team Name and Conference */}
                        <div className="flex-1 min-w-0">
                          <h3 className="text-lg font-semibold text-gray-900 dark:text-white group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors truncate">
                            {teamData.team}
                          </h3>
                          <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                            {getConferenceShorthand(teamData.conference)}
                          </p>
                        </div>

                        {/* Arrow indicator */}
                        <div className="flex-shrink-0 opacity-0 group-hover:opacity-100 transition-opacity">
                          <svg 
                            className="w-6 h-6 text-blue-600 dark:text-blue-400" 
                            fill="none" 
                            stroke="currentColor" 
                            viewBox="0 0 24 24"
                          >
                            <path 
                              strokeLinecap="round" 
                              strokeLinejoin="round" 
                              strokeWidth={2} 
                              d="M9 5l7 7-7 7" 
                            />
                          </svg>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <div className="text-center py-20">
                <p className="text-xl text-gray-600 dark:text-gray-400">
                  No teams found matching "{searchQuery}"
                </p>
                <button
                  onClick={() => setSearchQuery('')}
                  className="mt-4 px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                >
                  Clear Search
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}