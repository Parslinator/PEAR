'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { Search, Filter, Shuffle } from 'lucide-react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || '';

interface TeamConferences {
  [teamName: string]: string;
}

export default function TeamsPage() {
  const router = useRouter();
  const [teams, setTeams] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [conferences, setConferences] = useState<string[]>([]);
  const [teamConferences, setTeamConferences] = useState<TeamConferences>({});
  const [selectedConference, setSelectedConference] = useState<string>('All');

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    await Promise.all([
      fetchTeams(),
      fetchConferences(),
      fetchTeamConferences()
    ]);
  };

  const fetchTeams = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${API_URL}/api/softball/teams`);
      if (response.ok) {
        const data = await response.json();
        const sortedTeams = (data.teams || []).sort((a: string, b: string) => 
          a.localeCompare(b)
        );
        setTeams(sortedTeams);
      }
    } catch (err) {
      console.error('Error fetching teams:', err);
    } finally {
      setLoading(false);
    }
  };

  const fetchConferences = async () => {
    try {
      const response = await fetch(`${API_URL}/api/softball/conferences`);
      if (response.ok) {
        const data = await response.json();
        setConferences(data.conferences || []);
      }
    } catch (err) {
      console.error('Error fetching conferences:', err);
    }
  };

  const fetchTeamConferences = async () => {
    try {
      const response = await fetch(`${API_URL}/api/softball/team-conferences`);
      if (response.ok) {
        const data = await response.json();
        setTeamConferences(data.team_conferences || {});
      }
    } catch (err) {
      console.error('Error fetching team conferences:', err);
    }
  };

  const handleTeamClick = (teamName: string) => {
    router.push(`/csoft/team-profile?team=${encodeURIComponent(teamName)}`);
  };

  const handleRandomTeam = () => {
    if (teams.length === 0) return;
    const randomIndex = Math.floor(Math.random() * teams.length);
    const randomTeam = teams[randomIndex];
    router.push(`/csoft/team-profile?team=${encodeURIComponent(randomTeam)}`);
  };

  const filteredTeams = teams.filter(team => {
    const matchesSearch = team.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesConference = selectedConference === 'All' || teamConferences[team] === selectedConference;
    return matchesSearch && matchesConference;
  });

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800 pt-20">
      <div className="max-w-7xl mx-auto px-4 py-6">
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-3xl sm:text-4xl font-bold text-gray-900 dark:text-white mb-2">
            College Softball Teams
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            Browse {teams.length} Division I college softball teams
          </p>
        </div>

        {/* Search, Filter, and Random Bar */}
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

          {/* Conference Filter */}
          <div className="relative sm:w-64">
            <Filter className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 pointer-events-none" size={20} />
            <select
              value={selectedConference}
              onChange={(e) => setSelectedConference(e.target.value)}
              className="w-full pl-10 pr-4 py-3 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500 appearance-none cursor-pointer"
            >
              <option value="All">All Conferences</option>
              {conferences.map((conference) => (
                <option key={conference} value={conference}>
                  {conference}
                </option>
              ))}
            </select>
            {/* Custom dropdown arrow */}
            <div className="absolute right-3 top-1/2 transform -translate-y-1/2 pointer-events-none">
              <svg className="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </div>
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

        {/* Active Filter Display */}
        {selectedConference !== 'All' && (
          <div className="mb-4 flex items-center gap-2">
            <span className="text-sm text-gray-600 dark:text-gray-400">Filtering by:</span>
            <span className="inline-flex items-center gap-2 px-3 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-300 rounded-full text-sm font-medium">
              {selectedConference}
              <button
                onClick={() => setSelectedConference('All')}
                className="hover:text-blue-900 dark:hover:text-blue-100"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </span>
          </div>
        )}

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
                  {filteredTeams.map((team) => (
                    <div
                      key={team}
                      onClick={() => handleTeamClick(team)}
                      className="bg-white dark:bg-gray-800 rounded-lg shadow-md hover:shadow-xl transition-all duration-200 cursor-pointer border border-gray-200 dark:border-gray-700 hover:border-blue-500 dark:hover:border-blue-400 group"
                    >
                      <div className="p-6 flex items-center gap-4">
                        {/* Team Logo */}
                        <div className="flex-shrink-0">
                          <img
                            src={`${API_URL}/api/softball-logo/${encodeURIComponent(team)}`}
                            alt={`${team} logo`}
                            className="w-16 h-16 object-contain group-hover:scale-110 transition-transform duration-200"
                            onError={(e) => {
                              e.currentTarget.style.display = 'none';
                            }}
                          />
                        </div>

                        {/* Team Name and Conference */}
                        <div className="flex-1 min-w-0">
                          <h3 className="text-lg font-semibold text-gray-900 dark:text-white group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors truncate">
                            {team}
                          </h3>
                          {teamConferences[team] && (
                            <p className="text-sm text-gray-500 dark:text-gray-400 truncate mt-1">
                              {teamConferences[team]}
                            </p>
                          )}
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
                  No teams found {searchQuery && `matching "${searchQuery}"`}
                  {selectedConference !== 'All' && ` in ${selectedConference}`}
                </p>
                <button
                  onClick={() => {
                    setSearchQuery('');
                    setSelectedConference('All');
                  }}
                  className="mt-4 px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                >
                  Clear Filters
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}