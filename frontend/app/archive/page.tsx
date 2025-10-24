'use client';

import { useState, useEffect } from 'react';
import axios from 'axios';
import { ArrowLeft, History, TrendingUp } from 'lucide-react';
import Link from 'next/link';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface HistoricalData {
  Team: string;
  'Normalized Rating': number;
  Season: string;
}

interface TeamHistoryData {
  Season: string;
  'Normalized Rating': number;
  most_deserving: number;
  SOS: number;
  SOR: number;
  offensive_rank: number;
  defensive_rank: number;
  STM_rank: number;
  PBR_rank: number;
  DCE_rank: number;
  DDE_rank: number;
}

const YEAR_WEEK_MAP: { [key: number]: number } = {
  2024: 17, 2023: 15, 2022: 16, 2021: 16,
  2020: 17, 2019: 17, 2018: 16, 2017: 16,
  2016: 16, 2015: 16, 2014: 17
};

export default function ArchivePage() {
  const [historicalData, setHistoricalData] = useState<HistoricalData[]>([]);
  const [selectedTeam, setSelectedTeam] = useState('');
  const [teamHistory, setTeamHistory] = useState<TeamHistoryData[]>([]);
  const [selectedYear, setSelectedYear] = useState<number | null>(null);
  const [yearRatings, setYearRatings] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [teamLoading, setTeamLoading] = useState(false);
  const [yearLoading, setYearLoading] = useState(false);

  const teams = Array.from(new Set(historicalData.map(d => d.Team))).sort();
  const years = Object.keys(YEAR_WEEK_MAP).map(Number).sort((a, b) => b - a);

  useEffect(() => {
    fetchHistoricalData();
  }, []);

  const fetchHistoricalData = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/historical-ratings`);
      setHistoricalData(response.data.data);
      setLoading(false);
    } catch (error) {
      console.error('Error fetching historical data:', error);
      setLoading(false);
    }
  };

  const fetchTeamHistory = async (team: string) => {
    setTeamLoading(true);
    try {
      const response = await axios.get(`${API_URL}/api/team-history/${team}`);
      setTeamHistory(response.data.data);
    } catch (error) {
      console.error('Error fetching team history:', error);
    } finally {
      setTeamLoading(false);
    }
  };

  const fetchYearRatings = async (year: number) => {
    setYearLoading(true);
    try {
      const week = YEAR_WEEK_MAP[year];
      const response = await axios.get(`${API_URL}/api/ratings/${year}/${week}`);
      setYearRatings(response.data.data);
    } catch (error) {
      console.error('Error fetching year ratings:', error);
    } finally {
      setYearLoading(false);
    }
  };

  const handleTeamSelect = (team: string) => {
    setSelectedTeam(team);
    if (team) {
      fetchTeamHistory(team);
    } else {
      setTeamHistory([]);
    }
  };

  const handleYearSelect = (year: number | null) => {
    setSelectedYear(year);
    if (year) {
      fetchYearRatings(year);
    } else {
      setYearRatings([]);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      {/* Header */}
      <div className="bg-gradient-to-r from-slate-700 to-slate-800 text-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <Link href="/" className="inline-flex items-center text-slate-300 hover:text-white mb-6 transition-colors">
            <ArrowLeft className="w-5 h-5 mr-2" />
            Back to Current Ratings
          </Link>
          <div className="flex items-center space-x-4">
            <History className="w-12 h-12 text-slate-300" />
            <div>
              <h1 className="text-4xl md:text-5xl font-bold mb-2">Historical Archive</h1>
              <p className="text-slate-300 text-lg">CFB PEAR Ratings • 2014-2024</p>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Year Normalized Ratings */}
        <section className="mb-12">
          <div className="bg-white rounded-xl shadow-lg overflow-hidden">
            <div className="bg-gradient-to-r from-purple-600 to-purple-700 px-6 py-4">
              <h2 className="text-2xl font-bold text-white">Year Normalized Ratings</h2>
              <p className="text-purple-100 text-sm mt-1">All-time rankings across seasons</p>
            </div>
            <div className="p-6">
              {loading ? (
                <div className="text-center py-12">
                  <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-purple-600"></div>
                  <p className="mt-4 text-gray-600">Loading historical data...</p>
                </div>
              ) : (
                <div className="overflow-x-auto max-h-[500px] overflow-y-auto">
                  <table className="w-full text-sm">
                    <thead className="bg-gray-50 sticky top-0">
                      <tr>
                        <th className="px-4 py-3 text-left font-semibold text-gray-700">Rank</th>
                        <th className="px-4 py-3 text-left font-semibold text-gray-700">Team</th>
                        <th className="px-4 py-3 text-center font-semibold text-gray-700">Normalized Rating</th>
                        <th className="px-4 py-3 text-center font-semibold text-gray-700">Season</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-200">
                      {historicalData.slice(0, 100).map((item, index) => (
                        <tr key={index} className="hover:bg-gray-50">
                          <td className="px-4 py-3 text-gray-700 font-medium">{index + 1}</td>
                          <td className="px-4 py-3 font-semibold text-gray-900">{item.Team}</td>
                          <td className="px-4 py-3 text-center">
                            <span className="inline-block px-3 py-1 rounded-full bg-purple-100 text-purple-800 font-semibold">
                              {item['Normalized Rating'].toFixed(2)}
                            </span>
                          </td>
                          <td className="px-4 py-3 text-center text-gray-700">{item.Season}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        </section>

        {/* Two Column Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
          {/* Team History Lookup */}
          <section>
            <div className="bg-white rounded-xl shadow-lg overflow-hidden">
              <div className="bg-gradient-to-r from-blue-600 to-blue-700 px-6 py-4">
                <h2 className="text-2xl font-bold text-white">Team History Lookup</h2>
                <p className="text-blue-100 text-sm mt-1">View a specific team's historical stats</p>
              </div>
              <div className="p-6">
                <div className="mb-4">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Select Team
                  </label>
                  <select
                    value={selectedTeam}
                    onChange={(e) => handleTeamSelect(e.target.value)}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="">Choose a team...</option>
                    {teams.map((team) => (
                      <option key={team} value={team}>
                        {team}
                      </option>
                    ))}
                  </select>
                </div>

                {teamLoading && (
                  <div className="text-center py-8">
                    <div className="inline-block animate-spin rounded-full h-10 w-10 border-b-2 border-blue-600"></div>
                  </div>
                )}

                {!teamLoading && teamHistory.length > 0 && (
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead className="bg-gray-50">
                        <tr>
                          <th className="px-3 py-2 text-left font-semibold text-gray-700">Year</th>
                          <th className="px-3 py-2 text-center font-semibold text-gray-700">Rating</th>
                          <th className="px-3 py-2 text-center font-semibold text-gray-700">MD</th>
                          <th className="px-3 py-2 text-center font-semibold text-gray-700">OFF</th>
                          <th className="px-3 py-2 text-center font-semibold text-gray-700">DEF</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-200">
                        {teamHistory.map((item, index) => (
                          <tr key={index} className="hover:bg-gray-50">
                            <td className="px-3 py-2 font-semibold text-gray-900">{item.Season}</td>
                            <td className="px-3 py-2 text-center">
                              <span className="inline-block px-2 py-1 rounded bg-blue-100 text-blue-800 text-xs font-semibold">
                                {item['Normalized Rating'].toFixed(2)}
                              </span>
                            </td>
                            <td className="px-3 py-2 text-center text-gray-700">{item.most_deserving}</td>
                            <td className="px-3 py-2 text-center text-gray-700">{item.offensive_rank}</td>
                            <td className="px-3 py-2 text-center text-gray-700">{item.defensive_rank}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </div>
          </section>

          {/* Year Selector */}
          <section>
            <div className="bg-white rounded-xl shadow-lg overflow-hidden">
              <div className="bg-gradient-to-r from-green-600 to-green-700 px-6 py-4">
                <h2 className="text-2xl font-bold text-white">Season Ratings</h2>
                <p className="text-green-100 text-sm mt-1">View final ratings by year</p>
              </div>
              <div className="p-6">
                <div className="mb-4">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Select Season
                  </label>
                  <select
                    value={selectedYear || ''}
                    onChange={(e) => handleYearSelect(e.target.value ? Number(e.target.value) : null)}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent"
                  >
                    <option value="">Choose a season...</option>
                    {years.map((year) => (
                      <option key={year} value={year}>
                        {year} Season
                      </option>
                    ))}
                  </select>
                </div>

                {yearLoading && (
                  <div className="text-center py-8">
                    <div className="inline-block animate-spin rounded-full h-10 w-10 border-b-2 border-green-600"></div>
                  </div>
                )}

                {!yearLoading && yearRatings.length > 0 && (
                  <div className="overflow-x-auto max-h-[400px] overflow-y-auto">
                    <table className="w-full text-sm">
                      <thead className="bg-gray-50 sticky top-0">
                        <tr>
                          <th className="px-3 py-2 text-left font-semibold text-gray-700">Rank</th>
                          <th className="px-3 py-2 text-left font-semibold text-gray-700">Team</th>
                          <th className="px-3 py-2 text-center font-semibold text-gray-700">Rating</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-200">
                        {yearRatings.slice(0, 25).map((item, index) => (
                          <tr key={index} className="hover:bg-gray-50">
                            <td className="px-3 py-2 text-gray-700 font-medium">{index + 1}</td>
                            <td className="px-3 py-2 font-semibold text-gray-900">{item.Team}</td>
                            <td className="px-3 py-2 text-center">
                              <span className="inline-block px-2 py-1 rounded bg-green-100 text-green-800 font-semibold">
                                {item.Rating.toFixed(2)}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </div>
          </section>
        </div>

        {/* Year by Year Sections */}
        <section>
          <h2 className="text-3xl font-bold text-gray-900 mb-6 flex items-center">
            <TrendingUp className="w-8 h-8 mr-3 text-blue-600" />
            Season by Season Breakdown
          </h2>
          
          <div className="space-y-8">
            {years.map((year) => (
              <YearRatingsSection key={year} year={year} week={YEAR_WEEK_MAP[year]} />
            ))}
          </div>
        </section>
      </div>

      {/* Footer */}
      <footer className="bg-slate-900 text-white py-8 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <p className="text-slate-400">© 2025 PEAR Ratings. Advanced College Football Analytics.</p>
          <p className="text-slate-500 text-sm mt-2">@PEARatings</p>
        </div>
      </footer>
    </div>
  );
}

function YearRatingsSection({ year, week }: { year: number; week: number }) {
  const [data, setData] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    if (expanded && data.length === 0) {
      fetchData();
    }
  }, [expanded]);

  const fetchData = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/ratings/${year}/${week}`);
      setData(response.data.data);
      setLoading(false);
    } catch (error) {
      console.error(`Error fetching ${year} data:`, error);
      setLoading(false);
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-lg overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full bg-gradient-to-r from-slate-600 to-slate-700 px-6 py-4 text-left hover:from-slate-700 hover:to-slate-800 transition-all"
      >
        <div className="flex items-center justify-between">
          <h3 className="text-2xl font-bold text-white">{year} Season Ratings</h3>
          <span className="text-white">
            {expanded ? '−' : '+'}
          </span>
        </div>
        <p className="text-slate-300 text-sm mt-1">Week {week} Final Rankings</p>
      </button>
      
      {expanded && (
        <div className="p-6">
          {loading ? (
            <div className="text-center py-8">
              <div className="inline-block animate-spin rounded-full h-10 w-10 border-b-2 border-slate-600"></div>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-4 py-3 text-left font-semibold text-gray-700">Rank</th>
                    <th className="px-4 py-3 text-left font-semibold text-gray-700">Team</th>
                    <th className="px-4 py-3 text-center font-semibold text-gray-700">Rating</th>
                    <th className="px-4 py-3 text-center font-semibold text-gray-700">MD</th>
                    <th className="px-4 py-3 text-center font-semibold text-gray-700">SOS</th>
                    <th className="px-4 py-3 text-center font-semibold text-gray-700">Conference</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200">
                  {data.slice(0, 25).map((item, index) => (
                    <tr key={index} className="hover:bg-gray-50">
                      <td className="px-4 py-3 text-gray-700 font-medium">{index + 1}</td>
                      <td className="px-4 py-3 font-semibold text-gray-900">{item.Team}</td>
                      <td className="px-4 py-3 text-center">
                        <span className="inline-block px-3 py-1 rounded-full bg-slate-100 text-slate-800 font-semibold">
                          {item.Rating.toFixed(2)}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-center text-gray-700">{item.MD}</td>
                      <td className="px-4 py-3 text-center text-gray-700">{item.SOS.toFixed(2)}</td>
                      <td className="px-4 py-3 text-center text-xs text-gray-600">{item.CONF}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}