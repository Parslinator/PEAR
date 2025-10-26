'use client';

import { useState, useEffect } from 'react';
import { Trophy, Calendar, Download, Search } from 'lucide-react';
import axios from 'axios';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface TeamStats {
  Team: string;
  Conference: string;
  Rating: number;
  NET: number;
  RPI: number;
  ELO: number;
  ELO_Rank: number;
  PRR: number;
  RQI: number;
  SOS: number;
  SOR: number;
  Q1: string;
  Q2: string;
  Q3: string;
  Q4: string;
}

type SortField = keyof TeamStats;

export default function CbasePage() {
  const [stats, setStats] = useState<TeamStats[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedConference, setSelectedConference] = useState('All');
  const [dataDate, setDataDate] = useState('');
  const [sortField, setSortField] = useState<SortField>('Rating');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');

  useEffect(() => {
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API_URL}/api/cbase/stats`);
      const statsData = response.data.stats.map((team: any) => ({
        ...team,
        ELO_Rank: team.ELO // Store the actual ELO rank
      }));
      setStats(statsData);
      setDataDate(response.data.date);
    } catch (error) {
      console.error('Error fetching stats:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      // Default sort directions
      if (field === 'NET' || field === 'RPI' || field === 'ELO_Rank' || field === 'PRR' || field === 'RQI' || field === 'SOS') {
        setSortDirection('asc');
      } else {
        setSortDirection('desc');
      }
    }
  };

  const conferences = ['All', ...Array.from(new Set(stats.map(s => s.Conference))).sort()];

  const filteredAndSortedStats = stats
    .filter(team => {
      const matchesSearch = team.Team.toLowerCase().includes(searchQuery.toLowerCase());
      const matchesConference = selectedConference === 'All' || team.Conference === selectedConference;
      return matchesSearch && matchesConference;
    })
    .sort((a, b) => {
      let aVal = a[sortField];
      let bVal = b[sortField];
      
      // Handle string values (Q1, Q2, Q3, Q4)
      if (typeof aVal === 'string' && typeof bVal === 'string') {
        return sortDirection === 'asc' ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
      }
      
      // Handle numeric values
      if (typeof aVal === 'number' && typeof bVal === 'number') {
        return sortDirection === 'asc' ? aVal - bVal : bVal - aVal;
      }
      
      return 0;
    });

  const downloadCSV = () => {
    const headers = ['Rank', 'Team', 'NET', 'RPI', 'ELO', 'TSR', 'RQI', 'SOS', 'Q1', 'Q2', 'Q3', 'Q4', 'Conference'];
    const csvData = filteredAndSortedStats.map((team, index) => [
      index + 1,
      team.Team,
      team.NET,
      team.RPI,
      team.ELO_Rank,
      team.PRR,
      team.RQI,
      team.SOS,
      team.Q1,
      team.Q2,
      team.Q3,
      team.Q4,
      team.Conference
    ]);
    
    const csvContent = [
      headers.join(','),
      ...csvData.map(row => row.join(','))
    ].join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'cbase_ratings_resume.csv';
    a.click();
    window.URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-blue-900 via-blue-800 to-blue-900 text-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
          <div className="text-center">
            <div className="flex items-center justify-center mb-6">
              <Trophy className="w-16 h-16 text-yellow-400 mr-4" />
              <h1 className="text-5xl font-bold">CBASE PEAR</h1>
            </div>
            {dataDate && (
              <div className="mt-4 flex items-center justify-center space-x-8 text-sm">
                <div className="flex items-center">
                  <Calendar className="w-5 h-5 mr-2 text-yellow-400" />
                  <span>Data as of {dataDate}</span>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <section>
          <div className="bg-white rounded-xl shadow-lg overflow-hidden">
            <div className="bg-gradient-to-r from-blue-600 to-blue-700 px-6 py-4 flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-bold text-white">Ratings & Resume</h2>
                <p className="text-blue-100 text-sm mt-1">Current season rankings and resume metrics</p>
              </div>
              <button
                onClick={downloadCSV}
                className="flex items-center space-x-2 bg-white text-blue-600 px-4 py-2 rounded-lg hover:bg-blue-50 transition-colors font-semibold"
              >
                <Download className="w-4 h-4" />
                <span>Download CSV</span>
              </button>
            </div>

            <div className="p-6">
              {/* Filters */}
              <div className="mb-6 flex gap-4 flex-wrap items-center">
                <div className="flex-1 min-w-[200px] relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
                  <input
                    type="text"
                    placeholder="Search teams..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
                <select
                  value={selectedConference}
                  onChange={(e) => setSelectedConference(e.target.value)}
                  className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  {conferences.map(conf => (
                    <option key={conf} value={conf}>{conf}</option>
                  ))}
                </select>
              </div>

              {loading ? (
                <div className="text-center py-12">
                  <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
                  <p className="mt-4 text-gray-600">Loading ratings...</p>
                </div>
              ) : (
                <>
                  <div className="overflow-x-auto max-h-[600px] overflow-y-auto">
                    <table className="w-full text-sm">
                      <thead className="bg-gray-50 sticky top-0">
                        <tr>
                          <th className="px-3 py-3 text-left font-semibold text-gray-700">Rank</th>
                          <th 
                            className="px-3 py-3 text-left font-semibold text-gray-700 cursor-pointer hover:bg-gray-100"
                            onClick={() => handleSort('Team')}
                          >
                            Team {sortField === 'Team' && (sortDirection === 'asc' ? '↑' : '↓')}
                          </th>
                          <th 
                            className="px-3 py-3 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100"
                            onClick={() => handleSort('NET')}
                          >
                            NET {sortField === 'NET' && (sortDirection === 'asc' ? '↑' : '↓')}
                          </th>
                          <th 
                            className="px-3 py-3 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100"
                            onClick={() => handleSort('RPI')}
                          >
                            RPI {sortField === 'RPI' && (sortDirection === 'asc' ? '↑' : '↓')}
                          </th>
                          <th 
                            className="px-3 py-3 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100"
                            onClick={() => handleSort('ELO_Rank')}
                          >
                            ELO {sortField === 'ELO_Rank' && (sortDirection === 'asc' ? '↑' : '↓')}
                          </th>
                          <th 
                            className="px-3 py-3 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100"
                            onClick={() => handleSort('PRR')}
                          >
                            TSR {sortField === 'PRR' && (sortDirection === 'asc' ? '↑' : '↓')}
                          </th>
                          <th 
                            className="px-3 py-3 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100"
                            onClick={() => handleSort('RQI')}
                          >
                            RQI {sortField === 'RQI' && (sortDirection === 'asc' ? '↑' : '↓')}
                          </th>
                          <th 
                            className="px-3 py-3 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100"
                            onClick={() => handleSort('SOS')}
                          >
                            SOS {sortField === 'SOS' && (sortDirection === 'asc' ? '↑' : '↓')}
                          </th>
                          <th 
                            className="px-3 py-3 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100"
                            onClick={() => handleSort('Q1')}
                          >
                            Q1 {sortField === 'Q1' && (sortDirection === 'asc' ? '↑' : '↓')}
                          </th>
                          <th 
                            className="px-3 py-3 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100"
                            onClick={() => handleSort('Q2')}
                          >
                            Q2 {sortField === 'Q2' && (sortDirection === 'asc' ? '↑' : '↓')}
                          </th>
                          <th 
                            className="px-3 py-3 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100"
                            onClick={() => handleSort('Q3')}
                          >
                            Q3 {sortField === 'Q3' && (sortDirection === 'asc' ? '↑' : '↓')}
                          </th>
                          <th 
                            className="px-3 py-3 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100"
                            onClick={() => handleSort('Q4')}
                          >
                            Q4 {sortField === 'Q4' && (sortDirection === 'asc' ? '↑' : '↓')}
                          </th>
                          <th 
                            className="px-3 py-3 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100"
                            onClick={() => handleSort('Conference')}
                          >
                            Conf {sortField === 'Conference' && (sortDirection === 'asc' ? '↑' : '↓')}
                          </th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-200">
                        {filteredAndSortedStats.map((team, index) => (
                          <tr key={index} className="hover:bg-gray-50">
                            <td className="px-3 py-3 text-gray-700 font-medium">{index + 1}</td>
                            <td className="px-3 py-3 font-semibold text-gray-900">{team.Team}</td>
                            <td className="px-3 py-3 text-center text-gray-700">{team.NET}</td>
                            <td className="px-3 py-3 text-center text-gray-700">{team.RPI}</td>
                            <td className="px-3 py-3 text-center text-gray-700">{team.ELO_Rank}</td>
                            <td className="px-3 py-3 text-center text-gray-700">{team.PRR}</td>
                            <td className="px-3 py-3 text-center text-gray-700">{team.RQI}</td>
                            <td className="px-3 py-3 text-center text-gray-700">{team.SOS}</td>
                            <td className="px-3 py-3 text-center text-xs text-gray-600">{team.Q1}</td>
                            <td className="px-3 py-3 text-center text-xs text-gray-600">{team.Q2}</td>
                            <td className="px-3 py-3 text-center text-xs text-gray-600">{team.Q3}</td>
                            <td className="px-3 py-3 text-center text-xs text-gray-600">{team.Q4}</td>
                            <td className="px-3 py-3 text-center text-xs text-gray-600">{team.Conference}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  <div className="mt-6 text-sm text-gray-600 space-y-1">
                    <p><strong>NET</strong> - Mimicking the NCAA Evaluation Tool using TSR, RQI, SOS</p>
                    <p><strong>RPI</strong> - Warren Nolan's Live RPI | <strong>TSR</strong> - Team Strength Rank | <strong>RQI</strong> - Resume Quality Index | <strong>SOS</strong> - Strength of Schedule</p>
                    <p><strong>Q1-Q4</strong> - Quadrant records (wins-losses)</p>
                  </div>
                </>
              )}
            </div>
          </div>
        </section>
      </div>

      {/* Footer */}
      <footer className="bg-gray-900 text-gray-300 mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center">
            <p className="text-sm">
              © {new Date().getFullYear()} CBASE PEAR. All rights reserved.
            </p>
            <p className="text-xs mt-2 text-gray-400">
              Performance Evaluation and Analytics Rating System
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}