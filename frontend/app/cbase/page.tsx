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
  NET_Score: number;
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
  const [sortField, setSortField] = useState<SortField>('NET');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');

  useEffect(() => {
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API_URL}/api/cbase/stats`);
      setStats(response.data.stats);
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
      if (field === 'Rating' || field === 'NET_Score') {
        setSortDirection('desc'); // Higher is better for Rating and NET_Score
      } else if (field === 'NET' || field === 'RPI' || field === 'ELO_Rank' || field === 'PRR' || field === 'RQI' || field === 'SOS') {
        setSortDirection('asc'); // Lower is better for ranks
      } else {
        setSortDirection('desc');
      }
    }
  };

  // Calculate national rank for a team's stat value
  const getNationalRank = (value: number, field: SortField, higherIsBetter: boolean = true): number => {
    if (value === null || value === undefined) return stats.length;
    
    const sortedValues = stats
      .map(s => s[field] as number)
      .filter(v => v !== null && v !== undefined)
      .sort((a, b) => higherIsBetter ? b - a : a - b);
    
    return sortedValues.indexOf(value) + 1;
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

  const getTextColor = (bgColor: string) => {
    const match = bgColor.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
    if (!match) return 'white';
    
    const r = parseInt(match[1]);
    const g = parseInt(match[2]);
    const b = parseInt(match[3]);
    
    const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
    return luminance > 0.5 ? 'black' : 'white';
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
    const headers = ['Rank', 'Team', 'TSR', 'NET', 'RPI', 'ELO', 'RQI', 'SOS', 'Q1', 'Q2', 'Q3', 'Q4', 'Conference'];
    const csvData = filteredAndSortedStats.map((team, index) => [
      index + 1,
      team.Team,
      team.Rating?.toFixed(2) || '',
      team.NET_Score?.toFixed(3) || '',
      team.RPI,
      team.ELO_Rank,
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
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">Ratings and Resume</h1>
          {dataDate && (
            <p className="text-lg text-gray-600">Data as of {dataDate}</p>
          )}
        </div>

        <section>
          <div className="bg-white rounded-xl shadow-lg overflow-hidden">
            <div className="bg-gradient-to-r from-blue-600 to-blue-700 px-6 py-4 flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-bold text-white"></h2>
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
                            onClick={() => handleSort('Rating')}
                          >
                            TSR {sortField === 'Rating' && (sortDirection === 'asc' ? '↑' : '↓')}
                          </th>
                          <th 
                            className="px-3 py-3 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100"
                            onClick={() => handleSort('NET_Score')}
                          >
                            NET {sortField === 'NET_Score' && (sortDirection === 'asc' ? '↑' : '↓')}
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
                        {filteredAndSortedStats.map((team, index) => {
                          // Calculate background colors for TSR and NET
                          const tsrBg = getRatingColor(team.Rating || 0, stats.map(s => s.Rating || 0).filter(v => v !== null), true);
                          const netBg = getRatingColor(team.NET_Score || 0, stats.map(s => s.NET_Score || 0).filter(v => v !== null), true);

                          // Get national ranks for TSR and NET
                          const tsrRank = getNationalRank(team.Rating, 'Rating', true);
                          const netRank = getNationalRank(team.NET_Score, 'NET_Score', true);

                          return (
                            <tr key={index} className="hover:bg-gray-50">
                              <td className="px-3 py-3 text-gray-700 font-medium">{index + 1}</td>
                              <td className="px-3 py-3 font-semibold text-gray-900">{team.Team}</td>
                              <td className="px-3 py-3 text-center">
                                <div className="flex flex-col items-center gap-0.5">
                                  <span 
                                    className="inline-block px-2 py-1 rounded text-xs font-semibold"
                                    style={{ backgroundColor: tsrBg, color: getTextColor(tsrBg) }}
                                  >
                                    {team.Rating?.toFixed(2)}
                                  </span>
                                  <span className="text-[10px] text-gray-500">#{tsrRank}</span>
                                </div>
                              </td>
                              <td className="px-3 py-3 text-center">
                                <div className="flex flex-col items-center gap-0.5">
                                  <span 
                                    className="inline-block px-2 py-1 rounded text-xs font-semibold"
                                    style={{ backgroundColor: netBg, color: getTextColor(netBg) }}
                                  >
                                    {team.NET_Score?.toFixed(3)}
                                  </span>
                                  <span className="text-[10px] text-gray-500">#{netRank}</span>
                                </div>
                              </td>
                              <td className="px-3 py-3 text-center text-gray-700">{team.RPI}</td>
                              <td className="px-3 py-3 text-center text-gray-700">{team.ELO_Rank}</td>
                              <td className="px-3 py-3 text-center text-gray-700">{team.RQI}</td>
                              <td className="px-3 py-3 text-center text-gray-700">{team.SOS}</td>
                              <td className="px-3 py-3 text-center text-xs text-gray-600">{team.Q1}</td>
                              <td className="px-3 py-3 text-center text-xs text-gray-600">{team.Q2}</td>
                              <td className="px-3 py-3 text-center text-xs text-gray-600">{team.Q3}</td>
                              <td className="px-3 py-3 text-center text-xs text-gray-600">{team.Q4}</td>
                              <td className="px-3 py-3 text-center text-xs text-gray-600">{team.Conference}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>

                  <div className="mt-6 text-sm text-gray-600 space-y-1">
                    <p><strong>TSR</strong> - Team Strength Rating | <strong>NET</strong> - Mimicking the NCAA Evaluation Tool using TSR, RQI, SOS</p>
                    <p><strong>RPI</strong> - Warren Nolan's Live RPI | <strong>RQI</strong> - Resume Quality Index | <strong>SOS</strong> - Strength of Schedule</p>
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