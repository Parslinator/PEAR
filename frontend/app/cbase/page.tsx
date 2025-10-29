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
  resume_quality: number;
  avg_expected_wins: number;
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
    const headers = ['Rank', 'Team', 'NET', 'TSR', 'RQI', 'SOS', 'ELO', 'RPI', 'Q1', 'Q2', 'Q3', 'Q4', 'Conference'];
    const csvData = filteredAndSortedStats.map((team, index) => [
      index + 1,
      team.Team,
      team.NET_Score?.toFixed(3) || '',
      team.Rating?.toFixed(2) || '',
      team.resume_quality?.toFixed(3) || '',
      team.avg_expected_wins?.toFixed(3) || '',
      team.ELO?.toFixed(1) || '',
      team.RPI,
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
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-gray-900 dark:to-gray-800 pt-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Simple Header with Export */}
        <div className="flex justify-between items-center mb-6">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Ratings and Resume</h1>
            {dataDate && (
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">Data as of {dataDate}</p>
            )}
          </div>
          <button
            onClick={downloadCSV}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 dark:bg-blue-500 text-white rounded-lg hover:bg-blue-700 dark:hover:bg-blue-600 font-semibold transition-colors"
          >
            <Download className="w-4 h-4" />
            Export CSV
          </button>
        </div>

        <section>
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md overflow-hidden">
            {/* Search and Filter Controls */}
            <div className="p-4 border-b border-gray-200 dark:border-gray-700">
              <div className="flex gap-4 flex-wrap items-center">
                <div className="flex-1 min-w-[200px] relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 dark:text-gray-500 w-5 h-5" />
                  <input
                    type="text"
                    placeholder="Search teams..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="w-full pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400"
                  />
                </div>
                <select
                  value={selectedConference}
                  onChange={(e) => setSelectedConference(e.target.value)}
                  className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                >
                  {conferences.map(conf => (
                    <option key={conf} value={conf}>{conf}</option>
                  ))}
                </select>
              </div>
            </div>

            {/* Table Container */}
            <div className="p-6">
              {loading ? (
                <div className="text-center py-12">
                  <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 dark:border-blue-400"></div>
                  <p className="mt-4 text-gray-600 dark:text-gray-400">Loading ratings...</p>
                </div>
              ) : (
                <>
                  <div className="overflow-x-auto max-h-[600px] overflow-y-auto">
                    <table className="w-full text-sm">
                      <thead className="bg-gray-50 dark:bg-gray-700 sticky top-0">
                        <tr>
                          <th className="px-2 py-2 text-left font-semibold text-gray-700 dark:text-gray-200">Rank</th>
                          <th 
                            className="px-3 py-3 text-left font-semibold text-gray-700 dark:text-gray-200 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-600"
                            onClick={() => handleSort('Team')}
                          >
                            Team {sortField === 'Team' && (sortDirection === 'asc' ? '↑' : '↓')}
                          </th>
                          <th 
                            className="px-1 py-2 text-center font-semibold text-gray-700 dark:text-gray-200 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-600"
                            onClick={() => handleSort('NET_Score')}
                          >
                            NET {sortField === 'NET_Score' && (sortDirection === 'asc' ? '↑' : '↓')}
                          </th>
                          <th 
                            className="px-1 py-2 text-center font-semibold text-gray-700 dark:text-gray-200 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-600"
                            onClick={() => handleSort('Rating')}
                          >
                            TSR {sortField === 'Rating' && (sortDirection === 'asc' ? '↑' : '↓')}
                          </th>
                          <th 
                            className="px-1 py-2 text-center font-semibold text-gray-700 dark:text-gray-200 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-600"
                            onClick={() => handleSort('resume_quality')}
                          >
                            RQI {sortField === 'resume_quality' && (sortDirection === 'asc' ? '↑' : '↓')}
                          </th>
                          <th 
                            className="px-1 py-2 text-center font-semibold text-gray-700 dark:text-gray-200 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-600"
                            onClick={() => handleSort('avg_expected_wins')}
                          >
                            SOS {sortField === 'avg_expected_wins' && (sortDirection === 'asc' ? '↑' : '↓')}
                          </th>
                          <th 
                            className="px-1 py-2 text-center font-semibold text-gray-700 dark:text-gray-200 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-600"
                            onClick={() => handleSort('ELO')}
                          >
                            ELO {sortField === 'ELO' && (sortDirection === 'asc' ? '↑' : '↓')}
                          </th>
                          <th 
                            className="px-1 py-2 text-center font-semibold text-gray-700 dark:text-gray-200 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-600"
                            onClick={() => handleSort('RPI')}
                          >
                            RPI {sortField === 'RPI' && (sortDirection === 'asc' ? '↑' : '↓')}
                          </th>
                          <th className="px-1 py-2 text-center font-semibold text-gray-700 dark:text-gray-200 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-600"
                            onClick={() => handleSort('Q1')}
                          >
                            Q1 {sortField === 'Q1' && (sortDirection === 'asc' ? '↑' : '↓')}
                          </th>
                          <th className="px-1 py-2 text-center font-semibold text-gray-700 dark:text-gray-200 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-600"
                            onClick={() => handleSort('Q2')}
                          >
                            Q2 {sortField === 'Q2' && (sortDirection === 'asc' ? '↑' : '↓')}
                          </th>
                          <th className="px-1 py-2 text-center font-semibold text-gray-700 dark:text-gray-200 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-600"
                            onClick={() => handleSort('Q3')}
                          >
                            Q3 {sortField === 'Q3' && (sortDirection === 'asc' ? '↑' : '↓')}
                          </th>
                          <th className="px-1 py-2 text-center font-semibold text-gray-700 dark:text-gray-200 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-600"
                            onClick={() => handleSort('Q4')}
                          >
                            Q4 {sortField === 'Q4' && (sortDirection === 'asc' ? '↑' : '↓')}
                          </th>
                          <th className="px-1 py-2 text-center font-semibold text-gray-700 dark:text-gray-200 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-600"
                            onClick={() => handleSort('Conference')}
                          >
                            CONF {sortField === 'Conference' && (sortDirection === 'asc' ? '↑' : '↓')}
                          </th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                        {filteredAndSortedStats.map((team, index) => {
                          // Calculate background colors
                          const netBg = getRatingColor(team.NET_Score || 0, stats.map(s => s.NET_Score || 0).filter(v => v !== null), true);
                          const tsrBg = getRatingColor(team.Rating || 0, stats.map(s => s.Rating || 0).filter(v => v !== null), true);
                          const rqiBg = getRatingColor(team.resume_quality || 0, stats.map(s => s.resume_quality || 0).filter(v => v !== null), true);
                          const sosBg = getRatingColor(team.avg_expected_wins || 0, stats.map(s => s.avg_expected_wins || 0).filter(v => v !== null), false);
                          const eloBg = getRatingColor(team.ELO || 0, stats.map(s => s.ELO || 0).filter(v => v !== null), true);
                          const rpiBg = getRatingColor(team.RPI || 0, stats.map(s => s.RPI || 0).filter(v => v !== null), false);

                          // Get national ranks
                          const netRank = getNationalRank(team.NET_Score, 'NET_Score', true);
                          const tsrRank = getNationalRank(team.Rating, 'Rating', true);
                          const rqiRank = getNationalRank(team.resume_quality, 'resume_quality', true);
                          const sosRank = getNationalRank(team.avg_expected_wins, 'avg_expected_wins', false);
                          const eloRank = getNationalRank(team.ELO, 'ELO', true);
                          const rpiRank = getNationalRank(team.RPI, 'RPI', false);

                          return (
                            <tr key={index} className="hover:bg-gray-50 dark:hover:bg-gray-700/50">
                              <td className="px-2 py-2 text-gray-700 dark:text-gray-300 font-medium border-r-2 border-gray-300 dark:border-gray-600">{index + 1}</td>
                              <td className="px-2 py-2 font-semibold text-gray-900 dark:text-white text-sm">
                                <div className="flex items-center gap-2">
                                  <img 
                                    src={`http://localhost:8000/api/baseball-logo/${encodeURIComponent(team.Team)}`}
                                    alt={`${team.Team} logo`}
                                    className="w-6 h-6 object-contain"
                                    onError={(e) => {
                                      e.currentTarget.style.display = 'none';
                                    }}
                                  />
                                  <span>{team.Team}</span>
                                </div>
                              </td>
                              <td className="px-1 py-2 text-center">
                                <div className="flex items-center justify-center gap-1 min-w-[70px] mx-auto">
                                  <span className="text-[10px] font-medium text-gray-700 dark:text-gray-300 text-right w-[32px]">
                                    {team.NET_Score?.toFixed(3)}
                                  </span>
                                  <span className="inline-flex items-center justify-center px-2 py-1 rounded text-[9px] font-semibold min-w-[35px]" style={{ backgroundColor: netBg, color: getTextColor(netBg) }}>{netRank}</span>
                                </div>
                              </td>
                              <td className="px-1 py-2 text-center">
                                <div className="flex items-center justify-center gap-1 min-w-[70px] mx-auto">
                                  <span className="text-[10px] font-medium text-gray-700 dark:text-gray-300 text-right w-[32px]">
                                    {team.Rating?.toFixed(2)}
                                  </span>
                                  <span className="inline-flex items-center justify-center px-2 py-1 rounded text-[9px] font-semibold min-w-[35px]" style={{ backgroundColor: tsrBg, color: getTextColor(tsrBg) }}>{tsrRank}</span>
                                </div>
                              </td>
                              <td className="px-1 py-2 text-center">
                                <div className="flex items-center justify-center gap-1 min-w-[70px] mx-auto">
                                  <span className="text-[10px] font-medium text-gray-700 dark:text-gray-300 text-right w-[32px]">
                                    {team.resume_quality?.toFixed(3)}
                                  </span>
                                  <span className="inline-flex items-center justify-center px-2 py-1 rounded text-[9px] font-semibold min-w-[35px]" style={{ backgroundColor: rqiBg, color: getTextColor(rqiBg) }}>{rqiRank}</span>
                                </div>
                              </td>
                              <td className="px-1 py-2 text-center">
                                <div className="flex items-center justify-center gap-1 min-w-[70px] mx-auto">
                                  <span className="text-[10px] font-medium text-gray-700 dark:text-gray-300 text-right w-[32px]">
                                    {team.avg_expected_wins?.toFixed(3)}
                                  </span>
                                  <span className="inline-flex items-center justify-center px-2 py-1 rounded text-[9px] font-semibold min-w-[35px]" style={{ backgroundColor: sosBg, color: getTextColor(sosBg) }}>{sosRank}</span>
                                </div>
                              </td>
                              <td className="px-1 py-2 text-center">
                                <div className="flex items-center justify-center gap-1 min-w-[70px] mx-auto">
                                  <span className="text-[10px] font-medium text-gray-700 dark:text-gray-300 text-right w-[32px]">
                                    {team.ELO?.toFixed(1)}
                                  </span>
                                  <span className="inline-flex items-center justify-center px-2 py-1 rounded text-[9px] font-semibold min-w-[35px]" style={{ backgroundColor: eloBg, color: getTextColor(eloBg) }}>{eloRank}</span>
                                </div>
                              </td>
                              <td className="px-1 py-2 text-center">
                                <div className="flex items-center justify-center gap-1 min-w-[50px] mx-auto">
                                  <span className="inline-flex items-center justify-center px-2 py-1 rounded text-[9px] font-semibold min-w-[35px]" style={{ backgroundColor: rpiBg, color: getTextColor(rpiBg) }}>{rpiRank}</span>
                                </div>
                              </td>
                              <td className="px-1 py-2 text-center text-xs text-gray-600 dark:text-gray-400">{team.Q1}</td>
                              <td className="px-1 py-2 text-center text-xs text-gray-600 dark:text-gray-400">{team.Q2}</td>
                              <td className="px-1 py-2 text-center text-xs text-gray-600 dark:text-gray-400">{team.Q3}</td>
                              <td className="px-1 py-2 text-center text-xs text-gray-600 dark:text-gray-400">{team.Q4}</td>
                              <td className="px-1 py-2 text-center text-xs text-gray-600 dark:text-gray-400">{team.Conference}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>

                  <div className="mt-6 text-sm text-gray-600 dark:text-gray-400 space-y-1 border-t dark:border-gray-700 pt-4">
                    <p><strong className="dark:text-gray-300">TSR</strong> - Team Strength Rating | <strong className="dark:text-gray-300">NET</strong> - Mimicking the NCAA Evaluation Tool using TSR, RQI, SOS</p>
                    <p><strong className="dark:text-gray-300">RPI</strong> - Warren Nolan's Live RPI | <strong className="dark:text-gray-300">RQI</strong> - Resume Quality Index | <strong className="dark:text-gray-300">SOS</strong> - Strength of Schedule</p>
                    <p><strong className="dark:text-gray-300">Q1-Q4</strong> - Quadrant records (wins-losses)</p>
                  </div>
                </>
              )}
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}