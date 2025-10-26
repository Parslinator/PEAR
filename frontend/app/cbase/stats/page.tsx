'use client';

import { useState, useEffect } from 'react';
import { Trophy, Download, Search } from 'lucide-react';
import axios from 'axios';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface TeamStats {
  Team: string;
  Conference: string;
  Rating: number;
  fWAR: number;
  oWAR_z: number;
  pWAR_z: number;
  PYTHAG: number;
  ERA: number;
  WHIP: number;
  KP9: number;
  RPG: number;
  BA: number;
  OBP: number;
  SLG: number;
  OPS: number;
}

type SortField = keyof TeamStats;

export default function CbaseStatsPage() {
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
      // Default sort directions - for ERA and WHIP, lower is better (asc), for others higher is better (desc)
      if (field === 'ERA' || field === 'WHIP') {
        setSortDirection('asc');
      } else {
        setSortDirection('desc');
      }
    }
  };

  const getRatingColor = (value: number, allValues: number[], higherIsBetter: boolean = true) => {
    const max = Math.max(...allValues);
    const min = Math.min(...allValues);
    const range = max - min;
    let normalized = (value - min) / range;
    
    if (!higherIsBetter) {
      normalized = 1 - normalized;
    }
    
    // Blue (good) to Grey to Red (bad)
    if (normalized >= 0.5) {
      const t = (normalized - 0.5) * 2;
      const r = Math.round(128 + (0 - 128) * t);
      const g = Math.round(128 + (100 - 128) * t);
      const b = Math.round(128 + (200 - 128) * t);
      return `rgb(${r}, ${g}, ${b})`;
    } else {
      const t = normalized * 2;
      const r = Math.round(200 + (128 - 200) * t);
      const g = Math.round(50 + (128 - 50) * t);
      const b = Math.round(50 + (128 - 50) * t);
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
      
      // Handle string values
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
    const headers = ['Rank', 'Team', 'WAR', 'PYTHAG', 'ERA', 'WHIP', 'KP9', 'RPG', 'BA', 'OBP', 'SLG', 'OPS', 'Conference'];
    const csvData = filteredAndSortedStats.map((team, index) => [
      index + 1,
      team.Team,
      team.fWAR?.toFixed(2) || '',
      team.PYTHAG?.toFixed(3) || '',
      team.ERA?.toFixed(2) || '',
      team.WHIP?.toFixed(2) || '',
      team.KP9?.toFixed(1) || '',
      team.RPG?.toFixed(1) || '',
      team.BA?.toFixed(3) || '',
      team.OBP?.toFixed(3) || '',
      team.SLG?.toFixed(3) || '',
      team.OPS?.toFixed(3) || '',
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
    a.download = 'cbase_advanced_stats.csv';
    a.click();
    window.URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-900 via-blue-800 to-blue-900 text-white py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-center mb-4">
            <Trophy className="w-12 h-12 text-yellow-400 mr-3" />
            <h1 className="text-4xl font-bold">Advanced Team Statistics</h1>
          </div>
          {dataDate && (
            <p className="text-center text-blue-200">Data as of {dataDate}</p>
          )}
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="bg-white rounded-xl shadow-lg overflow-hidden">
          <div className="bg-gradient-to-r from-green-600 to-green-700 px-6 py-4 flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-bold text-white">Team Stats</h2>
              <p className="text-green-100 text-sm mt-1">Advanced offensive and pitching metrics</p>
            </div>
            <button
              onClick={downloadCSV}
              className="flex items-center space-x-2 bg-white text-green-600 px-4 py-2 rounded-lg hover:bg-green-50 transition-colors font-semibold"
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
                  className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent"
                />
              </div>
              <select
                value={selectedConference}
                onChange={(e) => setSelectedConference(e.target.value)}
                className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent"
              >
                {conferences.map(conf => (
                  <option key={conf} value={conf}>{conf}</option>
                ))}
              </select>
            </div>

            {loading ? (
              <div className="text-center py-12">
                <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-green-600"></div>
                <p className="mt-4 text-gray-600">Loading statistics...</p>
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
                          onClick={() => handleSort('fWAR')}
                        >
                          WAR {sortField === 'fWAR' && (sortDirection === 'asc' ? '↑' : '↓')}
                        </th>
                        <th 
                          className="px-3 py-3 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100"
                          onClick={() => handleSort('PYTHAG')}
                        >
                          PYTHAG {sortField === 'PYTHAG' && (sortDirection === 'asc' ? '↑' : '↓')}
                        </th>
                        <th 
                          className="px-3 py-3 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100"
                          onClick={() => handleSort('ERA')}
                        >
                          ERA {sortField === 'ERA' && (sortDirection === 'asc' ? '↑' : '↓')}
                        </th>
                        <th 
                          className="px-3 py-3 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100"
                          onClick={() => handleSort('WHIP')}
                        >
                          WHIP {sortField === 'WHIP' && (sortDirection === 'asc' ? '↑' : '↓')}
                        </th>
                        <th 
                          className="px-3 py-3 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100"
                          onClick={() => handleSort('KP9')}
                        >
                          K/9 {sortField === 'KP9' && (sortDirection === 'asc' ? '↑' : '↓')}
                        </th>
                        <th 
                          className="px-3 py-3 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100"
                          onClick={() => handleSort('RPG')}
                        >
                          RPG {sortField === 'RPG' && (sortDirection === 'asc' ? '↑' : '↓')}
                        </th>
                        <th 
                          className="px-3 py-3 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100"
                          onClick={() => handleSort('BA')}
                        >
                          BA {sortField === 'BA' && (sortDirection === 'asc' ? '↑' : '↓')}
                        </th>
                        <th 
                          className="px-3 py-3 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100"
                          onClick={() => handleSort('OBP')}
                        >
                          OBP {sortField === 'OBP' && (sortDirection === 'asc' ? '↑' : '↓')}
                        </th>
                        <th 
                          className="px-3 py-3 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100"
                          onClick={() => handleSort('SLG')}
                        >
                          SLG {sortField === 'SLG' && (sortDirection === 'asc' ? '↑' : '↓')}
                        </th>
                        <th 
                          className="px-3 py-3 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100"
                          onClick={() => handleSort('OPS')}
                        >
                          OPS {sortField === 'OPS' && (sortDirection === 'asc' ? '↑' : '↓')}
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
                        const warBg = getRatingColor(team.fWAR || 0, stats.map(s => s.fWAR || 0).filter(v => v !== null), true);
                        const pythagBg = getRatingColor(team.PYTHAG || 0, stats.map(s => s.PYTHAG || 0).filter(v => v !== null), true);
                        const eraBg = getRatingColor(team.ERA || 0, stats.map(s => s.ERA || 0).filter(v => v !== null), false);
                        const whipBg = getRatingColor(team.WHIP || 0, stats.map(s => s.WHIP || 0).filter(v => v !== null), false);
                        const kp9Bg = getRatingColor(team.KP9 || 0, stats.map(s => s.KP9 || 0).filter(v => v !== null), true);
                        const rpgBg = getRatingColor(team.RPG || 0, stats.map(s => s.RPG || 0).filter(v => v !== null), true);
                        const baBg = getRatingColor(team.BA || 0, stats.map(s => s.BA || 0).filter(v => v !== null), true);
                        const obpBg = getRatingColor(team.OBP || 0, stats.map(s => s.OBP || 0).filter(v => v !== null), true);
                        const slgBg = getRatingColor(team.SLG || 0, stats.map(s => s.SLG || 0).filter(v => v !== null), true);
                        const opsBg = getRatingColor(team.OPS || 0, stats.map(s => s.OPS || 0).filter(v => v !== null), true);

                        return (
                          <tr key={index} className="hover:bg-gray-50">
                            <td className="px-3 py-3 text-gray-700 font-medium">{index + 1}</td>
                            <td className="px-3 py-3 font-semibold text-gray-900">{team.Team}</td>
                            <td className="px-3 py-3 text-center">
                              <span 
                                className="inline-block px-2 py-1 rounded text-xs font-semibold"
                                style={{ backgroundColor: warBg, color: getTextColor(warBg) }}
                              >
                                {team.fWAR?.toFixed(2)}
                              </span>
                            </td>
                            <td className="px-3 py-3 text-center">
                              <span 
                                className="inline-block px-2 py-1 rounded text-xs font-semibold"
                                style={{ backgroundColor: pythagBg, color: getTextColor(pythagBg) }}
                              >
                                {team.PYTHAG?.toFixed(3)}
                              </span>
                            </td>
                            <td className="px-3 py-3 text-center">
                              <span 
                                className="inline-block px-2 py-1 rounded text-xs font-semibold"
                                style={{ backgroundColor: eraBg, color: getTextColor(eraBg) }}
                              >
                                {team.ERA?.toFixed(2)}
                              </span>
                            </td>
                            <td className="px-3 py-3 text-center">
                              <span 
                                className="inline-block px-2 py-1 rounded text-xs font-semibold"
                                style={{ backgroundColor: whipBg, color: getTextColor(whipBg) }}
                              >
                                {team.WHIP?.toFixed(2)}
                              </span>
                            </td>
                            <td className="px-3 py-3 text-center">
                              <span 
                                className="inline-block px-2 py-1 rounded text-xs font-semibold"
                                style={{ backgroundColor: kp9Bg, color: getTextColor(kp9Bg) }}
                              >
                                {team.KP9?.toFixed(1)}
                              </span>
                            </td>
                            <td className="px-3 py-3 text-center">
                              <span 
                                className="inline-block px-2 py-1 rounded text-xs font-semibold"
                                style={{ backgroundColor: rpgBg, color: getTextColor(rpgBg) }}
                              >
                                {team.RPG?.toFixed(1)}
                              </span>
                            </td>
                            <td className="px-3 py-3 text-center">
                              <span 
                                className="inline-block px-2 py-1 rounded text-xs font-semibold"
                                style={{ backgroundColor: baBg, color: getTextColor(baBg) }}
                              >
                                {team.BA?.toFixed(3)}
                              </span>
                            </td>
                            <td className="px-3 py-3 text-center">
                              <span 
                                className="inline-block px-2 py-1 rounded text-xs font-semibold"
                                style={{ backgroundColor: obpBg, color: getTextColor(obpBg) }}
                              >
                                {team.OBP?.toFixed(3)}
                              </span>
                            </td>
                            <td className="px-3 py-3 text-center">
                              <span 
                                className="inline-block px-2 py-1 rounded text-xs font-semibold"
                                style={{ backgroundColor: slgBg, color: getTextColor(slgBg) }}
                              >
                                {team.SLG?.toFixed(3)}
                              </span>
                            </td>
                            <td className="px-3 py-3 text-center">
                              <span 
                                className="inline-block px-2 py-1 rounded text-xs font-semibold"
                                style={{ backgroundColor: opsBg, color: getTextColor(opsBg) }}
                              >
                                {team.OPS?.toFixed(3)}
                              </span>
                            </td>
                            <td className="px-3 py-3 text-center text-xs text-gray-600">{team.Conference}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                <div className="mt-6 text-sm text-gray-600 space-y-1">
                  <p><strong>WAR</strong> - Team WAR Rank | <strong>PYTHAG</strong> - Pythagorean Win Percentage</p>
                  <p><strong>ERA</strong> - Earned Run Average | <strong>WHIP</strong> - Walks + Hits per Inning Pitched | <strong>K/9</strong> - Strikeouts Per 9 Innings</p>
                  <p><strong>RPG</strong> - Runs Per Game | <strong>BA</strong> - Batting Average | <strong>OBP</strong> - On Base Percentage | <strong>SLG</strong> - Slugging | <strong>OPS</strong> - On Base Plus Slugging</p>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}