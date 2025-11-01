'use client';

import { useState, useEffect } from 'react';
import { Download, Search, Camera } from 'lucide-react';
import axios from 'axios';
import html2canvas from 'html2canvas';

const API_URL = process.env.NEXT_PUBLIC_API_URL || '';

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
  PCT: number;
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
      if (field === 'ERA' || field === 'WHIP') {
        setSortDirection('asc');
      } else {
        setSortDirection('desc');
      }
    }
  };

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
      
      if (typeof aVal === 'string' && typeof bVal === 'string') {
        return sortDirection === 'asc' ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
      }
      
      if (typeof aVal === 'number' && typeof bVal === 'number') {
        return sortDirection === 'asc' ? aVal - bVal : bVal - aVal;
      }
      
      return 0;
    });

  const downloadCSV = () => {
    const headers = ['Rank', 'Team', 'Rating', 'WAR', 'PYTHAG', 'ERA', 'WHIP', 'KP9', 'RPG', 'BA', 'OBP', 'SLG', 'OPS', 'PCT', 'Conference'];
    const csvData = filteredAndSortedStats.map((team, index) => [
      index + 1,
      team.Team,
      team.Rating?.toFixed(2) || '',
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
      team.PCT?.toFixed(3) || '',
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

  const captureScreenshot = async () => {
    const tableElement = document.querySelector('.stats-table-container');
    if (tableElement) {
      try {
        const canvas = await html2canvas(tableElement as HTMLElement, {
          scale: 2,
          backgroundColor: '#ffffff',
          logging: false,
          useCORS: true
        });
        
        // Wait a bit to ensure canvas is fully rendered
        await new Promise(resolve => setTimeout(resolve, 100));
        
        // Add watermark
        const ctx = canvas.getContext('2d');
        if (ctx) {
          const padding = 15;
          const fontSize = 32;
          ctx.font = `bold ${fontSize}px Arial`;
          const text = '@PEARatings';
          const textMetrics = ctx.measureText(text);
          const textWidth = textMetrics.width;
          const textHeight = fontSize;
          
          // Position in top-right corner
          const x = canvas.width - textWidth - padding * 2;
          const y = padding;
          
          // Draw semi-transparent white background
          ctx.fillStyle = 'rgba(255, 255, 255, 0.85)';
          ctx.fillRect(x - padding, y, textWidth + padding * 2, textHeight + padding * 2);
          
          // Draw border
          ctx.strokeStyle = 'rgba(0, 0, 0, 0.2)';
          ctx.lineWidth = 1;
          ctx.strokeRect(x - padding, y, textWidth + padding * 2, textHeight + padding * 2);
          
          // Draw text
          ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
          ctx.textAlign = 'left';
          ctx.textBaseline = 'top';
          ctx.fillText(text, x, y + padding);
        }
        
        // Small delay before download
        await new Promise(resolve => setTimeout(resolve, 100));
        
        const link = document.createElement('a');
        link.download = `cbase_stats_${new Date().toISOString().split('T')[0]}.png`;
        link.href = canvas.toDataURL('image/png');
        link.click();
      } catch (error) {
        console.error('Error capturing screenshot:', error);
      }
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-gray-900 dark:to-gray-800 pt-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex justify-between items-center mb-6">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Team Statistics</h1>
            {dataDate && (
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">Data as of {dataDate}</p>
            )}
          </div>
          <div className="flex gap-2">
            {/* <button
              onClick={captureScreenshot}
              className="flex items-center gap-2 px-4 py-2 bg-green-600 dark:bg-green-500 text-white rounded-lg hover:bg-green-700 dark:hover:bg-green-600 font-semibold transition-colors"
            >
              <Camera className="w-4 h-4" />
              Screenshot
            </button> */}
            <button
              onClick={downloadCSV}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 dark:bg-blue-500 text-white rounded-lg hover:bg-blue-700 dark:hover:bg-blue-600 font-semibold transition-colors"
            >
              <Download className="w-4 h-4" />
              Export CSV
            </button>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4 mb-6">
          <div className="flex flex-col sm:flex-row gap-4">
            <div className="flex-1 relative">
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
                <option key={conf} value={conf}>
                  {conf === 'All' ? 'All Conferences' : conf}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md overflow-hidden stats-table-container">
          <div className="p-6">
            {loading ? (
              <div className="text-center py-12">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 dark:border-blue-400 mx-auto"></div>
                <p className="mt-4 text-gray-600 dark:text-gray-400">Loading statistics...</p>
              </div>
            ) : (
              <>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead className="bg-gray-50 dark:bg-gray-700 sticky top-0">
                      <tr>
                        <th className="px-2 py-2 text-left font-semibold text-gray-700 dark:text-gray-200">Rank</th>
                        <th className="px-2 py-2 text-left font-semibold text-gray-700 dark:text-gray-200 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-600"
                          onClick={() => handleSort('Team')}>
                          Team {sortField === 'Team' && (sortDirection === 'asc' ? '↑' : '↓')}
                        </th>
                        <th className="px-1 py-2 text-center font-semibold text-gray-700 dark:text-gray-200 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-600"
                          onClick={() => handleSort('Rating')}>
                          TSR {sortField === 'Rating' && (sortDirection === 'asc' ? '↑' : '↓')}
                        </th>
                        <th className="px-1 py-2 text-center font-semibold text-gray-700 dark:text-gray-200 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-600"
                          onClick={() => handleSort('fWAR')}>
                          WAR {sortField === 'fWAR' && (sortDirection === 'asc' ? '↑' : '↓')}
                        </th>
                        <th className="px-1 py-2 text-center font-semibold text-gray-700 dark:text-gray-200 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-600"
                          onClick={() => handleSort('PYTHAG')}>
                          PYTHAG {sortField === 'PYTHAG' && (sortDirection === 'asc' ? '↑' : '↓')}
                        </th>
                        <th className="px-1 py-2 text-center font-semibold text-gray-700 dark:text-gray-200 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-600"
                          onClick={() => handleSort('ERA')}>
                          ERA {sortField === 'ERA' && (sortDirection === 'asc' ? '↑' : '↓')}
                        </th>
                        <th className="px-1 py-2 text-center font-semibold text-gray-700 dark:text-gray-200 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-600"
                          onClick={() => handleSort('WHIP')}>
                          WHIP {sortField === 'WHIP' && (sortDirection === 'asc' ? '↑' : '↓')}
                        </th>
                        <th className="px-1 py-2 text-center font-semibold text-gray-700 dark:text-gray-200 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-600"
                          onClick={() => handleSort('KP9')}>
                          K/9 {sortField === 'KP9' && (sortDirection === 'asc' ? '↑' : '↓')}
                        </th>
                        <th className="px-1 py-2 text-center font-semibold text-gray-700 dark:text-gray-200 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-600"
                          onClick={() => handleSort('RPG')}>
                          RPG {sortField === 'RPG' && (sortDirection === 'asc' ? '↑' : '↓')}
                        </th>
                        <th className="px-1 py-2 text-center font-semibold text-gray-700 dark:text-gray-200 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-600"
                          onClick={() => handleSort('BA')}>
                          BA {sortField === 'BA' && (sortDirection === 'asc' ? '↑' : '↓')}
                        </th>
                        <th className="px-1 py-2 text-center font-semibold text-gray-700 dark:text-gray-200 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-600"
                          onClick={() => handleSort('OBP')}>
                          OBP {sortField === 'OBP' && (sortDirection === 'asc' ? '↑' : '↓')}
                        </th>
                        <th className="px-1 py-2 text-center font-semibold text-gray-700 dark:text-gray-200 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-600"
                          onClick={() => handleSort('SLG')}>
                          SLG {sortField === 'SLG' && (sortDirection === 'asc' ? '↑' : '↓')}
                        </th>
                        <th className="px-1 py-2 text-center font-semibold text-gray-700 dark:text-gray-200 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-600"
                          onClick={() => handleSort('OPS')}>
                          OPS {sortField === 'OPS' && (sortDirection === 'asc' ? '↑' : '↓')}
                        </th>
                        <th className="px-1 py-2 text-center font-semibold text-gray-700 dark:text-gray-200 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-600"
                          onClick={() => handleSort('PCT')}>
                          PCT {sortField === 'PCT' && (sortDirection === 'asc' ? '↑' : '↓')}
                        </th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                      {filteredAndSortedStats.map((team, index) => {
                        const ratingBg = getRatingColor(team.Rating || 0, stats.map(s => s.Rating || 0).filter(v => v !== null), true);
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
                        const pctBg = getRatingColor(team.PCT || 0, stats.map(s => s.PCT || 0).filter(v => v !== null), true);

                        const ratingRank = getNationalRank(team.Rating, 'Rating', true);
                        const warRank = getNationalRank(team.fWAR, 'fWAR', true);
                        const pythagRank = getNationalRank(team.PYTHAG, 'PYTHAG', true);
                        const eraRank = getNationalRank(team.ERA, 'ERA', false);
                        const whipRank = getNationalRank(team.WHIP, 'WHIP', false);
                        const kp9Rank = getNationalRank(team.KP9, 'KP9', true);
                        const rpgRank = getNationalRank(team.RPG, 'RPG', true);
                        const baRank = getNationalRank(team.BA, 'BA', true);
                        const obpRank = getNationalRank(team.OBP, 'OBP', true);
                        const slgRank = getNationalRank(team.SLG, 'SLG', true);
                        const opsRank = getNationalRank(team.OPS, 'OPS', true);
                        const pctRank = getNationalRank(team.PCT, 'PCT', true);

                        return (
                          <tr key={index} className="hover:bg-gray-50 dark:hover:bg-gray-700/50">
                            <td className="px-2 py-2 text-gray-700 dark:text-gray-300 font-medium border-r-2 border-gray-300 dark:border-gray-600">{index + 1}</td>
                            <td className="px-2 py-2 font-semibold text-gray-900 dark:text-white text-sm">
                              <div className="flex items-center gap-2">
                                <img 
                                  src={`${API_URL}/api/baseball-logo/${encodeURIComponent(team.Team)}`}
                                  alt={`${team.Team} logo`}
                                  className="w-6 h-6 object-contain"
                                  onError={(e) => { e.currentTarget.style.display = 'none'; }}
                                />
                                <span>{team.Team}</span>
                              </div>
                            </td>
                            <td className="px-1 py-2 text-center">
                              <div className="flex items-center justify-center gap-1 min-w-[70px] mx-auto">
                                <span className="text-[10px] font-medium text-gray-700 dark:text-gray-300 text-right w-[32px]">
                                  {team.Rating?.toFixed(2)}
                                </span>
                                <span className="inline-flex items-center justify-center px-2 py-1 rounded text-[9px] font-semibold min-w-[35px]" style={{ backgroundColor: ratingBg, color: getTextColor(ratingBg) }}>{ratingRank}</span>
                              </div>
                            </td>
                            <td className="px-1 py-2 text-center">
                              <div className="flex items-center justify-center gap-1 min-w-[70px] mx-auto">
                                <span className="text-[10px] font-medium text-gray-700 dark:text-gray-300 text-right w-[32px]">
                                  {team.fWAR?.toFixed(2)}
                                </span>
                                <span className="inline-flex items-center justify-center px-2 py-1 rounded text-[9px] font-semibold min-w-[35px]" style={{ backgroundColor: warBg, color: getTextColor(warBg) }}>{warRank}</span>
                              </div>
                            </td>
                            <td className="px-1 py-2 text-center">
                              <div className="flex items-center justify-center gap-1 min-w-[70px] mx-auto">
                                <span className="text-[10px] font-medium text-gray-700 dark:text-gray-300 text-right w-[32px]">
                                  {team.PYTHAG?.toFixed(3)}
                                </span>
                                <span className="inline-flex items-center justify-center px-2 py-1 rounded text-[9px] font-semibold min-w-[35px]" style={{ backgroundColor: pythagBg, color: getTextColor(pythagBg) }}>{pythagRank}</span>
                              </div>
                            </td>
                            <td className="px-1 py-2 text-center">
                              <div className="flex items-center justify-center gap-1 min-w-[70px] mx-auto">
                                <span className="text-[10px] font-medium text-gray-700 dark:text-gray-300 text-right w-[32px]">
                                  {team.ERA?.toFixed(2)}
                                </span>
                                <span className="inline-flex items-center justify-center px-2 py-1 rounded text-[9px] font-semibold min-w-[35px]" style={{ backgroundColor: eraBg, color: getTextColor(eraBg) }}>{eraRank}</span>
                              </div>
                            </td>
                            <td className="px-1 py-2 text-center">
                              <div className="flex items-center justify-center gap-1 min-w-[70px] mx-auto">
                                <span className="text-[10px] font-medium text-gray-700 dark:text-gray-300 text-right w-[32px]">
                                  {team.WHIP?.toFixed(2)}
                                </span>
                                <span className="inline-flex items-center justify-center px-2 py-1 rounded text-[9px] font-semibold min-w-[35px]" style={{ backgroundColor: whipBg, color: getTextColor(whipBg) }}>{whipRank}</span>
                              </div>
                            </td>
                            <td className="px-1 py-2 text-center">
                              <div className="flex items-center justify-center gap-1 min-w-[70px] mx-auto">
                                <span className="text-[10px] font-medium text-gray-700 dark:text-gray-300 text-right w-[32px]">
                                  {team.KP9?.toFixed(1)}
                                </span>
                                <span className="inline-flex items-center justify-center px-2 py-1 rounded text-[9px] font-semibold min-w-[35px]" style={{ backgroundColor: kp9Bg, color: getTextColor(kp9Bg) }}>{kp9Rank}</span>
                              </div>
                            </td>
                            <td className="px-1 py-2 text-center">
                              <div className="flex items-center justify-center gap-1 min-w-[70px] mx-auto">
                                <span className="text-[10px] font-medium text-gray-700 dark:text-gray-300 text-right w-[32px]">
                                  {team.RPG?.toFixed(1)}
                                </span>
                                <span className="inline-flex items-center justify-center px-2 py-1 rounded text-[9px] font-semibold min-w-[35px]" style={{ backgroundColor: rpgBg, color: getTextColor(rpgBg) }}>{rpgRank}</span>
                              </div>
                            </td>
                            <td className="px-1 py-2 text-center">
                              <div className="flex items-center justify-center gap-1 min-w-[70px] mx-auto">
                                <span className="text-[10px] font-medium text-gray-700 dark:text-gray-300 text-right w-[32px]">
                                  {team.BA?.toFixed(3)}
                                </span>
                                <span className="inline-flex items-center justify-center px-2 py-1 rounded text-[9px] font-semibold min-w-[35px]" style={{ backgroundColor: baBg, color: getTextColor(baBg) }}>{baRank}</span>
                              </div>
                            </td>
                            <td className="px-1 py-2 text-center">
                              <div className="flex items-center justify-center gap-1 min-w-[70px] mx-auto">
                                <span className="text-[10px] font-medium text-gray-700 dark:text-gray-300 text-right w-[32px]">
                                  {team.OBP?.toFixed(3)}
                                </span>
                                <span className="inline-flex items-center justify-center px-2 py-1 rounded text-[9px] font-semibold min-w-[35px]" style={{ backgroundColor: obpBg, color: getTextColor(obpBg) }}>{obpRank}</span>
                              </div>
                            </td>
                            <td className="px-1 py-2 text-center">
                              <div className="flex items-center justify-center gap-1 min-w-[70px] mx-auto">
                                <span className="text-[10px] font-medium text-gray-700 dark:text-gray-300 text-right w-[32px]">
                                  {team.SLG?.toFixed(3)}
                                </span>
                                <span className="inline-flex items-center justify-center px-2 py-1 rounded text-[9px] font-semibold min-w-[35px]" style={{ backgroundColor: slgBg, color: getTextColor(slgBg) }}>{slgRank}</span>
                              </div>
                            </td>
                            <td className="px-1 py-2 text-center">
                              <div className="flex items-center justify-center gap-1 min-w-[70px] mx-auto">
                                <span className="text-[10px] font-medium text-gray-700 dark:text-gray-300 text-right w-[32px]">
                                  {team.OPS?.toFixed(3)}
                                </span>
                                <span className="inline-flex items-center justify-center px-2 py-1 rounded text-[9px] font-semibold min-w-[35px]" style={{ backgroundColor: opsBg, color: getTextColor(opsBg) }}>{opsRank}</span>
                              </div>
                            </td>
                            <td className="px-1 py-2 text-center">
                              <div className="flex items-center justify-center gap-1 min-w-[70px] mx-auto">
                                <span className="text-[10px] font-medium text-gray-700 dark:text-gray-300 text-right w-[32px]">
                                  {team.PCT?.toFixed(3)}
                                </span>
                                <span className="inline-flex items-center justify-center px-2 py-1 rounded text-[9px] font-semibold min-w-[35px]" style={{ backgroundColor: pctBg, color: getTextColor(pctBg) }}>{pctRank}</span>
                              </div>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                <div className="mt-6 text-sm text-gray-600 dark:text-gray-400 space-y-1 border-t dark:border-gray-700 pt-4">
                  <p><strong className="dark:text-gray-300">@PEARatings</strong></p>
                  <p><strong className="dark:text-gray-300">TSR</strong> - Team Strength Rating | <strong className="dark:text-gray-300">WAR</strong> - Team WAR Rank | <strong className="dark:text-gray-300">PYTHAG</strong> - Pythagorean Win Percentage | <strong className="dark:text-gray-300">ERA</strong> - Earned Run Average | <strong className="dark:text-gray-300">WHIP</strong> - Walks + Hits per Inning Pitched | <strong className="dark:text-gray-300">K/9</strong> - Strikeouts Per 9 Innings | <strong className="dark:text-gray-300">RPG</strong> - Runs Per Game | <strong className="dark:text-gray-300">BA</strong> - Batting Average | <strong className="dark:text-gray-300">OBP</strong> - On Base Percentage | <strong className="dark:text-gray-300">SLG</strong> - Slugging | <strong className="dark:text-gray-300">OPS</strong> - On Base Plus Slugging | <strong className="dark:text-gray-300">PCT</strong> - Fielding Percentage </p>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}