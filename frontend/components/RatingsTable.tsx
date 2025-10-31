'use client';

import { useState, useMemo } from 'react';
import { ChevronUp, ChevronDown, Download, X } from 'lucide-react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || '';

interface RatingData {
  Team: string;
  Rating: number;
  offensive_rating: number;
  defensive_rating: number;
  MD: number;
  SOS: number;
  CONF: string;
}

interface Props {
  data: RatingData[];
  year: number;
  week: number;
}

export default function RatingsTable({ data, year, week }: Props) {
  const [sortField, setSortField] = useState<keyof RatingData>('Rating');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');
  const [filter, setFilter] = useState('');
  const [conferenceFilter, setConferenceFilter] = useState('All');
  const [selectedTeam, setSelectedTeam] = useState<string | null>(null);

  // Get unique conferences from data
  const conferences = useMemo(() => {
    const uniqueConfs = Array.from(new Set(data.map(item => item.CONF))).sort();
    return ['All', ...uniqueConfs];
  }, [data]);

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

  const handleSort = (field: keyof RatingData) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      // For lower-is-better columns, default to ascending
      const lowerIsBetter = ['defensive_rating', 'SOS'].includes(field);
      setSortDirection(lowerIsBetter ? 'asc' : 'desc');
    }
  };

  // Calculate national rank for a team's stat value
  const getNationalRank = (value: number, field: keyof RatingData, higherIsBetter: boolean = true): number => {
    if (value === null || value === undefined) return data.length;
    
    const sortedValues = data
      .map(d => d[field] as number)
      .filter(v => v !== null && v !== undefined)
      .sort((a, b) => higherIsBetter ? b - a : a - b);
    
    return sortedValues.indexOf(value) + 1;
  };

  const sortedData = [...data].sort((a, b) => {
    const aVal = a[sortField];
    const bVal = b[sortField];
    
    if (typeof aVal === 'number' && typeof bVal === 'number') {
      return sortDirection === 'asc' ? aVal - bVal : bVal - aVal;
    }
    
    return sortDirection === 'asc'
      ? String(aVal).localeCompare(String(bVal))
      : String(bVal).localeCompare(String(aVal));
  });

  const filteredData = sortedData.filter(item => {
    const matchesSearch = item.Team.toLowerCase().includes(filter.toLowerCase()) ||
      item.CONF.toLowerCase().includes(filter.toLowerCase());
    const matchesConference = conferenceFilter === 'All' || item.CONF === conferenceFilter;
    return matchesSearch && matchesConference;
  });

  const SortIcon = ({ field }: { field: keyof RatingData }) => {
    if (sortField !== field) return null;
    return sortDirection === 'asc' ? (
      <ChevronUp className="w-4 h-4 inline ml-1" />
    ) : (
      <ChevronDown className="w-4 h-4 inline ml-1" />
    );
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

  const downloadCSV = () => {
    const headers = ['Rank', 'Team', 'Rating', 'OFF', 'DEF', 'MD', 'SOS', 'CONF'];
    const csvData = filteredData.map((item, index) => [
      index + 1,
      item.Team,
      item.Rating.toFixed(1),
      item.offensive_rating.toFixed(1),
      item.defensive_rating.toFixed(1),
      item.MD.toFixed(3),
      item.SOS.toFixed(2),
      item.CONF
    ]);
    
    const csvContent = [
      headers.join(','),
      ...csvData.map(row => row.join(','))
    ].join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'pear_ratings.csv';
    a.click();
    window.URL.revokeObjectURL(url);
  };

  const handleLogoClick = (teamName: string) => {
    setSelectedTeam(teamName);
  };

  const closeModal = () => {
    setSelectedTeam(null);
  };

  return (
    <div>
      <div className="mb-4 flex gap-4">
        <input
          type="text"
          placeholder="Search teams or conferences..."
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="flex-1 px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400"
        />
        <button
          onClick={downloadCSV}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 dark:bg-blue-500 text-white rounded-lg hover:bg-blue-700 dark:hover:bg-blue-600 font-semibold transition-colors"
        >
          <Download className="w-4 h-4" />
          Export CSV
        </button>
      </div>

      {/* Conference Filter Buttons */}
      <div className="mb-4 flex flex-wrap gap-2 justify-center">
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

      <div className="overflow-x-auto max-h-[500px] overflow-y-auto">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 dark:bg-gray-700 sticky top-0 z-10">
            <tr>
              <th className="px-4 py-3 text-left font-semibold text-gray-700 dark:text-gray-200">Rank</th>
              <th
                className="px-4 py-3 text-left font-semibold text-gray-700 dark:text-gray-200 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-600"
                onClick={() => handleSort('Team')}
              >
                Team <SortIcon field="Team" />
              </th>
              <th
                className="px-4 py-3 text-center font-semibold text-gray-700 dark:text-gray-200 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-600"
                onClick={() => handleSort('Rating')}
              >
                Rating <SortIcon field="Rating" />
              </th>
              <th
                className="px-4 py-3 text-center font-semibold text-gray-700 dark:text-gray-200 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-600"
                onClick={() => handleSort('offensive_rating')}
              >
                OFF <SortIcon field="offensive_rating" />
              </th>
              <th
                className="px-4 py-3 text-center font-semibold text-gray-700 dark:text-gray-200 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-600"
                onClick={() => handleSort('defensive_rating')}
              >
                DEF <SortIcon field="defensive_rating" />
              </th>
              <th
                className="px-4 py-3 text-center font-semibold text-gray-700 dark:text-gray-200 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-600"
                onClick={() => handleSort('MD')}
              >
                MD <SortIcon field="MD" />
              </th>
              <th
                className="px-4 py-3 text-center font-semibold text-gray-700 dark:text-gray-200 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-600"
                onClick={() => handleSort('SOS')}
              >
                SOS <SortIcon field="SOS" />
              </th>
              <th
                className="px-4 py-3 text-center font-semibold text-gray-700 dark:text-gray-200 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-600"
                onClick={() => handleSort('CONF')}
              >
                CONF <SortIcon field="CONF" />
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
            {filteredData.map((item, index) => {
              // Higher is better: Rating, OFF, MD
              const ratingBg = getRatingColor(item.Rating, data.map(d => d.Rating), true);
              const offBg = getRatingColor(item.offensive_rating, data.map(d => d.offensive_rating), true);
              const mdBg = getRatingColor(item.MD, data.map(d => d.MD), true);
              
              // Lower is better: DEF, SOS
              const defBg = getRatingColor(item.defensive_rating, data.map(d => d.defensive_rating), false);
              const sosBg = getRatingColor(item.SOS, data.map(d => d.SOS), false);

              // Get national ranks for Rating, OFF, DEF, MD, and SOS
              const ratingRank = getNationalRank(item.Rating, 'Rating', true);
              const offRank = getNationalRank(item.offensive_rating, 'offensive_rating', true);
              const defRank = getNationalRank(item.defensive_rating, 'defensive_rating', false);
              const mdRank = getNationalRank(item.MD, 'MD', true);
              const sosRank = getNationalRank(item.SOS, 'SOS', false);

              return (
                <tr key={index} className="hover:bg-gray-50 dark:hover:bg-gray-700/50">
                  <td className="px-4 py-3 text-gray-700 dark:text-gray-300 font-medium">{index + 1}</td>
                  <td className="px-4 py-3 font-semibold text-gray-900 dark:text-white">
                    <div className="flex items-center gap-2">
                      <img 
                        src={`${API_URL}/api/football-logo/${encodeURIComponent(item.Team)}`}
                        alt={`${item.Team} logo`}
                        className="w-6 h-6 object-contain cursor-pointer hover:opacity-75 transition-opacity"
                        onClick={() => handleLogoClick(item.Team)}
                        onError={(e) => {
                          e.currentTarget.style.display = 'none';
                        }}
                      />
                      <span>{item.Team}</span>
                    </div>
                  </td>
                  <td className="px-4 py-3 text-center">
                    <div className="flex items-center justify-center gap-2 min-w-[80px] mx-auto">
                      <span className="text-[11px] font-medium text-gray-700 dark:text-gray-300 text-right w-[35px]">
                        {item.Rating.toFixed(1)}
                      </span>
                      <span className="inline-flex items-center justify-center px-2 py-1 rounded-full text-[10px] font-semibold min-w-[38px]" style={{ backgroundColor: ratingBg, color: getTextColor(ratingBg) }}>{ratingRank}</span>
                    </div>
                  </td>
                  <td className="px-4 py-3 text-center">
                    <div className="flex items-center justify-center gap-2 min-w-[80px] mx-auto">
                      <span className="text-[11px] font-medium text-gray-700 dark:text-gray-300 text-right w-[35px]">
                        {item.offensive_rating.toFixed(1)}
                      </span>
                      <span className="inline-flex items-center justify-center px-2 py-1 rounded text-[10px] font-semibold min-w-[38px]" style={{ backgroundColor: offBg, color: getTextColor(offBg) }}>{offRank}</span>
                    </div>
                  </td>
                  <td className="px-4 py-3 text-center">
                    <div className="flex items-center justify-center gap-2 min-w-[80px] mx-auto">
                      <span className="text-[11px] font-medium text-gray-700 dark:text-gray-300 text-right w-[35px]">
                        {item.defensive_rating.toFixed(1)}
                      </span>
                      <span className="inline-flex items-center justify-center px-2 py-1 rounded text-[10px] font-semibold min-w-[38px]" style={{ backgroundColor: defBg, color: getTextColor(defBg) }}>{defRank}</span>
                    </div>
                  </td>
                  <td className="px-4 py-3 text-center">
                    <div className="flex items-center justify-center gap-2 min-w-[80px] mx-auto">
                      <span className="text-[11px] font-medium text-gray-700 dark:text-gray-300 text-right w-[35px]">
                        {item.MD.toFixed(3)}
                      </span>
                      <span className="inline-flex items-center justify-center px-2 py-1 rounded text-[10px] font-semibold min-w-[38px]" style={{ backgroundColor: mdBg, color: getTextColor(mdBg) }}>{mdRank}</span>
                    </div>
                  </td>
                  <td className="px-4 py-3 text-center">
                    <div className="flex items-center justify-center gap-2 min-w-[80px] mx-auto">
                      <span className="text-[11px] font-medium text-gray-700 dark:text-gray-300 text-right w-[35px]">
                        {item.SOS.toFixed(2)}
                      </span>
                      <span className="inline-flex items-center justify-center px-2 py-1 rounded text-[10px] font-semibold min-w-[38px]" style={{ backgroundColor: sosBg, color: getTextColor(sosBg) }}>{sosRank}</span>
                    </div>
                  </td>
                  <td className="px-4 py-3 text-center text-xs text-gray-600 dark:text-gray-400">{item.CONF}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Modal for Team Profile */}
      {selectedTeam && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4"
          onClick={closeModal}
        >
          <div 
            className="relative max-w-6xl max-h-[90vh] bg-white dark:bg-gray-800 rounded-lg shadow-2xl overflow-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              onClick={closeModal}
              className="absolute top-4 right-4 z-10 bg-white dark:bg-gray-700 rounded-full p-2 shadow-lg hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors"
            >
              <X className="w-6 h-6 text-gray-700 dark:text-gray-200" />
            </button>
            <div className="p-4">
              <h2 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">{selectedTeam} Profile</h2>
              <img
                src={`${API_URL}/api/team-profile/${year}/${week}/${encodeURIComponent(selectedTeam)}`}
                alt={`${selectedTeam} profile`}
                className="w-full h-auto"
                onError={(e) => {
                  e.currentTarget.src = '';
                  e.currentTarget.alt = 'Profile image not available';
                }}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}