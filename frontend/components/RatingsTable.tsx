'use client';

import { useState } from 'react';
import { ChevronUp, ChevronDown } from 'lucide-react';

interface RatingData {
  Team: string;
  Rating: number;
  offensive_rating: number;
  defensive_rating: number;
  MD: number;
  SOS: number;
  SOR: number;
  CONF: string;
}

interface Props {
  data: RatingData[];
}

export default function RatingsTable({ data }: Props) {
  const [sortField, setSortField] = useState<keyof RatingData>('Rating');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');
  const [filter, setFilter] = useState('');

  const handleSort = (field: keyof RatingData) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      // For lower-is-better columns, default to ascending
      const lowerIsBetter = ['defensive_rating', 'MD', 'SOS', 'SOR'].includes(field);
      setSortDirection(lowerIsBetter ? 'asc' : 'desc');
    }
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

  const filteredData = sortedData.filter(item =>
    item.Team.toLowerCase().includes(filter.toLowerCase()) ||
    item.CONF.toLowerCase().includes(filter.toLowerCase())
  );

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
    let normalized = (value - min) / range; // 0 to 1
    
    // If lower is better (defense), invert the normalization
    if (!higherIsBetter) {
      normalized = 1 - normalized;
    }
    
    // Colors: Dark Blue #00008B (0, 0, 139), Light Gray #D3D3D3 (211, 211, 211), Dark Red #8B0000 (139, 0, 0)
    if (normalized >= 0.5) {
      // Top half: Gray to Dark Blue
      const t = (normalized - 0.5) * 2; // 0 to 1 in top half
      const r = Math.round(211 + (0 - 211) * t);
      const g = Math.round(211 + (0 - 211) * t);
      const b = Math.round(211 + (139 - 211) * t);
      return `rgb(${r}, ${g}, ${b})`;
    } else {
      // Bottom half: Dark Red to Gray
      const t = normalized * 2; // 0 to 1 in bottom half
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
    const headers = ['Rank', 'Team', 'Rating', 'OFF', 'DEF', 'MD', 'SOR', 'SOS', 'CONF'];
    const csvData = filteredData.map((item, index) => [
      index + 1,
      item.Team,
      item.Rating.toFixed(1),
      item.offensive_rating.toFixed(1),
      item.defensive_rating.toFixed(1),
      item.MD,
      Math.round(item.SOR),
      Math.round(item.SOS),
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

  return (
    <div>
      <div className="mb-4 flex gap-4">
        <input
          type="text"
          placeholder="Search teams or conferences..."
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        />
        <button
          onClick={downloadCSV}
          className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-semibold"
        >
          Download CSV
        </button>
      </div>

      <div className="overflow-x-auto max-h-[500px] overflow-y-auto">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 sticky top-0 z-10">
            <tr>
              <th className="px-4 py-3 text-left font-semibold text-gray-700">Rank</th>
              <th
                className="px-4 py-3 text-left font-semibold text-gray-700 cursor-pointer hover:bg-gray-100"
                onClick={() => handleSort('Team')}
              >
                Team <SortIcon field="Team" />
              </th>
              <th
                className="px-4 py-3 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100"
                onClick={() => handleSort('Rating')}
              >
                Rating <SortIcon field="Rating" />
              </th>
              <th
                className="px-4 py-3 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100"
                onClick={() => handleSort('offensive_rating')}
              >
                OFF <SortIcon field="offensive_rating" />
              </th>
              <th
                className="px-4 py-3 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100"
                onClick={() => handleSort('defensive_rating')}
              >
                DEF <SortIcon field="defensive_rating" />
              </th>
              <th
                className="px-4 py-3 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100"
                onClick={() => handleSort('MD')}
              >
                MD <SortIcon field="MD" />
              </th>
              <th
                className="px-4 py-3 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100"
                onClick={() => handleSort('SOR')}
              >
                SOR <SortIcon field="SOR" />
              </th>
              <th
                className="px-4 py-3 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100"
                onClick={() => handleSort('SOS')}
              >
                SOS <SortIcon field="SOS" />
              </th>
              <th
                className="px-4 py-3 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100"
                onClick={() => handleSort('CONF')}
              >
                CONF <SortIcon field="CONF" />
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200">
            {filteredData.map((item, index) => {
              // Higher is better: Rating, OFF
              const ratingBg = getRatingColor(item.Rating, data.map(d => d.Rating), true);
              const offBg = getRatingColor(item.offensive_rating, data.map(d => d.offensive_rating), true);
              
              // Lower is better: DEF, SOS, MD, SOR
              const defBg = getRatingColor(item.defensive_rating, data.map(d => d.defensive_rating), false);
              const mdBg = getRatingColor(item.MD, data.map(d => d.MD), false);
              const sosBg = getRatingColor(item.SOS, data.map(d => d.SOS), false);
              const sorBg = getRatingColor(item.SOR, data.map(d => d.SOR), false);

              return (
                <tr key={index} className="hover:bg-gray-50">
                  <td className="px-4 py-3 text-gray-700 font-medium">{index + 1}</td>
                  <td className="px-4 py-3 font-semibold text-gray-900">{item.Team}</td>
                  <td className="px-4 py-3 text-center">
                    <span 
                      className="inline-block px-3 py-1 rounded-full font-semibold text-xs"
                      style={{ backgroundColor: ratingBg, color: getTextColor(ratingBg) }}
                    >
                      {item.Rating.toFixed(1)}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-center">
                    <span 
                      className="inline-block px-2 py-1 rounded text-xs font-semibold"
                      style={{ backgroundColor: offBg, color: getTextColor(offBg) }}
                    >
                      {item.offensive_rating.toFixed(1)}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-center">
                    <span 
                      className="inline-block px-2 py-1 rounded text-xs font-semibold"
                      style={{ backgroundColor: defBg, color: getTextColor(defBg) }}
                    >
                      {item.defensive_rating.toFixed(1)}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-center">
                    <span 
                      className="inline-block px-2 py-1 rounded text-xs font-semibold"
                      style={{ backgroundColor: mdBg, color: getTextColor(mdBg) }}
                    >
                      {item.MD}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-center">
                    <span 
                      className="inline-block px-2 py-1 rounded text-xs font-semibold"
                      style={{ backgroundColor: sorBg, color: getTextColor(sorBg) }}
                    >
                      {Math.round(item.SOR)}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-center">
                    <span 
                      className="inline-block px-2 py-1 rounded text-xs font-semibold"
                      style={{ backgroundColor: sosBg, color: getTextColor(sosBg) }}
                    >
                      {Math.round(item.SOS)}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-center text-xs text-gray-600">{item.CONF}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}