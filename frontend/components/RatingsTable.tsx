'use client';

import { useState } from 'react';
import { ChevronUp, ChevronDown } from 'lucide-react';

interface RatingData {
  Team: string;
  Rating: number;
  MD: number;
  SOS: number;
  SOR: number;
  OFF: number;
  DEF: number;
  PBR: number;
  DCE: number;
  DDE: number;
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
      setSortDirection('desc');
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

  const getRatingColor = (rating: number, allRatings: number[]) => {
    const max = Math.max(...allRatings);
    const min = Math.min(...allRatings);
    const range = max - min;
    const position = (rating - min) / range;
    
    if (position >= 0.75) return 'bg-blue-600 text-white';
    if (position >= 0.5) return 'bg-blue-400 text-white';
    if (position >= 0.25) return 'bg-gray-400 text-white';
    return 'bg-red-600 text-white';
  };

  const getStatColor = (rank: number, total: number = 136) => {
    const percentage = rank / total;
    if (percentage <= 0.25) return 'bg-blue-600 text-white';
    if (percentage <= 0.5) return 'bg-blue-400 text-white';
    if (percentage <= 0.75) return 'bg-gray-400 text-white';
    return 'bg-red-600 text-white';
  };

  const downloadCSV = () => {
    const headers = ['Rank', 'Team', 'Rating', 'MD', 'SOS', 'SOR', 'OFF', 'DEF', 'PBR', 'DCE', 'DDE', 'CONF'];
    const csvData = filteredData.map((item, index) => [
      index + 1,
      item.Team,
      item.Rating.toFixed(2),
      item.MD,
      Math.round(item.SOS),
      Math.round(item.SOR),
      item.OFF,
      item.DEF,
      item.PBR,
      item.DCE,
      item.DDE,
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

  const allRatings = data.map(d => d.Rating);

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
                onClick={() => handleSort('MD')}
              >
                MD <SortIcon field="MD" />
              </th>
              <th
                className="px-4 py-3 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100"
                onClick={() => handleSort('SOS')}
              >
                SOS <SortIcon field="SOS" />
              </th>
              <th
                className="px-4 py-3 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100"
                onClick={() => handleSort('SOR')}
              >
                SOR <SortIcon field="SOR" />
              </th>
              <th
                className="px-4 py-3 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100"
                onClick={() => handleSort('OFF')}
              >
                OFF <SortIcon field="OFF" />
              </th>
              <th
                className="px-4 py-3 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100"
                onClick={() => handleSort('DEF')}
              >
                DEF <SortIcon field="DEF" />
              </th>
              <th
                className="px-4 py-3 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100"
                onClick={() => handleSort('PBR')}
              >
                PBR <SortIcon field="PBR" />
              </th>
              <th
                className="px-4 py-3 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100"
                onClick={() => handleSort('DCE')}
              >
                DCE <SortIcon field="DCE" />
              </th>
              <th
                className="px-4 py-3 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100"
                onClick={() => handleSort('DDE')}
              >
                DDE <SortIcon field="DDE" />
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
            {filteredData.map((item, index) => (
              <tr key={index} className="hover:bg-gray-50">
                <td className="px-4 py-3 text-gray-700 font-medium">{index + 1}</td>
                <td className="px-4 py-3 font-semibold text-gray-900">{item.Team}</td>
                <td className="px-4 py-3 text-center">
                  <span className={`inline-block px-3 py-1 rounded-full font-semibold ${getRatingColor(item.Rating, allRatings)}`}>
                    {item.Rating.toFixed(2)}
                  </span>
                </td>
                <td className="px-4 py-3 text-center">
                  <span className={`inline-block px-2 py-1 rounded ${getStatColor(item.MD)}`}>
                    {item.MD}
                  </span>
                </td>
                <td className="px-4 py-3 text-center">
                  <span className={`inline-block px-2 py-1 rounded ${getStatColor(Math.round(item.SOS))}`}>
                    {Math.round(item.SOS)}
                  </span>
                </td>
                <td className="px-4 py-3 text-center">
                  <span className={`inline-block px-2 py-1 rounded ${getStatColor(Math.round(item.SOR))}`}>
                    {Math.round(item.SOR)}
                  </span>
                </td>
                <td className="px-4 py-3 text-center">
                  <span className={`inline-block px-2 py-1 rounded text-xs ${getStatColor(item.OFF)}`}>
                    {item.OFF}
                  </span>
                </td>
                <td className="px-4 py-3 text-center">
                  <span className={`inline-block px-2 py-1 rounded text-xs ${getStatColor(item.DEF)}`}>
                    {item.DEF}
                  </span>
                </td>
                <td className="px-4 py-3 text-center">
                  <span className={`inline-block px-2 py-1 rounded text-xs ${getStatColor(item.PBR)}`}>
                    {item.PBR}
                  </span>
                </td>
                <td className="px-4 py-3 text-center">
                  <span className={`inline-block px-2 py-1 rounded text-xs ${getStatColor(item.DCE)}`}>
                    {item.DCE}
                  </span>
                </td>
                <td className="px-4 py-3 text-center">
                  <span className={`inline-block px-2 py-1 rounded text-xs ${getStatColor(item.DDE)}`}>
                    {item.DDE}
                  </span>
                </td>
                <td className="px-4 py-3 text-center text-xs text-gray-600">{item.CONF}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}