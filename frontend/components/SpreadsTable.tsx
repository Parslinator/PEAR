'use client';

import { useState } from 'react';
import { ChevronUp, ChevronDown } from 'lucide-react';

interface SpreadData {
  home_team: string;
  away_team: string;
  PEAR: string;
  Vegas: string;
  difference: number;
  GQI: number;
  pr_spread: number;
}

interface Props {
  data: SpreadData[];
}

type SortField = 'matchup' | 'difference' | 'GQI' | 'pr_spread';

export default function SpreadsTable({ data }: Props) {
  const [sortBy, setSortBy] = useState<SortField>('difference');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');

  const handleSort = (field: SortField) => {
    if (sortBy === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(field);
      setSortDirection('desc');
    }
  };

  const sortedData = [...data].sort((a, b) => {
    let aVal, bVal;
    
    if (sortBy === 'matchup') {
      aVal = a.away_team;
      bVal = b.away_team;
      return sortDirection === 'asc' 
        ? aVal.localeCompare(bVal)
        : bVal.localeCompare(aVal);
    } else if (sortBy === 'difference') {
      aVal = a.difference;
      bVal = b.difference;
    } else if (sortBy === 'GQI') {
      aVal = a.GQI;
      bVal = b.GQI;
    } else {
      aVal = a.pr_spread;
      bVal = b.pr_spread;
    }
    
    return sortDirection === 'asc' ? aVal - bVal : bVal - aVal;
  });

  const SortIcon = ({ field }: { field: SortField }) => {
    if (sortBy !== field) return null;
    return sortDirection === 'asc' ? (
      <ChevronUp className="w-4 h-4 inline ml-1" />
    ) : (
      <ChevronDown className="w-4 h-4 inline ml-1" />
    );
  };

  const getGQIColor = (gqi: number) => {
    if (gqi >= 8) return 'bg-green-600 text-white';
    if (gqi >= 6) return 'bg-green-500 text-white';
    if (gqi >= 4) return 'bg-yellow-500 text-white';
    return 'bg-gray-400 text-white';
  };

  const getDifferenceColor = (diff: number) => {
    if (diff >= 5) return 'bg-red-100 text-red-800 border-red-300';
    if (diff >= 3) return 'bg-orange-100 text-orange-800 border-orange-300';
    if (diff >= 1) return 'bg-yellow-100 text-yellow-800 border-yellow-300';
    return 'bg-green-100 text-green-800 border-green-300';
  };

  const downloadCSV = () => {
    const headers = ['Away Team', 'Home Team', 'PEAR', 'Vegas', 'PEAR Raw', 'Difference', 'GQI'];
    const csvData = sortedData.map(item => [
      item.away_team,
      item.home_team,
      item.PEAR,
      item.pr_spread,
      item.Vegas,
      item.difference.toFixed(1),
      item.GQI.toFixed(1)
    ]);
    
    const csvContent = [
      headers.join(','),
      ...csvData.map(row => row.join(','))
    ].join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'pear_spreads.csv';
    a.click();
    window.URL.revokeObjectURL(url);
  };

  return (
    <div>
      <div className="mb-4 flex justify-end">
        <button
          onClick={downloadCSV}
          className="px-6 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors font-semibold"
        >
          Download CSV
        </button>
      </div>

      <div className="overflow-x-auto max-h-[600px] overflow-y-auto">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 sticky top-0">
            <tr>
              <th 
                className="px-4 py-3 text-left font-semibold text-gray-700 cursor-pointer hover:bg-gray-100"
                onClick={() => handleSort('matchup')}
              >
                Matchup <SortIcon field="matchup" />
              </th>
                            <th 
                className="px-4 py-3 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100"
                onClick={() => handleSort('pr_spread')}
              >
                PEAR <SortIcon field="pr_spread" />
              </th>
              <th 
                className="px-4 py-3 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100"
                onClick={() => handleSort('pr_spread')}
              >
                Vegas <SortIcon field="pr_spread" />
              </th>
                            <th 
                className="px-4 py-3 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100"
                onClick={() => handleSort('pr_spread')}
              >
                PEAR Raw <SortIcon field="pr_spread" />
              </th>
              <th 
                className="px-4 py-3 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100"
                onClick={() => handleSort('difference')}
              >
                Diff <SortIcon field="difference" />
              </th>
              <th 
                className="px-4 py-3 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100"
                onClick={() => handleSort('GQI')}
              >
                GQI <SortIcon field="GQI" />
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200">
            {sortedData.map((item, index) => (
              <tr key={index} className="hover:bg-gray-50">
                <td className="px-4 py-3">
                  <div className="font-semibold text-gray-900">{item.away_team}</div>
                  <div className="text-xs text-gray-600">at {item.home_team}</div>
                </td>
                <td className="px-4 py-3 text-center">
                  <span className="inline-block px-3 py-1 bg-blue-100 text-blue-800 rounded font-medium">
                    {item.PEAR}
                  </span>
                </td>
                <td className="px-4 py-3 text-center">
                  <span className="inline-block px-3 py-1 bg-gray-100 text-gray-800 rounded font-medium">
                    {item.Vegas}
                  </span>
                </td>
                <td className="px-4 py-3 text-center">
                  <span className="inline-block px-3 py-1 bg-purple-100 text-purple-800 rounded font-medium">
                    {item.pr_spread}
                  </span>
                </td>
                <td className="px-4 py-3 text-center">
                  <span className={`inline-block px-3 py-1 rounded font-bold border-2 ${getDifferenceColor(item.difference)}`}>
                    {item.difference.toFixed(1)}
                  </span>
                </td>
                <td className="px-4 py-3 text-center">
                  <span className={`inline-block px-3 py-1 rounded-full font-bold ${getGQIColor(item.GQI)}`}>
                    {item.GQI.toFixed(1)}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="mt-4 text-xs text-gray-600 space-y-1">
        <p><strong>GQI</strong> - Game Quality Index (1-10 scale, higher is better)</p>
        <p><strong>Diff</strong> - Absolute difference between PEAR and Vegas spreads</p>
        <p><strong>PEAR Raw</strong> - Raw PEAR spread before formatting</p>
      </div>
    </div>
  );
}