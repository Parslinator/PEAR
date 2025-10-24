'use client';

import { useState } from 'react';

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

export default function SpreadsTable({ data }: Props) {
  const [sortBy, setSortBy] = useState<'difference' | 'GQI'>('difference');

  const sortedData = [...data].sort((a, b) => {
    if (sortBy === 'difference') {
      return b.difference - a.difference;
    }
    return b.GQI - a.GQI;
  });

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

  return (
    <div>
      <div className="mb-4 flex gap-2">
        <button
          onClick={() => setSortBy('difference')}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            sortBy === 'difference'
              ? 'bg-green-600 text-white'
              : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
          }`}
        >
          Sort by Difference
        </button>
        <button
          onClick={() => setSortBy('GQI')}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            sortBy === 'GQI'
              ? 'bg-green-600 text-white'
              : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
          }`}
        >
          Sort by Game Quality
        </button>
      </div>

      <div className="overflow-x-auto max-h-[500px] overflow-y-auto">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 sticky top-0">
            <tr>
              <th className="px-4 py-3 text-left font-semibold text-gray-700">Matchup</th>
              <th className="px-4 py-3 text-center font-semibold text-gray-700">PEAR</th>
              <th className="px-4 py-3 text-center font-semibold text-gray-700">Vegas</th>
              <th className="px-4 py-3 text-center font-semibold text-gray-700">Diff</th>
              <th className="px-4 py-3 text-center font-semibold text-gray-700">GQI</th>
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
      </div>
    </div>
  );
}