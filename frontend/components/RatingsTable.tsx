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

  const getRankColor = (rank: number, total: number = 136) => {
    const percentage = rank / total;
    if (percentage <= 0.25) return 'bg-green-100 text-green-800';
    if (percentage <= 0.5) return 'bg-blue-100 text-blue-800';
    if (percentage <= 0.75) return 'bg-yellow-100 text-yellow-800';
    return 'bg-red-100 text-red-800';
  };

  return (
    <div>
      <div className="mb-4">
        <input
          type="text"
          placeholder="Search teams or conferences..."
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        />
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
                  <span className="inline-block px-3 py-1 rounded-full bg-blue-100 text-blue-800 font-semibold">
                    {item.Rating.toFixed(2)}
                  </span>
                </td>
                <td className="px-4 py-3 text-center">
                  <span className={`inline-block px-2 py-1 rounded ${getRankColor(item.MD)}`}>
                    {item.MD}
                  </span>
                </td>
                <td className="px-4 py-3 text-center text-gray-700">{item.SOS.toFixed(2)}</td>
                <td className="px-4 py-3 text-center text-gray-700">{item.SOR.toFixed(2)}</td>
                <td className="px-4 py-3 text-center">
                  <span className={`inline-block px-2 py-1 rounded text-xs ${getRankColor(item.OFF)}`}>
                    {item.OFF}
                  </span>
                </td>
                <td className="px-4 py-3 text-center">
                  <span className={`inline-block px-2 py-1 rounded text-xs ${getRankColor(item.DEF)}`}>
                    {item.DEF}
                  </span>
                </td>
                <td className="px-4 py-3 text-center">
                  <span className={`inline-block px-2 py-1 rounded text-xs ${getRankColor(item.PBR)}`}>
                    {item.PBR}
                  </span>
                </td>
                <td className="px-4 py-3 text-center">
                  <span className={`inline-block px-2 py-1 rounded text-xs ${getRankColor(item.DCE)}`}>
                    {item.DCE}
                  </span>
                </td>
                <td className="px-4 py-3 text-center">
                  <span className={`inline-block px-2 py-1 rounded text-xs ${getRankColor(item.DDE)}`}>
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