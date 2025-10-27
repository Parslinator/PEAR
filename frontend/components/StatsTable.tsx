'use client';

import { useState, useMemo } from 'react';
import { ChevronUp, ChevronDown } from 'lucide-react';

interface StatsData {
  team: string;
  power_rating: number;
  offensive_rating: number;
  Offense_successRate_adj: number;
  Offense_ppa_adj: number;
  Offense_rushing_adj: number;
  Offense_passing_adj: number;
  adj_offense_ppo: number;
  adj_offense_drive_quality: number;
  defensive_rating: number;
  Defense_successRate_adj: number;
  Defense_ppa_adj: number;
  Defense_rushing_adj: number;
  Defense_passing_adj: number;
  adj_defense_ppo: number;
  adj_defense_drive_quality: number;
  conference?: string;
}

interface Props {
  data: StatsData[];
}

type SortField = keyof StatsData;

export default function StatsTable({ data }: Props) {
  const [sortField, setSortField] = useState<SortField>('power_rating');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');
  const [filter, setFilter] = useState('');
  const [conferenceFilter, setConferenceFilter] = useState('All');

  // Get unique conferences from data
  const conferences = useMemo(() => {
    const uniqueConfs = Array.from(
      new Set(data.map(item => item.conference).filter(Boolean))
    ).sort();
    return ['All', ...uniqueConfs];
  }, [data]);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      const isDefenseStat = field.toString().includes('defense') || field.toString().includes('Defense');
      setSortDirection(isDefenseStat ? 'asc' : 'desc');
    }
  };

  // Calculate national rank for a team's stat value
  const getNationalRank = (value: number, field: SortField, higherIsBetter: boolean = true): number => {
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
    const matchesSearch = item.team.toLowerCase().includes(filter.toLowerCase()) ||
      (item.conference && item.conference.toLowerCase().includes(filter.toLowerCase()));
    const matchesConference = conferenceFilter === 'All' || item.conference === conferenceFilter;
    return matchesSearch && matchesConference;
  });

  const SortIcon = ({ field }: { field: SortField }) => {
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
    const headers = [
      'Rank', 'Team', 'Rating',
      'OFF Rating', 'Success Rate', 'PPA', 'Rushing', 'Passing', 'PPO', 'Drive Quality',
      'DEF Rating', 'Success Rate', 'PPA', 'Rushing', 'Passing', 'PPO', 'Drive Quality',
      'Conference'
    ];
    const csvData = filteredData.map((item, index) => [
      index + 1,
      item.team,
      item.power_rating.toFixed(1),
      item.offensive_rating.toFixed(1),
      item.Offense_successRate_adj.toFixed(2),
      item.Offense_ppa_adj.toFixed(2),
      item.Offense_rushing_adj.toFixed(2),
      item.Offense_passing_adj.toFixed(2),
      item.adj_offense_ppo.toFixed(2),
      item.adj_offense_drive_quality.toFixed(1),
      item.defensive_rating.toFixed(1),
      item.Defense_successRate_adj.toFixed(2),
      item.Defense_ppa_adj.toFixed(2),
      item.Defense_rushing_adj.toFixed(2),
      item.Defense_passing_adj.toFixed(2),
      item.adj_defense_ppo.toFixed(2),
      item.adj_defense_drive_quality.toFixed(1),
      item.conference || ''
    ]);
    
    const csvContent = [
      headers.join(','),
      ...csvData.map(row => row.join(','))
    ].join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'pear_team_stats.csv';
    a.click();
    window.URL.revokeObjectURL(url);
  };

  return (
    <div>
      <div className="mb-4 flex gap-4">
        <input
          type="text"
          placeholder="Search teams..."
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        />
        {conferences.length > 1 && (
          <select
            value={conferenceFilter}
            onChange={(e) => setConferenceFilter(e.target.value)}
            className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white"
          >
            {conferences.map(conf => (
              <option key={conf} value={conf}>
                {conf === 'All' ? 'All Conferences' : conf}
              </option>
            ))}
          </select>
        )}
        <button
          onClick={downloadCSV}
          className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-semibold"
        >
          Download CSV
        </button>
      </div>

      <div className="overflow-x-auto max-h-[600px] overflow-y-auto">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 sticky top-0 z-10">
            <tr>
              <th className="px-3 py-2 text-left font-semibold text-gray-700 border-r-2 border-gray-300">Rank</th>
              <th
                className="px-3 py-2 text-left font-semibold text-gray-700 cursor-pointer hover:bg-gray-100"
                onClick={() => handleSort('team')}
              >
                Team <SortIcon field="team" />
              </th>
              <th
                className="px-2 py-2 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100 border-r-2 border-gray-300"
                onClick={() => handleSort('power_rating')}
              >
                Rating <SortIcon field="power_rating" />
              </th>
              
              <th colSpan={7} className="px-2 py-2 text-center font-bold text-gray-700 bg-green-50 border-r-2 border-gray-300">
                OFFENSE
              </th>
              
              <th colSpan={7} className="px-2 py-2 text-center font-bold text-gray-700 bg-blue-50">
                DEFENSE
              </th>
            </tr>
            <tr>
              <th className="px-3 py-2 border-r-2 border-gray-300"></th>
              <th className="px-3 py-2"></th>
              <th className="px-2 py-2 border-r-2 border-gray-300"></th>
              
              <th
                className="px-2 py-2 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100 bg-green-50"
                onClick={() => handleSort('offensive_rating')}
              >
                Rating <SortIcon field="offensive_rating" />
              </th>
              <th
                className="px-2 py-2 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100 bg-green-50"
                onClick={() => handleSort('Offense_successRate_adj')}
              >
                Success <SortIcon field="Offense_successRate_adj" />
              </th>
              <th
                className="px-2 py-2 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100 bg-green-50"
                onClick={() => handleSort('Offense_ppa_adj')}
              >
                PPA <SortIcon field="Offense_ppa_adj" />
              </th>
              <th
                className="px-2 py-2 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100 bg-green-50"
                onClick={() => handleSort('Offense_rushing_adj')}
              >
                Rush <SortIcon field="Offense_rushing_adj" />
              </th>
              <th
                className="px-2 py-2 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100 bg-green-50"
                onClick={() => handleSort('Offense_passing_adj')}
              >
                Pass <SortIcon field="Offense_passing_adj" />
              </th>
              <th
                className="px-2 py-2 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100 bg-green-50"
                onClick={() => handleSort('adj_offense_ppo')}
              >
                PPO <SortIcon field="adj_offense_ppo" />
              </th>
              <th
                className="px-2 py-2 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100 bg-green-50 border-r-2 border-gray-300"
                onClick={() => handleSort('adj_offense_drive_quality')}
              >
                Drive Q <SortIcon field="adj_offense_drive_quality" />
              </th>
              
              <th
                className="px-2 py-2 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100 bg-blue-50"
                onClick={() => handleSort('defensive_rating')}
              >
                Rating <SortIcon field="defensive_rating" />
              </th>
              <th
                className="px-2 py-2 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100 bg-blue-50"
                onClick={() => handleSort('Defense_successRate_adj')}
              >
                Success <SortIcon field="Defense_successRate_adj" />
              </th>
              <th
                className="px-2 py-2 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100 bg-blue-50"
                onClick={() => handleSort('Defense_ppa_adj')}
              >
                PPA <SortIcon field="Defense_ppa_adj" />
              </th>
              <th
                className="px-2 py-2 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100 bg-blue-50"
                onClick={() => handleSort('Defense_rushing_adj')}
              >
                Rush <SortIcon field="Defense_rushing_adj" />
              </th>
              <th
                className="px-2 py-2 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100 bg-blue-50"
                onClick={() => handleSort('Defense_passing_adj')}
              >
                Pass <SortIcon field="Defense_passing_adj" />
              </th>
              <th
                className="px-2 py-2 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100 bg-blue-50"
                onClick={() => handleSort('adj_defense_ppo')}
              >
                PPO <SortIcon field="adj_defense_ppo" />
              </th>
              <th
                className="px-2 py-2 text-center font-semibold text-gray-700 cursor-pointer hover:bg-gray-100 bg-blue-50"
                onClick={() => handleSort('adj_defense_drive_quality')}
              >
                Drive Q <SortIcon field="adj_defense_drive_quality" />
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200">
            {filteredData.map((item, index) => {
              const ratingBg = getRatingColor(item.power_rating, data.map(d => d.power_rating), true);
              const ratingText = getTextColor(ratingBg);
              
              const offRatingBg = getRatingColor(item.offensive_rating, data.map(d => d.offensive_rating), true);
              const offSuccessBg = getRatingColor(item.Offense_successRate_adj, data.map(d => d.Offense_successRate_adj), true);
              const offPPABg = getRatingColor(item.Offense_ppa_adj, data.map(d => d.Offense_ppa_adj), true);
              const offRushBg = getRatingColor(item.Offense_rushing_adj, data.map(d => d.Offense_rushing_adj), true);
              const offPassBg = getRatingColor(item.Offense_passing_adj, data.map(d => d.Offense_passing_adj), true);
              const offPPOBg = getRatingColor(item.adj_offense_ppo, data.map(d => d.adj_offense_ppo), true);
              const offDQBg = getRatingColor(item.adj_offense_drive_quality, data.map(d => d.adj_offense_drive_quality), true);
              
              const defRatingBg = getRatingColor(item.defensive_rating, data.map(d => d.defensive_rating), false);
              const defSuccessBg = getRatingColor(item.Defense_successRate_adj, data.map(d => d.Defense_successRate_adj), false);
              const defPPABg = getRatingColor(item.Defense_ppa_adj, data.map(d => d.Defense_ppa_adj), false);
              const defRushBg = getRatingColor(item.Defense_rushing_adj, data.map(d => d.Defense_rushing_adj), false);
              const defPassBg = getRatingColor(item.Defense_passing_adj, data.map(d => d.Defense_passing_adj), false);
              const defPPOBg = getRatingColor(item.adj_defense_ppo, data.map(d => d.adj_defense_ppo), false);
              const defDQBg = getRatingColor(item.adj_defense_drive_quality, data.map(d => d.adj_defense_drive_quality), false);

              // Get national ranks
              const ratingRank = getNationalRank(item.power_rating, 'power_rating', true);
              const offRatingRank = getNationalRank(item.offensive_rating, 'offensive_rating', true);
              const offSuccessRank = getNationalRank(item.Offense_successRate_adj, 'Offense_successRate_adj', true);
              const offPPARank = getNationalRank(item.Offense_ppa_adj, 'Offense_ppa_adj', true);
              const offRushRank = getNationalRank(item.Offense_rushing_adj, 'Offense_rushing_adj', true);
              const offPassRank = getNationalRank(item.Offense_passing_adj, 'Offense_passing_adj', true);
              const offPPORank = getNationalRank(item.adj_offense_ppo, 'adj_offense_ppo', true);
              const offDQRank = getNationalRank(item.adj_offense_drive_quality, 'adj_offense_drive_quality', true);
              
              const defRatingRank = getNationalRank(item.defensive_rating, 'defensive_rating', false);
              const defSuccessRank = getNationalRank(item.Defense_successRate_adj, 'Defense_successRate_adj', false);
              const defPPARank = getNationalRank(item.Defense_ppa_adj, 'Defense_ppa_adj', false);
              const defRushRank = getNationalRank(item.Defense_rushing_adj, 'Defense_rushing_adj', false);
              const defPassRank = getNationalRank(item.Defense_passing_adj, 'Defense_passing_adj', false);
              const defPPORank = getNationalRank(item.adj_defense_ppo, 'adj_defense_ppo', false);
              const defDQRank = getNationalRank(item.adj_defense_drive_quality, 'adj_defense_drive_quality', false);

              return (
                <tr key={index} className="hover:bg-gray-50">
                  <td className="px-2 py-2 text-gray-700 font-medium border-r-2 border-gray-300">{index + 1}</td>
                  <td className="px-2 py-2 font-semibold text-gray-900 text-sm">{item.team}</td>
                  <td className="px-1 py-2 text-center border-r-2 border-gray-300">
                    <div className="flex items-center justify-center gap-1 min-w-[70px] mx-auto">
                      <span className="text-[10px] font-medium text-gray-700 text-right w-[32px]">
                        {item.power_rating.toFixed(1)}
                      </span>
                      <span className="inline-flex items-center justify-center px-2 py-1 rounded text-[9px] font-semibold min-w-[35px]" style={{ backgroundColor: ratingBg, color: ratingText }}>{ratingRank}</span>
                    </div>
                  </td>
                  
                  <td className="px-1 py-2 text-center">
                    <div className="flex items-center justify-center gap-1 min-w-[70px] mx-auto">
                      <span className="text-[10px] font-medium text-gray-700 text-right w-[32px]">
                        {item.offensive_rating.toFixed(1)}
                      </span>
                      <span className="inline-flex items-center justify-center px-2 py-1 rounded text-[9px] font-semibold min-w-[35px]" style={{ backgroundColor: offRatingBg, color: getTextColor(offRatingBg) }}>{offRatingRank}</span>
                    </div>
                  </td>
                  <td className="px-1 py-2 text-center">
                    <div className="flex items-center justify-center gap-1 min-w-[70px] mx-auto">
                      <span className="text-[10px] font-medium text-gray-700 text-right w-[32px]">
                        {item.Offense_successRate_adj.toFixed(2)}
                      </span>
                      <span className="inline-flex items-center justify-center px-2 py-1 rounded text-[9px] font-semibold min-w-[35px]" style={{ backgroundColor: offSuccessBg, color: getTextColor(offSuccessBg) }}>{offSuccessRank}</span>
                    </div>
                  </td>
                  <td className="px-1 py-2 text-center">
                    <div className="flex items-center justify-center gap-1 min-w-[70px] mx-auto">
                      <span className="text-[10px] font-medium text-gray-700 text-right w-[32px]">
                        {item.Offense_ppa_adj.toFixed(2)}
                      </span>
                      <span className="inline-flex items-center justify-center px-2 py-1 rounded text-[9px] font-semibold min-w-[35px]" style={{ backgroundColor: offPPABg, color: getTextColor(offPPABg) }}>{offPPARank}</span>
                    </div>
                  </td>
                  <td className="px-1 py-2 text-center">
                    <div className="flex items-center justify-center gap-1 min-w-[70px] mx-auto">
                      <span className="text-[10px] font-medium text-gray-700 text-right w-[32px]">
                        {item.Offense_rushing_adj.toFixed(2)}
                      </span>
                      <span className="inline-flex items-center justify-center px-2 py-1 rounded text-[9px] font-semibold min-w-[35px]" style={{ backgroundColor: offRushBg, color: getTextColor(offRushBg) }}>{offRushRank}</span>
                    </div>
                  </td>
                  <td className="px-1 py-2 text-center">
                    <div className="flex items-center justify-center gap-1 min-w-[70px] mx-auto">
                      <span className="text-[10px] font-medium text-gray-700 text-right w-[32px]">
                        {item.Offense_passing_adj.toFixed(2)}
                      </span>
                      <span className="inline-flex items-center justify-center px-2 py-1 rounded text-[9px] font-semibold min-w-[35px]" style={{ backgroundColor: offPassBg, color: getTextColor(offPassBg) }}>{offPassRank}</span>
                    </div>
                  </td>
                  <td className="px-1 py-2 text-center">
                    <div className="flex items-center justify-center gap-1 min-w-[70px] mx-auto">
                      <span className="text-[10px] font-medium text-gray-700 text-right w-[32px]">
                        {item.adj_offense_ppo.toFixed(2)}
                      </span>
                      <span className="inline-flex items-center justify-center px-2 py-1 rounded text-[9px] font-semibold min-w-[35px]" style={{ backgroundColor: offPPOBg, color: getTextColor(offPPOBg) }}>{offPPORank}</span>
                    </div>
                  </td>
                  <td className="px-1 py-2 text-center border-r-2 border-gray-300">
                    <div className="flex items-center justify-center gap-1 min-w-[70px] mx-auto">
                      <span className="text-[10px] font-medium text-gray-700 text-right w-[32px]">
                        {item.adj_offense_drive_quality.toFixed(1)}
                      </span>
                      <span className="inline-flex items-center justify-center px-2 py-1 rounded text-[9px] font-semibold min-w-[35px]" style={{ backgroundColor: offDQBg, color: getTextColor(offDQBg) }}>{offDQRank}</span>
                    </div>
                  </td>
                  
                  <td className="px-1 py-2 text-center">
                    <div className="flex items-center justify-center gap-1 min-w-[70px] mx-auto">
                      <span className="text-[10px] font-medium text-gray-700 text-right w-[32px]">
                        {item.defensive_rating.toFixed(1)}
                      </span>
                      <span className="inline-flex items-center justify-center px-2 py-1 rounded text-[9px] font-semibold min-w-[35px]" style={{ backgroundColor: defRatingBg, color: getTextColor(defRatingBg) }}>{defRatingRank}</span>
                    </div>
                  </td>
                  <td className="px-1 py-2 text-center">
                    <div className="flex items-center justify-center gap-1 min-w-[70px] mx-auto">
                      <span className="text-[10px] font-medium text-gray-700 text-right w-[32px]">
                        {item.Defense_successRate_adj.toFixed(2)}
                      </span>
                      <span className="inline-flex items-center justify-center px-2 py-1 rounded text-[9px] font-semibold min-w-[35px]" style={{ backgroundColor: defSuccessBg, color: getTextColor(defSuccessBg) }}>{defSuccessRank}</span>
                    </div>
                  </td>
                  <td className="px-1 py-2 text-center">
                    <div className="flex items-center justify-center gap-1 min-w-[70px] mx-auto">
                      <span className="text-[10px] font-medium text-gray-700 text-right w-[32px]">
                        {item.Defense_ppa_adj.toFixed(2)}
                      </span>
                      <span className="inline-flex items-center justify-center px-2 py-1 rounded text-[9px] font-semibold min-w-[35px]" style={{ backgroundColor: defPPABg, color: getTextColor(defPPABg) }}>{defPPARank}</span>
                    </div>
                  </td>
                  <td className="px-1 py-2 text-center">
                    <div className="flex items-center justify-center gap-1 min-w-[70px] mx-auto">
                      <span className="text-[10px] font-medium text-gray-700 text-right w-[32px]">
                        {item.Defense_rushing_adj.toFixed(2)}
                      </span>
                      <span className="inline-flex items-center justify-center px-2 py-1 rounded text-[9px] font-semibold min-w-[35px]" style={{ backgroundColor: defRushBg, color: getTextColor(defRushBg) }}>{defRushRank}</span>
                    </div>
                  </td>
                  <td className="px-1 py-2 text-center">
                    <div className="flex items-center justify-center gap-1 min-w-[70px] mx-auto">
                      <span className="text-[10px] font-medium text-gray-700 text-right w-[32px]">
                        {item.Defense_passing_adj.toFixed(2)}
                      </span>
                      <span className="inline-flex items-center justify-center px-2 py-1 rounded text-[9px] font-semibold min-w-[35px]" style={{ backgroundColor: defPassBg, color: getTextColor(defPassBg) }}>{defPassRank}</span>
                    </div>
                  </td>
                  <td className="px-1 py-2 text-center">
                    <div className="flex items-center justify-center gap-1 min-w-[70px] mx-auto">
                      <span className="text-[10px] font-medium text-gray-700 text-right w-[32px]">
                        {item.adj_defense_ppo.toFixed(2)}
                      </span>
                      <span className="inline-flex items-center justify-center px-2 py-1 rounded text-[9px] font-semibold min-w-[35px]" style={{ backgroundColor: defPPOBg, color: getTextColor(defPPOBg) }}>{defPPORank}</span>
                    </div>
                  </td>
                  <td className="px-1 py-2 text-center">
                    <div className="flex items-center justify-center gap-1 min-w-[80px] mx-auto">
                      <span className="text-[10px] font-medium text-gray-700 text-right w-[32px]">
                        {item.adj_defense_drive_quality.toFixed(1)}
                      </span>
                      <span className="inline-flex items-center justify-center px-2 py-1 rounded text-[9px] font-semibold min-w-[35px]" style={{ backgroundColor: defDQBg, color: getTextColor(defDQBg) }}>{defDQRank}</span>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}