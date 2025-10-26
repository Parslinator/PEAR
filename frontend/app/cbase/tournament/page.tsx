'use client';

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Trophy, Download, Filter, Search, Calendar, TrendingUp } from 'lucide-react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface TournamentOutlook {
  regionals: Array<{
    regional_number: number;
    host: string;
    seed_1: string;
    seed_2: string;
    seed_3: string;
    seed_4: string;
  }>;
  last_four_in: string[];
  first_four_out: string[];
  next_four_out: string[];
  multibid_conferences: { [key: string]: number };
}

interface SimulationResult {
  team: string;
  win_probability: number;
}

export default function CbaseTournamentPage() {
  const [teams, setTeams] = useState<string[]>([]);
  const [tournamentOutlook, setTournamentOutlook] = useState<TournamentOutlook | null>(null);
  const [simulationResult, setSimulationResult] = useState<SimulationResult[] | null>(null);
  const [simulationImage, setSimulationImage] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [simulationLoading, setSimulationLoading] = useState(false);
  const [error, setError] = useState('');

  // Simulation form state
  const [seed1, setSeed1] = useState('');
  const [seed2, setSeed2] = useState('');
  const [seed3, setSeed3] = useState('');
  const [seed4, setSeed4] = useState('');

  // Search/filter states
  const [seed1SearchTerm, setSeed1SearchTerm] = useState('');
  const [seed2SearchTerm, setSeed2SearchTerm] = useState('');
  const [seed3SearchTerm, setSeed3SearchTerm] = useState('');
  const [seed4SearchTerm, setSeed4SearchTerm] = useState('');
  const [showSeed1Dropdown, setShowSeed1Dropdown] = useState(false);
  const [showSeed2Dropdown, setShowSeed2Dropdown] = useState(false);
  const [showSeed3Dropdown, setShowSeed3Dropdown] = useState(false);
  const [showSeed4Dropdown, setShowSeed4Dropdown] = useState(false);

  useEffect(() => {
    fetchTournamentData();
    fetchTeams();
  }, []);

  const fetchTeams = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/cbase/teams`);
      setTeams(response.data.teams);
    } catch (error) {
      console.error('Error fetching teams:', error);
    }
  };

  const fetchTournamentData = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API_URL}/api/cbase/tournament-outlook`);
      setTournamentOutlook(response.data);
    } catch (error) {
      setError('Error loading tournament data. Please try again.');
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  };

  const simulateRegional = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!seed1 || !seed2 || !seed3 || !seed4) {
      setError('Please select all four seeds');
      return;
    }

    const seeds = [seed1, seed2, seed3, seed4];
    if (new Set(seeds).size !== 4) {
      setError('Please select four different teams');
      return;
    }

    setSimulationLoading(true);
    setError('');
    setSimulationResult(null);
    setSimulationImage(null);

    try {
      const response = await axios.post(`${API_URL}/api/cbase/simulate-regional`, {
        seed_1: seed1,
        seed_2: seed2,
        seed_3: seed3,
        seed_4: seed4
      }, {
        responseType: 'blob'
      });

      const imageObjectUrl = URL.createObjectURL(response.data);
      setSimulationImage(imageObjectUrl);
    } catch (error) {
      setError('Error simulating regional. Please try again.');
      console.error('Error:', error);
    } finally {
      setSimulationLoading(false);
    }
  };

  // Filter teams based on search
  const filteredSeed1Teams = teams.filter(team =>
    team.toLowerCase().includes(seed1SearchTerm.toLowerCase())
  );

  const filteredSeed2Teams = teams.filter(team =>
    team.toLowerCase().includes(seed2SearchTerm.toLowerCase())
  );

  const filteredSeed3Teams = teams.filter(team =>
    team.toLowerCase().includes(seed3SearchTerm.toLowerCase())
  );

  const filteredSeed4Teams = teams.filter(team =>
    team.toLowerCase().includes(seed4SearchTerm.toLowerCase())
  );

  const downloadCSV = () => {
    if (!tournamentOutlook) return;

    const headers = ['Regional', 'Host', '2 Seed', '3 Seed', '4 Seed'];
    const csvData = tournamentOutlook.regionals.map(regional => [
      regional.regional_number,
      regional.host,
      regional.seed_2,
      regional.seed_3,
      regional.seed_4
    ]);

    const csvContent = [
      headers.join(','),
      ...csvData.map(row => row.join(','))
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'tournament_outlook.csv';
    a.click();
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-900 via-blue-800 to-blue-900 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-yellow-400 mx-auto"></div>
          <p className="text-white mt-4 text-xl">Loading Tournament Data...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-900 via-blue-800 to-blue-900 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center mb-6">
            <Trophy className="w-16 h-16 text-yellow-400 mr-4" />
            <h1 className="text-5xl font-bold text-white">NCAA Tournament</h1>
          </div>
          <p className="text-xl text-blue-200">Projected Bracket & Regional Simulator</p>
        </div>

        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
            {error}
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Tournament Outlook */}
          <div className="bg-white rounded-lg shadow-xl p-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-2xl font-bold text-gray-900">Tournament Outlook</h2>
              <button
                onClick={downloadCSV}
                className="flex items-center px-3 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 text-sm"
              >
                <Download className="w-4 h-4 mr-1" />
                CSV
              </button>
            </div>

            {tournamentOutlook && (
              <>
                {/* Regionals Table */}
                <div className="overflow-x-auto max-h-[500px] overflow-y-auto border rounded-lg mb-4">
                  <table className="w-full text-sm">
                    <thead className="bg-gray-50 sticky top-0">
                      <tr>
                        <th className="px-3 py-3 text-center font-semibold text-gray-700">#</th>
                        <th className="px-3 py-3 text-left font-semibold text-gray-700">Host</th>
                        <th className="px-3 py-3 text-left font-semibold text-gray-700">2 Seed</th>
                        <th className="px-3 py-3 text-left font-semibold text-gray-700">3 Seed</th>
                        <th className="px-3 py-3 text-left font-semibold text-gray-700">4 Seed</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-200">
                      {tournamentOutlook.regionals.map((regional) => (
                        <tr key={regional.regional_number} className="hover:bg-gray-50">
                          <td className="px-3 py-3 text-center font-semibold text-gray-700">
                            {regional.regional_number}
                          </td>
                          <td className="px-3 py-3 font-bold text-gray-900">{regional.host}</td>
                          <td className="px-3 py-3 text-gray-700">{regional.seed_2}</td>
                          <td className="px-3 py-3 text-gray-700">{regional.seed_3}</td>
                          <td className="px-3 py-3 text-gray-700">{regional.seed_4}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {/* Bubble Teams */}
                <div className="space-y-2 text-sm">
                  <div className="bg-green-50 border border-green-200 rounded p-3">
                    <p className="font-semibold text-green-800 mb-1">Last 4 In</p>
                    <p className="text-green-700">{tournamentOutlook.last_four_in.join(', ')}</p>
                  </div>
                  <div className="bg-yellow-50 border border-yellow-200 rounded p-3">
                    <p className="font-semibold text-yellow-800 mb-1">First 4 Out</p>
                    <p className="text-yellow-700">{tournamentOutlook.first_four_out.join(', ')}</p>
                  </div>
                  <div className="bg-red-50 border border-red-200 rounded p-3">
                    <p className="font-semibold text-red-800 mb-1">Next 4 Out</p>
                    <p className="text-red-700">{tournamentOutlook.next_four_out.join(', ')}</p>
                  </div>
                </div>

                {/* Multi-bid Conferences */}
                <div className="mt-4 bg-blue-50 border border-blue-200 rounded p-3">
                  <p className="font-semibold text-blue-800 mb-2">Multi-Bid Conferences</p>
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(tournamentOutlook.multibid_conferences).map(([conf, count]) => (
                      <span key={conf} className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-xs font-semibold">
                        {conf}: {count}
                      </span>
                    ))}
                  </div>
                </div>
              </>
            )}
          </div>

          {/* Simulate Regional */}
          <div className="bg-white rounded-lg shadow-xl p-6">
            <h2 className="text-2xl font-bold text-gray-900 mb-4">Simulate Regional</h2>

            <form onSubmit={simulateRegional} className="space-y-4">
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">1 Seed (Host)</label>
                <div className="relative">
                  <input
                    type="text"
                    value={seed1 || seed1SearchTerm}
                    onChange={(e) => {
                      setSeed1SearchTerm(e.target.value);
                      setSeed1('');
                      setShowSeed1Dropdown(true);
                    }}
                    onFocus={() => setShowSeed1Dropdown(true)}
                    onBlur={() => setTimeout(() => setShowSeed1Dropdown(false), 200)}
                    placeholder="Search for a team..."
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                  {showSeed1Dropdown && filteredSeed1Teams.length > 0 && (
                    <div className="absolute z-10 w-full mt-1 bg-white border border-gray-300 rounded-lg shadow-lg max-h-60 overflow-y-auto">
                      {filteredSeed1Teams.map(team => (
                        <div
                          key={team}
                          onClick={() => {
                            setSeed1(team);
                            setSeed1SearchTerm('');
                            setShowSeed1Dropdown(false);
                          }}
                          className="px-3 py-2 hover:bg-blue-50 cursor-pointer"
                        >
                          {team}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">2 Seed</label>
                <div className="relative">
                  <input
                    type="text"
                    value={seed2 || seed2SearchTerm}
                    onChange={(e) => {
                      setSeed2SearchTerm(e.target.value);
                      setSeed2('');
                      setShowSeed2Dropdown(true);
                    }}
                    onFocus={() => setShowSeed2Dropdown(true)}
                    onBlur={() => setTimeout(() => setShowSeed2Dropdown(false), 200)}
                    placeholder="Search for a team..."
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                  {showSeed2Dropdown && filteredSeed2Teams.length > 0 && (
                    <div className="absolute z-10 w-full mt-1 bg-white border border-gray-300 rounded-lg shadow-lg max-h-60 overflow-y-auto">
                      {filteredSeed2Teams.map(team => (
                        <div
                          key={team}
                          onClick={() => {
                            setSeed2(team);
                            setSeed2SearchTerm('');
                            setShowSeed2Dropdown(false);
                          }}
                          className="px-3 py-2 hover:bg-blue-50 cursor-pointer"
                        >
                          {team}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">3 Seed</label>
                <div className="relative">
                  <input
                    type="text"
                    value={seed3 || seed3SearchTerm}
                    onChange={(e) => {
                      setSeed3SearchTerm(e.target.value);
                      setSeed3('');
                      setShowSeed3Dropdown(true);
                    }}
                    onFocus={() => setShowSeed3Dropdown(true)}
                    onBlur={() => setTimeout(() => setShowSeed3Dropdown(false), 200)}
                    placeholder="Search for a team..."
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                  {showSeed3Dropdown && filteredSeed3Teams.length > 0 && (
                    <div className="absolute z-10 w-full mt-1 bg-white border border-gray-300 rounded-lg shadow-lg max-h-60 overflow-y-auto">
                      {filteredSeed3Teams.map(team => (
                        <div
                          key={team}
                          onClick={() => {
                            setSeed3(team);
                            setSeed3SearchTerm('');
                            setShowSeed3Dropdown(false);
                          }}
                          className="px-3 py-2 hover:bg-blue-50 cursor-pointer"
                        >
                          {team}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">4 Seed</label>
                <div className="relative">
                  <input
                    type="text"
                    value={seed4 || seed4SearchTerm}
                    onChange={(e) => {
                      setSeed4SearchTerm(e.target.value);
                      setSeed4('');
                      setShowSeed4Dropdown(true);
                    }}
                    onFocus={() => setShowSeed4Dropdown(true)}
                    onBlur={() => setTimeout(() => setShowSeed4Dropdown(false), 200)}
                    placeholder="Search for a team..."
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                  {showSeed4Dropdown && filteredSeed4Teams.length > 0 && (
                    <div className="absolute z-10 w-full mt-1 bg-white border border-gray-300 rounded-lg shadow-lg max-h-60 overflow-y-auto">
                      {filteredSeed4Teams.map(team => (
                        <div
                          key={team}
                          onClick={() => {
                            setSeed4(team);
                            setSeed4SearchTerm('');
                            setShowSeed4Dropdown(false);
                          }}
                          className="px-3 py-2 hover:bg-blue-50 cursor-pointer"
                        >
                          {team}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              <button
                type="submit"
                disabled={simulationLoading}
                className="w-full bg-blue-600 text-white py-3 px-4 rounded-lg hover:bg-blue-700 font-semibold disabled:bg-gray-400 disabled:cursor-not-allowed"
              >
                {simulationLoading ? 'Simulating...' : 'Simulate Regional'}
              </button>
            </form>

            {/* Simulation Result */}
            {simulationImage && (
              <div className="mt-6">
                <h3 className="text-lg font-bold text-gray-900 mb-3">Regional Simulation Results</h3>
                <div className="flex justify-center">
                  <img 
                    src={simulationImage} 
                    alt="Regional Simulation" 
                    className="max-w-full h-auto rounded-lg shadow-lg"
                  />
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Info Section */}
        <div className="mt-6 bg-blue-50 border border-blue-200 rounded-lg p-4 text-sm text-blue-800">
          <p className="font-semibold mb-2">About the Tournament Outlook:</p>
          <ul className="space-y-1 list-disc list-inside">
            <li><strong>Tournament Outlook:</strong> Projects the 16 regional hosts and their opponents based on current NET rankings and automatic qualifiers</li>
            <li><strong>Regional Simulation:</strong> Uses PEAR ratings to simulate a double-elimination regional tournament 5,000 times with home field advantage</li>
            <li><strong>Bubble Teams:</strong> Last 4 In are the final at-large teams, First/Next 4 Out are the teams just outside the field</li>
            <li><strong>Conflict Resolution:</strong> Algorithm minimizes conference matchups in the first round by intelligently swapping seeds</li>
          </ul>
        </div>
      </div>
    </div>
  );
}