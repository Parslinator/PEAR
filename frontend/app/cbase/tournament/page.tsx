'use client';

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Trophy, Download, Filter, Search, Calendar, TrendingUp, AlertCircle, Target } from 'lucide-react';

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
  automatic_qualifiers?: string[];
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

  // Helper function to determine team status and color
  const getTeamColor = (team: string) => {
    if (!tournamentOutlook) return 'blue';
    
    // Check if automatic qualifier (green)
    if (tournamentOutlook.automatic_qualifiers?.includes(team)) {
      return 'green';
    }
    
    // Check if last four in (orange)
    if (tournamentOutlook.last_four_in.includes(team)) {
      return 'orange';
    }
    
    // Regular at-large team (blue)
    return 'blue';
  };

  // Pair regionals: 1 vs 16, 2 vs 15, etc.
  const getRegionalPairs = () => {
    if (!tournamentOutlook) return [];
    
    const regionals = [...tournamentOutlook.regionals].sort((a, b) => a.regional_number - b.regional_number);
    const pairs = [];
    
    for (let i = 0; i < 8; i++) {
      pairs.push({
        regional1: regionals[i],
        regional2: regionals[15 - i]
      });
    }
    
    return pairs;
  };

  // Component to render a single regional
  const RegionalCard = ({ regional }: { regional: any }) => {
    const seed1Color = getTeamColor(regional.seed_1);
    const seed2Color = getTeamColor(regional.seed_2);
    const seed3Color = getTeamColor(regional.seed_3);
    const seed4Color = getTeamColor(regional.seed_4);

    const colorClasses = {
      green: {
        bg: 'bg-green-50',
        border: 'border-green-600',
        text: 'text-green-700',
        badge: 'bg-green-600'
      },
      blue: {
        bg: 'bg-blue-50',
        border: 'border-blue-500',
        text: 'text-blue-600',
        badge: 'bg-blue-600'
      },
      orange: {
        bg: 'bg-orange-50',
        border: 'border-orange-500',
        text: 'text-orange-600',
        badge: 'bg-orange-600'
      }
    };

    return (
      <div className="border-2 border-gray-200 rounded-lg hover:border-blue-400 hover:shadow-md transition-all">
        {/* Regional Header */}
        <div className="bg-gradient-to-r from-gray-800 to-gray-700 text-white px-4 py-2 rounded-t-lg">
          <div className="flex items-center justify-between">
            <span className="font-bold">{regional.host} Regional</span>
            <span className="text-xs bg-white/20 px-2 py-1 rounded">Seed #{regional.regional_number}</span>
          </div>
        </div>

        {/* Regional Seeds */}
        <div className="p-3 space-y-1">
          {/* 1 Seed - Host */}
          <div className={`flex items-center gap-2 ${colorClasses[seed1Color].bg} border-l-4 ${colorClasses[seed1Color].border} px-3 py-2 rounded`}>
            <span className={`font-bold ${colorClasses[seed1Color].text} text-sm w-6`}>1</span>
            <span className="font-bold text-gray-900 flex-1">{regional.seed_1}</span>
            <span className={`text-xs ${colorClasses[seed1Color].badge} text-white px-2 py-0.5 rounded font-semibold`}>
              {seed1Color === 'green' ? 'AQ' : 'HOST'}
            </span>
          </div>

          {/* 2 Seed */}
          <div className={`flex items-center gap-2 ${colorClasses[seed2Color].bg} border-l-4 ${colorClasses[seed2Color].border} px-3 py-2 rounded`}>
            <span className={`font-bold ${colorClasses[seed2Color].text} text-sm w-6`}>2</span>
            <span className="text-gray-900 flex-1">{regional.seed_2}</span>
            {seed2Color === 'green' && (
              <span className="text-xs bg-green-600 text-white px-2 py-0.5 rounded font-semibold">AQ</span>
            )}
          </div>

          {/* 3 Seed */}
          <div className={`flex items-center gap-2 ${colorClasses[seed3Color].bg} border-l-4 ${colorClasses[seed3Color].border} px-3 py-2 rounded`}>
            <span className={`font-bold ${colorClasses[seed3Color].text} text-sm w-6`}>3</span>
            <span className="text-gray-900 flex-1">{regional.seed_3}</span>
            {seed3Color === 'green' && (
              <span className="text-xs bg-green-600 text-white px-2 py-0.5 rounded font-semibold">AQ</span>
            )}
          </div>

          {/* 4 Seed */}
          <div className={`flex items-center gap-2 ${colorClasses[seed4Color].bg} border-l-4 ${colorClasses[seed4Color].border} px-3 py-2 rounded`}>
            <span className={`font-bold ${colorClasses[seed4Color].text} text-sm w-6`}>4</span>
            <span className="text-gray-900 flex-1">{regional.seed_4}</span>
            {seed4Color === 'green' && (
              <span className="text-xs bg-green-600 text-white px-2 py-0.5 rounded font-semibold">AQ</span>
            )}
          </div>
        </div>
      </div>
    );
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
    <div className="min-h-screen bg-slate-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Simple Header with Export */}
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-3xl font-bold text-gray-900">Tournament Projection</h1>
          <button
            onClick={downloadCSV}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 font-semibold transition-colors"
          >
            <Download className="w-4 h-4" />
            Export CSV
          </button>
        </div>

        {error && (
          <div className="bg-red-50 border-l-4 border-red-500 text-red-800 px-4 py-3 rounded mb-6 flex items-center gap-2">
            <AlertCircle className="w-5 h-5" />
            {error}
          </div>
        )}

        {/* Conference Bids Summary */}
        {tournamentOutlook && (
          <div className="mb-8 bg-white rounded-lg shadow-md p-6 border-t-4 border-blue-600">
            <h2 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
              <Target className="w-5 h-5 text-blue-600" />
              Multi-Bid Conferences
            </h2>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3">
              {Object.entries(tournamentOutlook.multibid_conferences)
                .sort((a, b) => b[1] - a[1])
                .map(([conf, count]) => (
                  <div key={conf} className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg p-3 text-center border border-blue-200">
                    <div className="text-2xl font-bold text-blue-700">{count}</div>
                    <div className="text-xs font-semibold text-gray-700 uppercase tracking-wide">{conf}</div>
                  </div>
                ))}
            </div>
          </div>
        )}

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Regionals - Takes up 2 columns */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-lg shadow-md border-t-4 border-green-600">
              <div className="p-6 border-b border-gray-200">
                <h2 className="text-2xl font-bold text-gray-900">Projected Regional Hosts</h2>
                <p className="text-sm text-gray-600 mt-1">64-team field • 16 regional hosts • Top 16 national seeds</p>
                <div className="flex gap-4 mt-3 text-xs">
                  <div className="flex items-center gap-1">
                    <div className="w-3 h-3 bg-green-600 rounded"></div>
                    <span className="text-gray-600">Automatic Qualifier</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-3 h-3 bg-blue-600 rounded"></div>
                    <span className="text-gray-600">At-Large</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-3 h-3 bg-orange-600 rounded"></div>
                    <span className="text-gray-600">Last Four In</span>
                  </div>
                </div>
              </div>

              {tournamentOutlook && (
                <div className="p-6">
                  <div className="space-y-8">
                    {getRegionalPairs().map((pair, index) => (
                      <div key={index}>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                          <RegionalCard regional={pair.regional1} />
                          <RegionalCard regional={pair.regional2} />
                        </div>
                        {index < 7 && <div className="border-b border-gray-200 mt-8"></div>}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Sidebar - Bubble Watch & Simulator */}
          <div className="space-y-6">
            {/* Bubble Watch */}
            {tournamentOutlook && (
              <div className="bg-white rounded-lg shadow-md border-t-4 border-yellow-500">
                <div className="p-4 border-b border-gray-200">
                  <h3 className="text-xl font-bold text-gray-900 flex items-center gap-2">
                    <TrendingUp className="w-5 h-5 text-yellow-600" />
                    Bubble Watch
                  </h3>
                </div>

                <div className="p-4 space-y-4">
                  {/* Last Four In */}
                  <div>
                    <div className="flex items-center gap-2 mb-2">
                      <div className="w-3 h-3 bg-orange-500 rounded-full"></div>
                      <h4 className="font-bold text-sm text-gray-700 uppercase tracking-wide">Last Four In</h4>
                    </div>
                    <div className="space-y-1">
                      {tournamentOutlook.last_four_in.map((team, index) => (
                        <div key={index} className="bg-orange-50 border-l-2 border-orange-500 px-3 py-2 text-sm rounded">
                          {team}
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* First Four Out */}
                  <div>
                    <div className="flex items-center gap-2 mb-2">
                      <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                      <h4 className="font-bold text-sm text-gray-700 uppercase tracking-wide">First Four Out</h4>
                    </div>
                    <div className="space-y-1">
                      {tournamentOutlook.first_four_out.map((team, index) => (
                        <div key={index} className="bg-red-50 border-l-2 border-red-500 px-3 py-2 text-sm rounded">
                          {team}
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Next Four Out */}
                  <div>
                    <div className="flex items-center gap-2 mb-2">
                      <div className="w-3 h-3 bg-red-800 rounded-full"></div>
                      <h4 className="font-bold text-sm text-gray-700 uppercase tracking-wide">Next Four Out</h4>
                    </div>
                    <div className="space-y-1">
                      {tournamentOutlook.next_four_out.map((team, index) => (
                        <div key={index} className="bg-red-100 border-l-2 border-red-800 px-3 py-2 text-sm rounded">
                          {team}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Regional Simulator */}
            <div className="bg-white rounded-lg shadow-md border-t-4 border-purple-600">
              <div className="p-4 border-b border-gray-200">
                <h3 className="text-xl font-bold text-gray-900">Regional Simulator</h3>
                <p className="text-xs text-gray-600 mt-1">Simulate any 4-team regional</p>
              </div>

              <div className="p-4">
                <form onSubmit={simulateRegional} className="space-y-3">
                  {/* 1 Seed */}
                  <div>
                    <label className="block text-xs font-bold text-gray-700 mb-1 uppercase tracking-wide">
                      1 Seed (Host)
                    </label>
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
                        placeholder="Search team..."
                        className="w-full px-3 py-2 text-sm border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                      />
                      {showSeed1Dropdown && filteredSeed1Teams.length > 0 && (
                        <div className="absolute z-10 w-full mt-1 bg-white border-2 border-gray-300 rounded-lg shadow-lg max-h-48 overflow-y-auto">
                          {filteredSeed1Teams.map(team => (
                            <div
                              key={team}
                              onClick={() => {
                                setSeed1(team);
                                setSeed1SearchTerm('');
                                setShowSeed1Dropdown(false);
                              }}
                              className="px-3 py-2 hover:bg-purple-50 cursor-pointer text-sm"
                            >
                              {team}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>

                  {/* 2 Seed */}
                  <div>
                    <label className="block text-xs font-bold text-gray-700 mb-1 uppercase tracking-wide">2 Seed</label>
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
                        placeholder="Search team..."
                        className="w-full px-3 py-2 text-sm border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                      />
                      {showSeed2Dropdown && filteredSeed2Teams.length > 0 && (
                        <div className="absolute z-10 w-full mt-1 bg-white border-2 border-gray-300 rounded-lg shadow-lg max-h-48 overflow-y-auto">
                          {filteredSeed2Teams.map(team => (
                            <div
                              key={team}
                              onClick={() => {
                                setSeed2(team);
                                setSeed2SearchTerm('');
                                setShowSeed2Dropdown(false);
                              }}
                              className="px-3 py-2 hover:bg-purple-50 cursor-pointer text-sm"
                            >
                              {team}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>

                  {/* 3 Seed */}
                  <div>
                    <label className="block text-xs font-bold text-gray-700 mb-1 uppercase tracking-wide">3 Seed</label>
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
                        placeholder="Search team..."
                        className="w-full px-3 py-2 text-sm border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                      />
                      {showSeed3Dropdown && filteredSeed3Teams.length > 0 && (
                        <div className="absolute z-10 w-full mt-1 bg-white border-2 border-gray-300 rounded-lg shadow-lg max-h-48 overflow-y-auto">
                          {filteredSeed3Teams.map(team => (
                            <div
                              key={team}
                              onClick={() => {
                                setSeed3(team);
                                setSeed3SearchTerm('');
                                setShowSeed3Dropdown(false);
                              }}
                              className="px-3 py-2 hover:bg-purple-50 cursor-pointer text-sm"
                            >
                              {team}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>

                  {/* 4 Seed */}
                  <div>
                    <label className="block text-xs font-bold text-gray-700 mb-1 uppercase tracking-wide">4 Seed</label>
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
                        placeholder="Search team..."
                        className="w-full px-3 py-2 text-sm border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                      />
                      {showSeed4Dropdown && filteredSeed4Teams.length > 0 && (
                        <div className="absolute z-10 w-full mt-1 bg-white border-2 border-gray-300 rounded-lg shadow-lg max-h-48 overflow-y-auto">
                          {filteredSeed4Teams.map(team => (
                            <div
                              key={team}
                              onClick={() => {
                                setSeed4(team);
                                setSeed4SearchTerm('');
                                setShowSeed4Dropdown(false);
                              }}
                              className="px-3 py-2 hover:bg-purple-50 cursor-pointer text-sm"
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
                    className="w-full bg-purple-600 text-white py-2.5 px-4 rounded-lg hover:bg-purple-700 font-bold text-sm disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
                  >
                    {simulationLoading ? 'Simulating...' : 'Run Simulation'}
                  </button>
                </form>

                {/* Simulation Result */}
                {simulationImage && (
                  <div className="mt-4 pt-4 border-t border-gray-200">
                    <h4 className="text-sm font-bold text-gray-900 mb-2">Simulation Results</h4>
                    <div className="bg-gray-50 rounded-lg p-2">
                      <img 
                        src={simulationImage} 
                        alt="Regional Simulation" 
                        className="w-full h-auto rounded"
                      />
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Info Footer */}
        <div className="mt-8 bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg p-6">
          <h3 className="font-bold text-gray-900 mb-3 flex items-center gap-2">
            <AlertCircle className="w-5 h-5 text-blue-600" />
            How This Works
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-gray-700">
            <div>
              <p className="font-semibold text-gray-900 mb-1">Tournament Projection</p>
              <p>Projects the 16 regional hosts and their opponents based on current NET rankings. Automatic Qualifiers are determined via highest NET ranking. This does not account for location.</p>
            </div>
            <div>
              <p className="font-semibold text-gray-900 mb-1">Regional Simulator</p>
              <p>Uses PEAR ratings to simulate a double-elimination regional tournament 5,000 times. Home field advantage factored in for the host team.</p>
            </div>
            <div>
              <p className="font-semibold text-gray-900 mb-1">Bubble Teams</p>
              <p>Last 4 In are the final at-large teams projected to make the field. First/Next 4 Out are teams just outside the projected field.</p>
            </div>
            <div>
              <p className="font-semibold text-gray-900 mb-1">Conflict Resolution</p>
              <p>Algorithm minimizes conference matchups in the first round by intelligently swapping seeds across regionals while maintaining competitive balance.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}