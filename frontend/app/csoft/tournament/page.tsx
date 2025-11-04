'use client';

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Download, TrendingUp, AlertCircle, Target } from 'lucide-react';
import Image from 'next/image';

const API_URL = process.env.NEXT_PUBLIC_API_URL || '';

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

export default function CsoftTournamentPage() {
  const [teams, setTeams] = useState<string[]>([]);
  const [tournamentOutlook, setTournamentOutlook] = useState<TournamentOutlook | null>(null);
  const [simulationResult, setSimulationResult] = useState<SimulationResult[] | null>(null);
  const [simulationImage, setSimulationImage] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [simulationLoading, setSimulationLoading] = useState(false);
  const [error, setError] = useState('');

  const [seed1, setSeed1] = useState('');
  const [seed2, setSeed2] = useState('');
  const [seed3, setSeed3] = useState('');
  const [seed4, setSeed4] = useState('');

  const [seed1SearchTerm, setSeed1SearchTerm] = useState('');
  const [seed2SearchTerm, setSeed2SearchTerm] = useState('');
  const [seed3SearchTerm, setSeed3SearchTerm] = useState('');
  const [seed4SearchTerm, setSeed4SearchTerm] = useState('');
  const [showSeed1Dropdown, setShowSeed1Dropdown] = useState(false);
  const [showSeed2Dropdown, setShowSeed2Dropdown] = useState(false);
  const [showSeed3Dropdown, setShowSeed3Dropdown] = useState(false);
  const [showSeed4Dropdown, setShowSeed4Dropdown] = useState(false);

  const [fullscreenImage, setFullscreenImage] = useState<string | null>(null);
  
  const [teamConferences, setTeamConferences] = useState<{ [key: string]: string }>({});
  const [highlightedConference, setHighlightedConference] = useState<string | null>(null);

  useEffect(() => {
    fetchTournamentData();
    fetchTeams();
    fetchTeamConferences();
  }, []);

  const fetchTeams = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/softball/teams`);
      setTeams(response.data.teams);
    } catch (error) {
      console.error('Error fetching teams:', error);
    }
  };

  const fetchTeamConferences = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/softball/team-conferences`);
      setTeamConferences(response.data.team_conferences);
    } catch (error) {
      console.error('Error fetching team conferences:', error);
    }
  };

  const fetchTournamentData = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API_URL}/api/softball/tournament-outlook`);
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
      const response = await axios.post(`${API_URL}/api/softball/simulate-regional`, {
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
    const link = document.createElement('a');
    link.href = url;
    link.download = 'tournament-outlook.csv';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-24 w-24 border-t-2 border-b-2 border-purple-600 dark:border-purple-400 mx-auto"></div>
          <p className="mt-6 text-gray-900 dark:text-white text-xl font-bold">Loading Tournament Data...</p>
        </div>
      </div>
    );
  }

  if (error && !tournamentOutlook) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 flex items-center justify-center p-4">
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-8 max-w-md w-full">
          <div className="text-red-600 dark:text-red-400 text-center">
            <AlertCircle className="w-16 h-16 mx-auto mb-4" />
            <p className="text-xl font-bold">{error}</p>
            <button
              onClick={fetchTournamentData}
              className="mt-6 bg-purple-600 dark:bg-purple-500 text-white px-6 py-2 rounded-lg hover:bg-purple-700 dark:hover:bg-purple-600 transition-colors font-bold"
            >
              Try Again
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        <div className="text-center mb-8">
          <h1 className="text-4xl md:text-5xl font-bold text-gray-900 dark:text-white mb-2">
            NCAA Softball Tournament
          </h1>
          <p className="text-gray-600 dark:text-gray-400 text-lg">
            2025 Tournament Outlook & Regional Simulator
          </p>
        </div>

        {error && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-400 px-4 py-3 rounded-lg mb-6">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-5 h-5" />
              <span className="font-medium">{error}</span>
            </div>
          </div>
        )}

        {tournamentOutlook && (
          <>
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg mb-8 overflow-hidden">
              <div className="bg-gradient-to-r from-purple-600 to-indigo-600 dark:from-purple-500 dark:to-indigo-500 p-6">
                <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                  <div>
                    <h2 className="text-2xl font-bold text-white mb-2">Current Tournament Projection</h2>
                    <p className="text-purple-100 dark:text-purple-200">Based on latest NET rankings</p>
                  </div>
                  <button
                    onClick={downloadCSV}
                    className="bg-white dark:bg-gray-800 text-purple-600 dark:text-purple-400 px-6 py-2.5 rounded-lg hover:bg-purple-50 dark:hover:bg-gray-700 transition-colors font-bold text-sm flex items-center gap-2 shadow-md"
                  >
                    <Download className="w-4 h-4" />
                    Export to CSV
                  </button>
                </div>
              </div>

              <div className="p-6">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                  {tournamentOutlook.regionals.map((regional) => {
                    const hostConference = teamConferences[regional.host];
                    const isHighlighted = highlightedConference && hostConference === highlightedConference;
                    
                    return (
                      <div 
                        key={regional.regional_number}
                        className={`bg-gradient-to-br ${
                          isHighlighted 
                            ? 'from-purple-50 to-indigo-50 dark:from-purple-900/30 dark:to-indigo-900/30 ring-2 ring-purple-400 dark:ring-purple-500' 
                            : 'from-gray-50 to-gray-100 dark:from-gray-700 dark:to-gray-800'
                        } p-4 rounded-lg border border-gray-200 dark:border-gray-600 hover:shadow-md transition-all`}
                        onMouseEnter={() => setHighlightedConference(hostConference)}
                        onMouseLeave={() => setHighlightedConference(null)}
                      >
                        <div className="flex items-center justify-between mb-3">
                          <span className="text-xs font-bold text-purple-600 dark:text-purple-400 uppercase tracking-wider">
                            Regional {regional.regional_number}
                          </span>
                          <Target className="w-4 h-4 text-purple-600 dark:text-purple-400" />
                        </div>
                        <div className="space-y-2">
                          <div className="flex items-start gap-2">
                            <span className="text-xs font-bold text-gray-500 dark:text-gray-400 mt-0.5 flex-shrink-0">1.</span>
                            <div>
                              <span className="text-sm font-bold text-gray-900 dark:text-white block leading-tight">
                                {regional.host}
                              </span>
                              {hostConference && (
                                <span className="text-xs text-purple-600 dark:text-purple-400 font-medium">
                                  {hostConference}
                                </span>
                              )}
                            </div>
                          </div>
                          <div className="flex items-center gap-2">
                            <span className="text-xs font-bold text-gray-500 dark:text-gray-400">2.</span>
                            <span className="text-sm text-gray-700 dark:text-gray-300">{regional.seed_2}</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <span className="text-xs font-bold text-gray-500 dark:text-gray-400">3.</span>
                            <span className="text-sm text-gray-700 dark:text-gray-300">{regional.seed_3}</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <span className="text-xs font-bold text-gray-500 dark:text-gray-400">4.</span>
                            <span className="text-sm text-gray-700 dark:text-gray-300">{regional.seed_4}</span>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                  <div className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 p-5 rounded-lg border border-green-200 dark:border-green-800">
                    <h3 className="text-sm font-bold text-green-900 dark:text-green-400 mb-3 flex items-center gap-2 uppercase tracking-wide">
                      <TrendingUp className="w-4 h-4" />
                      Last Four In
                    </h3>
                    <div className="space-y-2">
                      {tournamentOutlook.last_four_in.map((team, idx) => (
                        <div key={idx} className="flex items-center gap-2 text-sm">
                          <span className="w-6 h-6 bg-green-600 dark:bg-green-500 text-white rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0">
                            {idx + 1}
                          </span>
                          <span className="text-gray-900 dark:text-gray-100 font-medium">{team}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="bg-gradient-to-br from-amber-50 to-orange-50 dark:from-amber-900/20 dark:to-orange-900/20 p-5 rounded-lg border border-amber-200 dark:border-amber-800">
                    <h3 className="text-sm font-bold text-amber-900 dark:text-amber-400 mb-3 flex items-center gap-2 uppercase tracking-wide">
                      <AlertCircle className="w-4 h-4" />
                      First Four Out
                    </h3>
                    <div className="space-y-2">
                      {tournamentOutlook.first_four_out.map((team, idx) => (
                        <div key={idx} className="flex items-center gap-2 text-sm">
                          <span className="w-6 h-6 bg-amber-600 dark:bg-amber-500 text-white rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0">
                            {idx + 1}
                          </span>
                          <span className="text-gray-900 dark:text-gray-100 font-medium">{team}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="bg-gradient-to-br from-red-50 to-rose-50 dark:from-red-900/20 dark:to-rose-900/20 p-5 rounded-lg border border-red-200 dark:border-red-800">
                  <h3 className="text-sm font-bold text-red-900 dark:text-red-400 mb-3 uppercase tracking-wide">
                    Next Four Out
                  </h3>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                    {tournamentOutlook.next_four_out.map((team, idx) => (
                      <div key={idx} className="flex items-center gap-2 text-sm">
                        <span className="w-6 h-6 bg-red-600 dark:bg-red-500 text-white rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0">
                          {idx + 1}
                        </span>
                        <span className="text-gray-900 dark:text-gray-100 font-medium">{team}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {Object.keys(tournamentOutlook.multibid_conferences).length > 0 && (
                  <div className="mt-6 bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 p-5 rounded-lg border border-blue-200 dark:border-blue-800">
                    <h3 className="text-sm font-bold text-blue-900 dark:text-blue-400 mb-3 uppercase tracking-wide">
                      Multi-Bid Conferences
                    </h3>
                    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
                      {Object.entries(tournamentOutlook.multibid_conferences)
                        .sort(([, a], [, b]) => b - a)
                        .map(([conference, count]) => (
                          <div key={conference} className="flex items-center justify-between bg-white dark:bg-gray-800 px-3 py-2 rounded-md border border-blue-200 dark:border-blue-700">
                            <span className="text-sm font-medium text-gray-900 dark:text-gray-100">{conference}</span>
                            <span className="text-sm font-bold text-blue-600 dark:text-blue-400">{count}</span>
                          </div>
                        ))}
                    </div>
                  </div>
                )}
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md border-t-4 border-purple-600 dark:border-purple-500">
              <div className="p-4 border-b border-gray-200 dark:border-gray-700">
                <h3 className="text-xl font-bold text-gray-900 dark:text-white">Regional Simulator</h3>
                <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">Select four teams to simulate a regional tournament</p>
              </div>

              <div className="p-4">
                <form onSubmit={simulateRegional} className="space-y-3">
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    <div>
                      <label className="block text-xs font-bold text-gray-700 dark:text-gray-300 mb-1 uppercase tracking-wide">
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
                          className="w-full px-3 py-2 text-sm border-2 border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400"
                        />
                        {showSeed1Dropdown && filteredSeed1Teams.length > 0 && (
                          <div className="absolute z-10 w-full mt-1 bg-white dark:bg-gray-700 border-2 border-gray-300 dark:border-gray-600 rounded-lg shadow-lg max-h-48 overflow-y-auto">
                            {filteredSeed1Teams.map(team => (
                              <div
                                key={team}
                                onClick={() => {
                                  setSeed1(team);
                                  setSeed1SearchTerm('');
                                  setShowSeed1Dropdown(false);
                                }}
                                className="px-3 py-2 hover:bg-purple-50 dark:hover:bg-purple-900/30 cursor-pointer text-sm text-gray-900 dark:text-gray-100"
                              >
                                {team}
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>

                    <div>
                      <label className="block text-xs font-bold text-gray-700 dark:text-gray-300 mb-1 uppercase tracking-wide">
                        2 Seed
                      </label>
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
                          className="w-full px-3 py-2 text-sm border-2 border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400"
                        />
                        {showSeed2Dropdown && filteredSeed2Teams.length > 0 && (
                          <div className="absolute z-10 w-full mt-1 bg-white dark:bg-gray-700 border-2 border-gray-300 dark:border-gray-600 rounded-lg shadow-lg max-h-48 overflow-y-auto">
                            {filteredSeed2Teams.map(team => (
                              <div
                                key={team}
                                onClick={() => {
                                  setSeed2(team);
                                  setSeed2SearchTerm('');
                                  setShowSeed2Dropdown(false);
                                }}
                                className="px-3 py-2 hover:bg-purple-50 dark:hover:bg-purple-900/30 cursor-pointer text-sm text-gray-900 dark:text-gray-100"
                              >
                                {team}
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>

                    <div>
                      <label className="block text-xs font-bold text-gray-700 dark:text-gray-300 mb-1 uppercase tracking-wide">
                        3 Seed
                      </label>
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
                          className="w-full px-3 py-2 text-sm border-2 border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400"
                        />
                        {showSeed3Dropdown && filteredSeed3Teams.length > 0 && (
                          <div className="absolute z-10 w-full mt-1 bg-white dark:bg-gray-700 border-2 border-gray-300 dark:border-gray-600 rounded-lg shadow-lg max-h-48 overflow-y-auto">
                            {filteredSeed3Teams.map(team => (
                              <div
                                key={team}
                                onClick={() => {
                                  setSeed3(team);
                                  setSeed3SearchTerm('');
                                  setShowSeed3Dropdown(false);
                                }}
                                className="px-3 py-2 hover:bg-purple-50 dark:hover:bg-purple-900/30 cursor-pointer text-sm text-gray-900 dark:text-gray-100"
                              >
                                {team}
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>

                    <div>
                      <label className="block text-xs font-bold text-gray-700 dark:text-gray-300 mb-1 uppercase tracking-wide">
                        4 Seed
                      </label>
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
                          className="w-full px-3 py-2 text-sm border-2 border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400"
                        />
                        {showSeed4Dropdown && filteredSeed4Teams.length > 0 && (
                          <div className="absolute z-10 w-full mt-1 bg-white dark:bg-gray-700 border-2 border-gray-300 dark:border-gray-600 rounded-lg shadow-lg max-h-48 overflow-y-auto">
                            {filteredSeed4Teams.map(team => (
                              <div
                                key={team}
                                onClick={() => {
                                  setSeed4(team);
                                  setSeed4SearchTerm('');
                                  setShowSeed4Dropdown(false);
                                }}
                                className="px-3 py-2 hover:bg-purple-50 dark:hover:bg-purple-900/30 cursor-pointer text-sm text-gray-900 dark:text-gray-100"
                              >
                                {team}
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>

                  <button
                    type="submit"
                    disabled={simulationLoading}
                    className="w-full bg-purple-600 dark:bg-purple-500 text-white py-2.5 px-4 rounded-lg hover:bg-purple-700 dark:hover:bg-purple-600 font-bold text-sm disabled:bg-gray-400 dark:disabled:bg-gray-600 disabled:cursor-not-allowed transition-colors"
                  >
                    {simulationLoading ? 'Simulating...' : 'Run Simulation'}
                  </button>
                </form>

                {simulationImage && (
                  <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
                    <h4 className="text-sm font-bold text-gray-900 dark:text-white mb-2">Simulation Results</h4>
                    <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-2 cursor-pointer hover:opacity-90 transition-opacity" onClick={() => setFullscreenImage(simulationImage)}>
                      <img 
                        src={simulationImage} 
                        alt="Regional Simulation" 
                        className="w-full h-auto rounded"
                      />
                    </div>
                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-1 text-center">Click to view fullscreen</p>
                  </div>
                )}
              </div>
            </div>
          </>
        )}

        <div className="mt-8 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-6">
          <h3 className="font-bold text-gray-900 dark:text-white mb-3 flex items-center gap-2">
            <AlertCircle className="w-5 h-5 text-blue-600 dark:text-blue-400" />
            How This Works
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm text-gray-700 dark:text-gray-300">
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1">Tournament Projection</p>
              <p>Projects the 16 regional hosts and their opponents based on current NET rankings. Automatic Qualifiers are determined via highest NET ranking. This does not account for location.</p>
            </div>
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1">Regional Simulator</p>
              <p>Uses PEAR ratings to simulate a double-elimination regional tournament 5,000 times. Home field advantage factored in for the host team.</p>
            </div>
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1">Bubble Teams</p>
              <p>Last 4 In are the final at-large teams projected to make the field. First/Next 4 Out are teams just outside the projected field.</p>
            </div>
          </div>
        </div>
      </div>

      {fullscreenImage && (
        <div 
          className="fixed inset-0 z-50 bg-black bg-opacity-95 flex items-center justify-center p-4"
          onClick={() => setFullscreenImage(null)}
        >
          <div className="relative max-w-7xl max-h-full">
            {simulationLoading && (
              <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50 rounded-lg z-20">
                <div className="text-center">
                  <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-white mx-auto"></div>
                  <p className="text-white mt-4 text-lg">Simulating Regional...</p>
                </div>
              </div>
            )}
            <button
              onClick={() => setFullscreenImage(null)}
              className="absolute top-4 right-4 bg-white dark:bg-gray-800 text-gray-900 dark:text-white rounded-full p-2 hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors z-10"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
            <img 
              src={fullscreenImage} 
              alt="Fullscreen view" 
              className="max-w-full max-h-[90vh] object-contain rounded-lg"
              onClick={(e) => e.stopPropagation()}
            />
          </div>
        </div>
      )}
    </div>
  );
}