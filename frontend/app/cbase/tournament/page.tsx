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

interface ConferenceTournamentResult {
  conference: string;
  image: string;
}

export default function CbaseTournamentPage() {
  const [teams, setTeams] = useState<string[]>([]);
  const [conferences, setConferences] = useState<string[]>([]);
  const [tournamentOutlook, setTournamentOutlook] = useState<TournamentOutlook | null>(null);
  const [simulationResult, setSimulationResult] = useState<SimulationResult[] | null>(null);
  const [simulationImage, setSimulationImage] = useState<string | null>(null);
  const [conferenceTournamentImage, setConferenceTournamentImage] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [simulationLoading, setSimulationLoading] = useState(false);
  const [conferenceTournamentLoading, setConferenceTournamentLoading] = useState(false);
  const [error, setError] = useState('');

  const [seed1, setSeed1] = useState('');
  const [seed2, setSeed2] = useState('');
  const [seed3, setSeed3] = useState('');
  const [seed4, setSeed4] = useState('');

  const [selectedConference, setSelectedConference] = useState('');
  const [conferenceSearchTerm, setConferenceSearchTerm] = useState('');
  const [showConferenceDropdown, setShowConferenceDropdown] = useState(false);

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
    fetchConferences();
    fetchTeamConferences();
  }, []);

  const fetchTeams = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/cbase/teams`);
      setTeams(response.data.teams);
    } catch (error) {
      console.error('Error fetching teams:', error);
    }
  };

  const fetchConferences = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/cbase/conferences`);
      setConferences(response.data.conferences);
    } catch (error) {
      console.error('Error fetching conferences:', error);
    }
  };

  const fetchTeamConferences = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/cbase/team-conferences`);
      setTeamConferences(response.data.team_conferences);
    } catch (error) {
      console.error('Error fetching team conferences:', error);
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

  const simulateConferenceTournament = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!selectedConference) {
      setError('Please select a conference');
      return;
    }

    setConferenceTournamentLoading(true);
    setError('');
    setConferenceTournamentImage(null);

    try {
      const response = await axios.post(`${API_URL}/api/cbase/simulate-conference-tournament`, {
        conference: selectedConference
      }, {
        responseType: 'blob'
      });

      const imageObjectUrl = URL.createObjectURL(response.data);
      setConferenceTournamentImage(imageObjectUrl);
    } catch (error) {
      setError('Error simulating conference tournament. Please try again.');
      console.error('Error:', error);
    } finally {
      setConferenceTournamentLoading(false);
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

  const filteredConferences = conferences.filter(conference =>
    conference.toLowerCase().includes(conferenceSearchTerm.toLowerCase())
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

  const getTeamColor = (team: string) => {
    if (!tournamentOutlook) return 'blue';
    
    if (tournamentOutlook.automatic_qualifiers?.includes(team)) {
      return 'green';
    }
    
    if (tournamentOutlook.last_four_in.includes(team)) {
      return 'orange';
    }
    
    return 'blue';
  };

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

  const colorClasses = {
    green: {
      bg: 'bg-green-50 dark:bg-green-900/20',
      border: 'border-green-600 dark:border-green-500',
      text: 'text-green-700 dark:text-green-400',
      badge: 'bg-green-600 dark:bg-green-500'
    },
    blue: {
      bg: 'bg-blue-50 dark:bg-blue-900/20',
      border: 'border-blue-500 dark:border-blue-400',
      text: 'text-blue-600 dark:text-blue-400',
      badge: 'bg-blue-600 dark:bg-blue-500'
    },
    orange: {
      bg: 'bg-orange-100 dark:bg-orange-900/20',
      border: 'border-orange-600 dark:border-orange-500',
      text: 'text-orange-700 dark:text-orange-400',
      badge: 'bg-orange-700 dark:bg-orange-600'
    }
  };

  const TeamRow = ({ seed, team, color }: { seed: number; team: string; color: string }) => {
    const isHighlighted = highlightedConference && teamConferences[team] === highlightedConference;
    
    return (
      <div className={`flex items-center gap-2 ${colorClasses[color].bg} border-l-4 ${colorClasses[color].border} px-3 py-2 rounded transition-all ${
        isHighlighted ? 'ring-4 ring-yellow-400 ring-offset-2 scale-105 shadow-lg' : ''
      }`}>
        <span className={`font-bold ${colorClasses[color].text} text-sm w-6`}>{seed}</span>
        <Image 
          src={`${API_URL}/api/baseball-logo/${encodeURIComponent(team)}`}
          alt={`${team} logo`}
          width={24}
          height={24}
          className="rounded"
          unoptimized
        />
        <span className="font-bold text-gray-900 dark:text-white flex-1">{team}</span>
        {color === 'green' ? (
          <span className="text-xs bg-green-600 dark:bg-green-500 text-white px-2 py-0.5 rounded font-semibold">AQ</span>
        ) : color === 'orange' ? (
          <span className="text-xs bg-orange-700 dark:bg-orange-600 text-white px-2 py-0.5 rounded font-semibold">L4I</span>
        ) : seed === 1 ? (
          <span className="text-xs bg-blue-600 dark:bg-blue-500 text-white px-2 py-0.5 rounded font-semibold">HOST</span>
        ) : null}
      </div>
    );
  };

  const RegionalCard = ({ regional }: { regional: any }) => {
    const seed1Color = getTeamColor(regional.seed_1);
    const seed2Color = getTeamColor(regional.seed_2);
    const seed3Color = getTeamColor(regional.seed_3);
    const seed4Color = getTeamColor(regional.seed_4);

    const handleRegionalClick = async () => {
      setSimulationLoading(true);
      setError('');

      try {
        const response = await axios.post(`${API_URL}/api/cbase/simulate-regional`, {
          seed_1: regional.seed_1,
          seed_2: regional.seed_2,
          seed_3: regional.seed_3,
          seed_4: regional.seed_4
        }, {
          responseType: 'blob'
        });

        const imageObjectUrl = URL.createObjectURL(response.data);
        setFullscreenImage(imageObjectUrl);
      } catch (error) {
        setError('Error simulating regional. Please try again.');
        console.error('Error:', error);
      } finally {
        setSimulationLoading(false);
      }
    };

    return (
      <div 
        className="border-2 border-gray-200 dark:border-gray-700 rounded-lg hover:border-blue-400 dark:hover:border-blue-500 hover:shadow-md transition-all cursor-pointer"
        onClick={handleRegionalClick}
      >
        <div className="bg-gradient-to-r from-gray-800 to-gray-700 dark:from-gray-900 dark:to-gray-800 text-white px-4 py-2 rounded-t-lg">
          <div className="flex items-center justify-between">
            <span className="font-bold">{regional.host} Regional</span>
            <span className="text-xs bg-white/20 dark:bg-white/10 px-2 py-1 rounded">Seed #{regional.regional_number}</span>
          </div>
        </div>

        <div className="p-3 space-y-1 bg-white dark:bg-gray-800">
          <TeamRow seed={1} team={regional.seed_1} color={seed1Color} />
          <TeamRow seed={2} team={regional.seed_2} color={seed2Color} />
          <TeamRow seed={3} team={regional.seed_3} color={seed3Color} />
          <TeamRow seed={4} team={regional.seed_4} color={seed4Color} />
        </div>
      </div>
    );
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-900 via-blue-800 to-blue-900 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-yellow-400 dark:border-blue-400 mx-auto"></div>
          <p className="text-white mt-4 text-xl">Loading Tournament Data...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-gray-900 dark:to-gray-800 pt-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">PEAR's Tournament Projection</h1>
          <button
            onClick={downloadCSV}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 dark:bg-blue-500 text-white rounded-lg hover:bg-blue-700 dark:hover:bg-blue-600 font-semibold transition-colors"
          >
            <Download className="w-4 h-4" />
            Export CSV
          </button>
        </div>

        {error && (
          <div className="bg-red-50 dark:bg-red-900/20 border-l-4 border-red-500 dark:border-red-600 text-red-800 dark:text-red-300 px-4 py-3 rounded mb-6 flex items-center gap-2">
            <AlertCircle className="w-5 h-5" />
            {error}
          </div>
        )}

        {tournamentOutlook && (
          <div className="mb-8 bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 border-t-4 border-blue-600 dark:border-blue-500">
            <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
              <Target className="w-5 h-5 text-blue-600 dark:text-blue-400" />
              Multi-Bid Conferences
              {highlightedConference && (
                <span className="ml-auto text-sm font-normal text-gray-600 dark:text-gray-400">
                  Showing: {highlightedConference}
                  <button
                    onClick={() => setHighlightedConference(null)}
                    className="ml-2 text-xs bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 px-2 py-1 rounded hover:bg-gray-300 dark:hover:bg-gray-600"
                  >
                    Clear
                  </button>
                </span>
              )}
            </h2>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3">
              {Object.entries(tournamentOutlook.multibid_conferences)
                .sort((a, b) => b[1] - a[1])
                .map(([conf, count]) => (
                  <button
                    key={conf}
                    onClick={() => setHighlightedConference(highlightedConference === conf ? null : conf)}
                    className={`rounded-lg p-3 text-center border transition-all transform hover:scale-105 ${
                      highlightedConference === conf
                        ? 'bg-yellow-200 dark:bg-yellow-700 border-yellow-500 dark:border-yellow-400 ring-2 ring-yellow-400 shadow-lg'
                        : 'bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/30 dark:to-blue-800/30 border-blue-200 dark:border-blue-700 hover:border-blue-400 dark:hover:border-blue-500'
                    }`}
                  >
                    <div className={`text-2xl font-bold ${
                      highlightedConference === conf ? 'text-yellow-900 dark:text-yellow-100' : 'text-blue-700 dark:text-blue-300'
                    }`}>{count}</div>
                    <div className={`text-xs font-semibold uppercase tracking-wide ${
                      highlightedConference === conf ? 'text-yellow-800 dark:text-yellow-200' : 'text-gray-700 dark:text-gray-300'
                    }`}>{conf}</div>
                  </button>
                ))}
            </div>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-3 text-center">
              Click on a conference to highlight its teams in the bracket
            </p>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md border-t-4 border-green-600 dark:border-green-500">
              <div className="p-6 border-b border-gray-200 dark:border-gray-700">
                <h2 className="text-2xl font-bold text-gray-900 dark:text-white">PEAR's Projected Regionals</h2>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">64-team field • Top 16 national seeds</p>
                <div className="flex gap-4 mt-3 text-xs">
                  <div className="flex items-center gap-1">
                    <div className="w-3 h-3 bg-green-600 dark:bg-green-500 rounded"></div>
                    <span className="text-gray-600 dark:text-gray-400">Automatic Qualifier</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-3 h-3 bg-blue-600 dark:bg-blue-500 rounded"></div>
                    <span className="text-gray-600 dark:text-gray-400">At-Large</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-3 h-3 bg-orange-600 rounded"></div>
                    <span className="text-gray-600 dark:text-gray-400">Last Four In</span>
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
                        {index < 7 && <div className="border-b border-gray-200 dark:border-gray-700 mt-8"></div>}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="space-y-6">
            {tournamentOutlook && (
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md border-t-4 border-yellow-500 dark:border-yellow-600">
                <div className="p-4 border-b border-gray-200 dark:border-gray-700">
                  <h3 className="text-xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
                    <TrendingUp className="w-5 h-5 text-yellow-600 dark:text-yellow-500" />
                    Bubble Watch
                  </h3>
                </div>

                <div className="p-4 space-y-4">
                  <div>
                    <div className="flex items-center gap-2 mb-2">
                      <div className="w-3 h-3 bg-orange-500 rounded-full"></div>
                      <h4 className="font-bold text-sm text-gray-700 dark:text-gray-300 uppercase tracking-wide">Last Four In</h4>
                    </div>
                    <div className="space-y-1">
                      {tournamentOutlook.last_four_in.map((team, index) => {
                        const isHighlighted = highlightedConference && teamConferences[team] === highlightedConference;
                        return (
                          <div 
                            key={index} 
                            className={`bg-orange-50 dark:bg-orange-900/20 border-l-2 border-orange-500 dark:border-orange-600 px-3 py-2 text-sm rounded text-gray-900 dark:text-gray-100 transition-all ${
                              isHighlighted ? 'ring-4 ring-yellow-400 ring-offset-2 scale-105 shadow-lg' : ''
                            }`}
                          >
                            {team}
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  <div>
                    <div className="flex items-center gap-2 mb-2">
                      <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                      <h4 className="font-bold text-sm text-gray-700 dark:text-gray-300 uppercase tracking-wide">First Four Out</h4>
                    </div>
                    <div className="space-y-1">
                      {tournamentOutlook.first_four_out.map((team, index) => {
                        const isHighlighted = highlightedConference && teamConferences[team] === highlightedConference;
                        return (
                          <div 
                            key={index} 
                            className={`bg-red-50 dark:bg-red-900/20 border-l-2 border-red-500 dark:border-red-600 px-3 py-2 text-sm rounded text-gray-900 dark:text-gray-100 transition-all ${
                              isHighlighted ? 'ring-4 ring-yellow-400 ring-offset-2 scale-105 shadow-lg' : ''
                            }`}
                          >
                            {team}
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  <div>
                    <div className="flex items-center gap-2 mb-2">
                      <div className="w-3 h-3 bg-red-800 dark:bg-red-900 rounded-full"></div>
                      <h4 className="font-bold text-sm text-gray-700 dark:text-gray-300 uppercase tracking-wide">Next Four Out</h4>
                    </div>
                    <div className="space-y-1">
                      {tournamentOutlook.next_four_out.map((team, index) => {
                        const isHighlighted = highlightedConference && teamConferences[team] === highlightedConference;
                        return (
                          <div 
                            key={index} 
                            className={`bg-red-100 dark:bg-red-950/30 border-l-2 border-red-800 dark:border-red-900 px-3 py-2 text-sm rounded text-gray-900 dark:text-gray-100 transition-all ${
                              isHighlighted ? 'ring-4 ring-yellow-400 ring-offset-2 scale-105 shadow-lg' : ''
                            }`}
                          >
                            {team}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              </div>
            )}

            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md border-t-4 border-purple-600 dark:border-purple-500">
              <div className="p-4 border-b border-gray-200 dark:border-gray-700">
                <h3 className="text-xl font-bold text-gray-900 dark:text-white">Regional Simulator</h3>
                <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">Simulate any 4-team regional</p>
              </div>

              <div className="p-4">
                <form onSubmit={simulateRegional} className="space-y-3">
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
                    <label className="block text-xs font-bold text-gray-700 dark:text-gray-300 mb-1 uppercase tracking-wide">3 Seed</label>
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
                    <label className="block text-xs font-bold text-gray-700 dark:text-gray-300 mb-1 uppercase tracking-wide">4 Seed</label>
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

            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md border-t-4 border-indigo-600 dark:border-indigo-500">
              <div className="p-4 border-b border-gray-200 dark:border-gray-700">
                <h3 className="text-xl font-bold text-gray-900 dark:text-white">Conference Tournament Simulator</h3>
                <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">Simulate any conference tournament</p>
              </div>

              <div className="p-4">
                <form onSubmit={simulateConferenceTournament} className="space-y-3">
                  <div>
                    <label className="block text-xs font-bold text-gray-700 dark:text-gray-300 mb-1 uppercase tracking-wide">
                      Conference
                    </label>
                    <div className="relative">
                      <input
                        type="text"
                        value={selectedConference || conferenceSearchTerm}
                        onChange={(e) => {
                          setConferenceSearchTerm(e.target.value);
                          setSelectedConference('');
                          setShowConferenceDropdown(true);
                        }}
                        onFocus={() => setShowConferenceDropdown(true)}
                        onBlur={() => setTimeout(() => setShowConferenceDropdown(false), 200)}
                        placeholder="Search conference..."
                        className="w-full px-3 py-2 text-sm border-2 border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400"
                      />
                      {showConferenceDropdown && filteredConferences.length > 0 && (
                        <div className="absolute z-10 w-full mt-1 bg-white dark:bg-gray-700 border-2 border-gray-300 dark:border-gray-600 rounded-lg shadow-lg max-h-48 overflow-y-auto">
                          {filteredConferences.map(conference => (
                            <div
                              key={conference}
                              onClick={() => {
                                setSelectedConference(conference);
                                setConferenceSearchTerm('');
                                setShowConferenceDropdown(false);
                              }}
                              className="px-3 py-2 hover:bg-indigo-50 dark:hover:bg-indigo-900/30 cursor-pointer text-sm text-gray-900 dark:text-gray-100"
                            >
                              {conference}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>

                  <button
                    type="submit"
                    disabled={conferenceTournamentLoading}
                    className="w-full bg-indigo-600 dark:bg-indigo-500 text-white py-2.5 px-4 rounded-lg hover:bg-indigo-700 dark:hover:bg-indigo-600 font-bold text-sm disabled:bg-gray-400 dark:disabled:bg-gray-600 disabled:cursor-not-allowed transition-colors"
                  >
                    {conferenceTournamentLoading ? 'Simulating...' : 'Run Simulation'}
                  </button>
                </form>

                {conferenceTournamentImage && (
                  <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
                    <h4 className="text-sm font-bold text-gray-900 dark:text-white mb-2">Tournament Results</h4>
                    <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-2 cursor-pointer hover:opacity-90 transition-opacity" onClick={() => setFullscreenImage(conferenceTournamentImage)}>
                      <img 
                        src={conferenceTournamentImage} 
                        alt="Conference Tournament Simulation" 
                        className="w-full h-auto rounded"
                      />
                    </div>
                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-1 text-center">Click to view fullscreen</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        <div className="mt-8 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-6">
          <h3 className="font-bold text-gray-900 dark:text-white mb-3 flex items-center gap-2">
            <AlertCircle className="w-5 h-5 text-blue-600 dark:text-blue-400" />
            How This Works
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-gray-700 dark:text-gray-300">
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
            <div>
              <p className="font-semibold text-gray-900 dark:text-white mb-1">Conference Tournament Simulator</p>
              <p>Simulates conference tournaments using PEAR ratings with 1,000 iterations. Each conference has its own unique tournament format and bracket structure.</p>
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