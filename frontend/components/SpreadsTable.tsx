'use client';

import { useState, useMemo } from 'react';
import { Download, X } from 'lucide-react';

interface SpreadData {
  start_date: string;
  start_time: string;
  home_team: string;
  away_team: string;
  home_score: number;
  away_score: number;
  PEAR_win_prob: number;
  PEAR: string;
  Vegas: string;
  difference: number;
  GQI: number;
  pr_spread: number;
}

interface Props {
  data: SpreadData[];
  year: number;
  week: number;
}

type SortMode = 'date' | 'gqi' | 'difference';

export default function SpreadsBoxes({ data, year, week }: Props) {
  const [selectedGame, setSelectedGame] = useState<SpreadData | null>(null);
  const [sortMode, setSortMode] = useState<SortMode>('date');
  
  console.log('Component rendered, selectedGame:', selectedGame);
  console.log('Year:', year, 'Week:', week);

  const getGQIColor = (gqi: number) => {
    if (gqi >= 9) return 'bg-green-700 dark:bg-green-800';
    if (gqi >= 8) return 'bg-green-500 dark:bg-green-500';
    if (gqi >= 7) return 'bg-green-400 dark:bg-green-400';
    if (gqi >= 6) return 'bg-yellow-500 dark:bg-yellow-600';
    if (gqi >= 4) return 'bg-orange-500 dark:bg-orange-600';
    return 'bg-red-600 dark:bg-red-600';
  };

  const formatDate = (dateString: string) => {
    let date: Date;
    
    if (!dateString || dateString === 'Invalid Date') {
      return 'Invalid Date';
    }
    
    if (dateString.includes('/')) {
      const [month, day, year] = dateString.split('/').map(Number);
      date = new Date(year, month - 1, day);
    } else if (dateString.includes('-')) {
      const parts = dateString.split('-');
      if (parts[0].length === 4) {
        const [year, month, day] = parts.map(Number);
        date = new Date(year, month - 1, day);
      } else {
        const [month, day, year] = parts.map(Number);
        date = new Date(year, month - 1, day);
      }
    } else {
      date = new Date(dateString);
    }
    
    if (isNaN(date.getTime())) {
      return 'Invalid Date';
    }
    
    return date.toLocaleDateString('en-US', { 
      weekday: 'long', 
      month: 'long', 
      day: 'numeric' 
    });
  };

  const parseTime = (timeString: string): number => {
    if (!timeString || timeString === 'TBD') return 9999; // TBD goes to end
    
    const match = timeString.match(/(\d+):(\d+)\s*(AM|PM)/i);
    if (!match) return 9999;
    
    let [_, hourStr, minuteStr, period] = match;
    let hours = parseInt(hourStr);
    const minutes = parseInt(minuteStr);
    
    if (period.toUpperCase() === 'PM' && hours !== 12) {
      hours += 12;
    } else if (period.toUpperCase() === 'AM' && hours === 12) {
      hours = 0;
    }
    
    return hours * 60 + minutes;
  };

  const getGamePreviewUrl = (homeTeam: string, awayTeam: string) => {
    const filename = `${homeTeam} vs ${awayTeam}`;
    const url = `http://localhost:8000/api/game-preview/${year}/${week}/${encodeURIComponent(filename)}`;
    console.log('Game preview URL:', url);
    return url;
  };

  const getLogoUrl = (teamName: string) => {
    const url = `http://localhost:8000/api/football-logo/${encodeURIComponent(teamName)}`;
    return url;
  };

  // Group and sort games
  const gamesByDate = useMemo(() => {
    if (sortMode === 'gqi') {
      // Sort all games by GQI descending
      const sortedGames = [...data].sort((a, b) => b.GQI - a.GQI);
      return [['Games Sorted by GQI', sortedGames]] as [string, SpreadData[]][];
    } else if (sortMode === 'difference') {
      // Sort all games by difference descending
      const sortedGames = [...data].sort((a, b) => b.difference - a.difference);
      return [['Games Sorted by Difference to Vegas Spread', sortedGames]] as [string, SpreadData[]][];
    }
    
    // Default: group by date
    const grouped = new Map<string, SpreadData[]>();
    
    data.forEach(game => {
      const dateKey = game.start_date;
      if (!grouped.has(dateKey)) {
        grouped.set(dateKey, []);
      }
      grouped.get(dateKey)!.push(game);
    });

    const sortedEntries = Array.from(grouped.entries()).sort((a, b) => {
      let dateA: Date, dateB: Date;
      
      if (a[0].includes('/')) {
        const [monthA, dayA, yearA] = a[0].split('/').map(Number);
        dateA = new Date(yearA, monthA - 1, dayA);
      } else {
        const [yearA, monthA, dayA] = a[0].split('-').map(Number);
        dateA = new Date(yearA, monthA - 1, dayA);
      }
      
      if (b[0].includes('/')) {
        const [monthB, dayB, yearB] = b[0].split('/').map(Number);
        dateB = new Date(yearB, monthB - 1, dayB);
      } else {
        const [yearB, monthB, dayB] = b[0].split('-').map(Number);
        dateB = new Date(yearB, monthB - 1, dayB);
      }
      
      return dateA.getTime() - dateB.getTime();
    });

    // Sort games within each date by time then GQI
    sortedEntries.forEach(([_, games]) => {
      games.sort((a, b) => {
        const timeA = parseTime(a.start_time);
        const timeB = parseTime(b.start_time);
        
        if (timeA !== timeB) return timeA - timeB;
        return b.GQI - a.GQI;
      });
    });

    return sortedEntries;
  }, [data, sortMode]);

  const downloadCSV = () => {
    const headers = ['Date', 'Time', 'Away Team', 'Away Score', 'Home Team', 'Home Score', 
                     'Away Win %', 'Home Win %', 'PEAR', 'Vegas', 'PEAR Raw', 'Difference', 'GQI'];
    const csvData = data.map(item => [
      item.start_date,
      item.start_time || 'TBD',
      item.away_team,
      item.away_score || '',
      item.home_team,
      item.home_score || '',
      `${(100 - item.PEAR_win_prob).toFixed(1)}%`,
      `${item.PEAR_win_prob.toFixed(1)}%`,
      item.PEAR,
      item.Vegas,
      item.pr_spread,
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

  const handleGameClick = (game: SpreadData) => {
    console.log('GQI bubble clicked, game:', game);
    setSelectedGame(game);
  };

  return (
    <div className="relative">
      {console.log('Render - selectedGame is:', selectedGame)}
      
      {/* Game Preview Modal */}
      {selectedGame ? (
        <div 
          className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-[9999] p-4"
          onClick={() => {
            console.log('Closing modal');
            setSelectedGame(null);
          }}
        >
          <div 
            className="relative max-w-6xl max-h-[90vh] bg-white dark:bg-gray-800 rounded-lg shadow-2xl overflow-hidden"
            onClick={(e) => {
              console.log('Clicked inside modal, stopping propagation');
              e.stopPropagation();
            }}
          >
            <button
              onClick={() => {
                console.log('Close button clicked');
                setSelectedGame(null);
              }}
              className="absolute top-4 right-4 p-2 bg-gray-900 bg-opacity-50 hover:bg-opacity-75 rounded-full text-white transition-all z-10"
            >
              <X className="w-6 h-6" />
            </button>
            <img
              src={getGamePreviewUrl(selectedGame.home_team, selectedGame.away_team)}
              alt={`${selectedGame.away_team} vs ${selectedGame.home_team} preview`}
              className="max-w-full max-h-[90vh] object-contain"
              onLoad={() => console.log('Image loaded successfully')}
              onError={(e) => {
                console.error('Failed to load game preview');
                e.currentTarget.src = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="400" height="300"><rect width="400" height="300" fill="%23f3f4f6"/><text x="50%" y="50%" text-anchor="middle" fill="%23374151" font-size="18">Preview not available</text></svg>';
              }}
            />
          </div>
        </div>
      ) : null}

      <div className="mb-4 flex justify-between items-center">
        <div className="flex gap-2">
          <button
            onClick={() => setSortMode('date')}
            className={`px-4 py-2 rounded-lg font-semibold transition-colors ${
              sortMode === 'date'
                ? 'bg-[#CECEB2] text-gray-900'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
            }`}
          >
            DATE
          </button>
          <button
            onClick={() => setSortMode('gqi')}
            className={`px-4 py-2 rounded-lg font-semibold transition-colors ${
              sortMode === 'gqi'
                ? 'bg-[#CECEB2] text-gray-900'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
            }`}
          >
            GQI
          </button>
          <button
            onClick={() => setSortMode('difference')}
            className={`px-4 py-2 rounded-lg font-semibold transition-colors ${
              sortMode === 'difference'
                ? 'bg-[#CECEB2] text-gray-900'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
            }`}
          >
            DIFF
          </button>
        </div>
        <button
          onClick={downloadCSV}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 dark:bg-blue-500 text-white rounded-lg hover:bg-blue-700 dark:hover:bg-blue-600 font-semibold transition-colors"
        >
          <Download className="w-4 h-4" />
          Export CSV
        </button>
      </div>

      <div className="space-y-6">
        {gamesByDate.map(([date, games]) => (
          <div key={date}>
            <h2 className="text-xl font-bold mb-3 text-gray-900 dark:text-white">
              {sortMode === 'date' ? formatDate(date) : date}
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
              {games.map((game, index) => {
                const homeWinning = game.PEAR_win_prob > 50;
                
                return (
                  <div 
                    key={`${date}-${index}`}
                    className="bg-white dark:bg-gray-800 rounded-lg p-3 shadow-md border-2 border-gray-200 dark:border-gray-700 hover:shadow-lg hover:border-blue-400 dark:hover:border-blue-500 transition-all cursor-pointer"
                    onClick={() => handleGameClick(game)}
                  >
                    {/* Header: Time and GQI */}
                    <div className="flex justify-between items-center mb-2 pb-2 border-b border-gray-200 dark:border-gray-700">
                      <span className="text-xs font-semibold text-gray-600 dark:text-gray-400">
                        {game.start_time || 'TBD'}
                      </span>
                      <span className={`px-2 py-0.5 rounded-full text-white text-xs font-bold ${getGQIColor(game.GQI)}`}>
                        {game.GQI.toFixed(1)}
                      </span>
                    </div>

                    {/* Away Team */}
                    <div className="flex justify-between items-center mb-2">
                      <div className="flex items-center gap-2 flex-1 min-w-0">
                        <img 
                          src={getLogoUrl(game.away_team)}
                          alt={`${game.away_team} logo`}
                          className="w-6 h-6 object-contain flex-shrink-0"
                          onError={(e) => {
                            e.currentTarget.style.display = 'none';
                          }}
                        />
                        <span className="text-sm font-semibold text-gray-900 dark:text-white truncate">
                          {game.away_team}
                        </span>
                      </div>
                      <div className="flex items-center gap-2 flex-shrink-0">
                        <span className="text-xs font-medium text-gray-600 dark:text-gray-400">
                          {(100 - game.PEAR_win_prob).toFixed(1)}%
                        </span>
                        {game.away_score !== null && game.away_score !== undefined && (
                          <span className={`text-base font-bold ${!homeWinning ? 'text-blue-600 dark:text-blue-400' : 'text-red-600 dark:text-red-400'}`}>
                            {game.away_score}
                          </span>
                        )}
                      </div>
                    </div>

                    {/* Home Team */}
                    <div className="flex justify-between items-center mb-2">
                      <div className="flex items-center gap-2 flex-1 min-w-0">
                        <img 
                          src={getLogoUrl(game.home_team)}
                          alt={`${game.home_team} logo`}
                          className="w-6 h-6 object-contain flex-shrink-0"
                          onError={(e) => {
                            e.currentTarget.style.display = 'none';
                          }}
                        />
                        <span className="text-sm font-semibold text-gray-900 dark:text-white truncate">
                          {game.home_team}
                        </span>
                      </div>
                      <div className="flex items-center gap-2 flex-shrink-0">
                        <span className="text-xs font-medium text-gray-600 dark:text-gray-400">
                          {game.PEAR_win_prob.toFixed(1)}%
                        </span>
                        {game.home_score !== null && game.home_score !== undefined && (
                          <span className={`text-base font-bold ${homeWinning ? 'text-blue-600 dark:text-blue-400' : 'text-red-600 dark:text-red-400'}`}>
                            {game.home_score}
                          </span>
                        )}
                      </div>
                    </div>

                    {/* Spreads */}
                    <div className="pt-2 border-t border-gray-200 dark:border-gray-700 space-y-1">
                      <div className="flex justify-between items-center">
                        <span className="text-xs text-gray-600 dark:text-gray-400">PROJECTION</span>
                        <span className="px-2 py-0.5 bg-blue-100 dark:bg-blue-900/50 text-blue-800 dark:text-blue-200 rounded font-medium text-xs">
                          {game.PEAR}
                        </span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-xs text-gray-600 dark:text-gray-400">VEGAS</span>
                        <span className="px-2 py-0.5 bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200 rounded font-medium text-xs">
                          {game.Vegas}
                        </span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>

      <div className="mt-6 text-xs text-gray-600 dark:text-gray-400 space-y-1">
        <p><strong className="dark:text-gray-300">GQI</strong> - Game Quality Index (1-10 scale, higher is better). Click any game to view preview!</p>
        <p><strong className="dark:text-gray-300">Win %</strong> - PEAR's projected win probability for each team</p>
        <p><strong className="dark:text-gray-300">PROJECTION</strong> - PEAR's predicted spread</p>
        <p><strong className="dark:text-gray-300">VEGAS</strong> - Vegas betting line</p>
        <p><strong className="dark:text-gray-300">Scores</strong> - Blue = projected winner, Red = projected loser</p>
      </div>
    </div>
  );
}