"use client";

import { useState, useEffect, useRef } from "react";
import Image from "next/image";

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function TeamAnalysisPage() {
  const [teamName, setTeamName] = useState("");
  const [profileImage, setProfileImage] = useState<string | null>(null);
  const [historicalImage, setHistoricalImage] = useState<string | null>(null);
  const [loadingProfile, setLoadingProfile] = useState(false);
  const [loadingHistorical, setLoadingHistorical] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [teams, setTeams] = useState<string[]>([]);
  const [teamsLoaded, setTeamsLoaded] = useState(false);
  const [filteredTeams, setFilteredTeams] = useState<string[]>([]);
  const [showDropdown, setShowDropdown] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(-1);
  const [fullscreenImage, setFullscreenImage] = useState<string | null>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Load teams when component mounts
  useEffect(() => {
    loadTeams();
  }, []);

  // Handle click outside dropdown
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setShowDropdown(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // Handle escape key for fullscreen modal
  useEffect(() => {
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape" && fullscreenImage) {
        setFullscreenImage(null);
      }
    };

    document.addEventListener("keydown", handleEscape);
    return () => document.removeEventListener("keydown", handleEscape);
  }, [fullscreenImage]);

  // Filter teams based on input
  useEffect(() => {
    if (teamName.trim() === "") {
      setFilteredTeams(teams);
    } else {
      const filtered = teams.filter((team) =>
        team.toLowerCase().includes(teamName.toLowerCase())
      );
      setFilteredTeams(filtered);
    }
    setSelectedIndex(-1);
  }, [teamName, teams]);

  const loadTeams = async () => {
    if (teamsLoaded) return;
    
    try {
      const response = await fetch(`${API_URL}/api/cbase/teams`);
      if (!response.ok) {
        throw new Error("Failed to load teams");
      }
      const data = await response.json();
      setTeams(data.teams);
      setFilteredTeams(data.teams);
      setTeamsLoaded(true);
    } catch (err) {
      console.error("Error loading teams:", err);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setTeamName(e.target.value);
    setShowDropdown(true);
  };

  const handleTeamSelect = (team: string) => {
    setTeamName(team);
    setShowDropdown(false);
    setSelectedIndex(-1);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (!showDropdown) {
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        setShowDropdown(true);
      }
      return;
    }

    switch (e.key) {
      case "ArrowDown":
        e.preventDefault();
        setSelectedIndex((prev) =>
          prev < filteredTeams.length - 1 ? prev + 1 : prev
        );
        break;
      case "ArrowUp":
        e.preventDefault();
        setSelectedIndex((prev) => (prev > 0 ? prev - 1 : -1));
        break;
      case "Enter":
        e.preventDefault();
        if (selectedIndex >= 0 && selectedIndex < filteredTeams.length) {
          handleTeamSelect(filteredTeams[selectedIndex]);
        } else if (filteredTeams.length === 1) {
          handleTeamSelect(filteredTeams[0]);
        }
        break;
      case "Escape":
        setShowDropdown(false);
        setSelectedIndex(-1);
        break;
    }
  };

  const handleTeamProfile = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!teamName.trim()) {
      setError("Please enter a team name");
      return;
    }

    setLoadingProfile(true);
    setError(null);

    try {
      const response = await fetch(`${API_URL}/api/cbase/team-profile`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          team_name: teamName,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Failed to generate team profile");
      }

      const blob = await response.blob();
      const imageUrl = URL.createObjectURL(blob);
      setProfileImage(imageUrl);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
      console.error("Error:", err);
    } finally {
      setLoadingProfile(false);
    }
  };

  const handleHistoricalPerformance = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!teamName.trim()) {
      setError("Please enter a team name");
      return;
    }

    setLoadingHistorical(true);
    setError(null);

    try {
      const response = await fetch(`${API_URL}/api/cbase/historical-performance`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          team_name: teamName,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Failed to generate historical performance");
      }

      const blob = await response.blob();
      const imageUrl = URL.createObjectURL(blob);
      setHistoricalImage(imageUrl);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
      console.error("Error:", err);
    } finally {
      setLoadingHistorical(false);
    }
  };

  const handleDownload = (imageUrl: string, filename: string) => {
    const link = document.createElement("a");
    link.href = imageUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-gray-900 dark:to-gray-800 pt-24 pb-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2">
            Team Profiles
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-400">
            View team profiles and historical performance
          </p>
        </div>

        {/* Team Selection Form */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 mb-8">
          <div className="space-y-4">
            <div className="relative" ref={dropdownRef}>
              <label
                htmlFor="team-select"
                className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2"
              >
                Select Team
              </label>
              <input
                ref={inputRef}
                id="team-select"
                type="text"
                value={teamName}
                onChange={handleInputChange}
                onFocus={() => setShowDropdown(true)}
                onKeyDown={handleKeyDown}
                placeholder="Type to search for a team..."
                className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400"
                autoComplete="off"
              />
              
              {/* Dropdown menu */}
              {showDropdown && filteredTeams.length > 0 && (
                <div className="absolute z-10 w-full mt-1 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-md shadow-lg max-h-60 overflow-auto">
                  {filteredTeams.map((team, index) => (
                    <div
                      key={team}
                      onClick={() => handleTeamSelect(team)}
                      className={`px-4 py-2 cursor-pointer text-gray-900 dark:text-white ${
                        index === selectedIndex
                          ? "bg-blue-100 dark:bg-blue-900"
                          : "hover:bg-gray-100 dark:hover:bg-gray-600"
                      }`}
                    >
                      {team}
                    </div>
                  ))}
                </div>
              )}

              {/* No results message */}
              {showDropdown && filteredTeams.length === 0 && teamName && (
                <div className="absolute z-10 w-full mt-1 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-md shadow-lg">
                  <div className="px-4 py-2 text-gray-500 dark:text-gray-400">No teams found</div>
                </div>
              )}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <button
                onClick={handleTeamProfile}
                disabled={loadingProfile || !teamName.trim()}
                className="w-full bg-blue-600 dark:bg-blue-500 text-white py-3 px-6 rounded-md font-semibold hover:bg-blue-700 dark:hover:bg-blue-600 disabled:bg-gray-400 dark:disabled:bg-gray-600 disabled:cursor-not-allowed transition-colors"
              >
                {loadingProfile ? "Generating..." : "Generate Team Profile"}
              </button>

              <button
                onClick={handleHistoricalPerformance}
                disabled={loadingHistorical || !teamName.trim()}
                className="w-full bg-green-600 dark:bg-green-500 text-white py-3 px-6 rounded-md font-semibold hover:bg-green-700 dark:hover:bg-green-600 disabled:bg-gray-400 dark:disabled:bg-gray-600 disabled:cursor-not-allowed transition-colors"
              >
                {loadingHistorical ? "Generating..." : "Generate Historical Performance"}
              </button>
            </div>
          </div>

          {error && (
            <div className="mt-4 p-4 bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-800 rounded-md">
              <p className="text-red-800 dark:text-red-300 text-sm">{error}</p>
            </div>
          )}
        </div>

        {/* Results Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Team Profile Section */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">Team Profile</h2>
              {profileImage && (
                <button
                  onClick={() =>
                    handleDownload(profileImage, `${teamName}_profile.png`)
                  }
                  className="text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 font-medium text-sm"
                >
                  Download
                </button>
              )}
            </div>

            {loadingProfile ? (
              <div className="flex items-center justify-center h-96">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 dark:border-blue-400"></div>
              </div>
            ) : profileImage ? (
              <div className="relative w-full cursor-pointer" onClick={() => setFullscreenImage(profileImage)}>
                <Image
                  src={profileImage}
                  alt="Team Profile"
                  width={800}
                  height={1000}
                  className="w-full h-auto rounded-lg hover:opacity-90 transition-opacity"
                  unoptimized
                />
              </div>
            ) : (
              <div className="flex items-center justify-center h-96 bg-gray-100 dark:bg-gray-700 rounded-lg">
                <p className="text-gray-500 dark:text-gray-400">
                  Select a team and click "Generate Team Profile"
                </p>
              </div>
            )}
          </div>

          {/* Historical Performance Section */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
                Historical Performance
              </h2>
              {historicalImage && (
                <button
                  onClick={() =>
                    handleDownload(
                      historicalImage,
                      `${teamName}_historical.png`
                    )
                  }
                  className="text-green-600 dark:text-green-400 hover:text-green-800 dark:hover:text-green-300 font-medium text-sm"
                >
                  Download
                </button>
              )}
            </div>

            {loadingHistorical ? (
              <div className="flex items-center justify-center h-96">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-green-600 dark:border-green-400"></div>
              </div>
            ) : historicalImage ? (
              <div className="relative w-full cursor-pointer" onClick={() => setFullscreenImage(historicalImage)}>
                <Image
                  src={historicalImage}
                  alt="Historical Performance"
                  width={1000}
                  height={700}
                  className="w-full h-auto rounded-lg hover:opacity-90 transition-opacity"
                  unoptimized
                />
              </div>
            ) : (
              <div className="flex items-center justify-center h-96 bg-gray-100 dark:bg-gray-700 rounded-lg">
                <p className="text-gray-500 dark:text-gray-400">
                  Select a team and click "Generate Historical Performance"
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Info Section */}
        <div className="mt-8 bg-blue-50 dark:bg-blue-900/30 border border-blue-200 dark:border-blue-800 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-blue-900 dark:text-blue-200 mb-2">
            About These Visualizations
          </h3>
          <div className="text-sm text-blue-800 dark:text-blue-300 space-y-2">
            <p>
              <strong>Team Profile:</strong> Shows comprehensive team statistics including NET rankings, 
              percentile rankings across various metrics, quad records, and recent performance.
            </p>
            <p>
              <strong>Historical Performance:</strong> Displays the team's performance trajectory 
              since 2008, plotting Team Strength against NET Score across multiple seasons. 
              National champions are highlighted in gold.
            </p>
          </div>
        </div>

        {/* Fullscreen Modal */}
        {fullscreenImage && (
          <div
            className="fixed inset-0 z-50 bg-black bg-opacity-90 flex items-center justify-center p-4"
            onClick={() => setFullscreenImage(null)}
          >
            <button
              onClick={() => setFullscreenImage(null)}
              className="absolute top-4 right-4 text-white hover:text-gray-300 text-4xl font-bold z-10"
              aria-label="Close fullscreen"
            >
              ×
            </button>
            <div className="relative max-w-full max-h-full">
              <Image
                src={fullscreenImage}
                alt="Fullscreen view"
                width={2000}
                height={2000}
                className="max-w-full max-h-[95vh] w-auto h-auto object-contain"
                unoptimized
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}