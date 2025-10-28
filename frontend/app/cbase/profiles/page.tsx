"use client";

import { useState, useEffect, useRef } from "react";
import Image from "next/image";

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
      const response = await fetch("http://localhost:8000/api/cbase/teams");
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
      const response = await fetch("http://localhost:8000/api/cbase/team-profile", {
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
      const response = await fetch("http://localhost:8000/api/cbase/historical-performance", {
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
    <div className="min-h-screen bg-gray-50 py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Team Profiles
          </h1>
          <p className="text-lg text-gray-600">
            View team profiles and historical performance
          </p>
        </div>

        {/* Team Selection Form */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-8">
          <div className="space-y-4">
            <div className="relative" ref={dropdownRef}>
              <label
                htmlFor="team-select"
                className="block text-sm font-medium text-gray-700 mb-2"
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
                className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                autoComplete="off"
              />
              
              {/* Dropdown menu */}
              {showDropdown && filteredTeams.length > 0 && (
                <div className="absolute z-10 w-full mt-1 bg-white border border-gray-300 rounded-md shadow-lg max-h-60 overflow-auto">
                  {filteredTeams.map((team, index) => (
                    <div
                      key={team}
                      onClick={() => handleTeamSelect(team)}
                      className={`px-4 py-2 cursor-pointer ${
                        index === selectedIndex
                          ? "bg-blue-100"
                          : "hover:bg-gray-100"
                      }`}
                    >
                      {team}
                    </div>
                  ))}
                </div>
              )}

              {/* No results message */}
              {showDropdown && filteredTeams.length === 0 && teamName && (
                <div className="absolute z-10 w-full mt-1 bg-white border border-gray-300 rounded-md shadow-lg">
                  <div className="px-4 py-2 text-gray-500">No teams found</div>
                </div>
              )}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <button
                onClick={handleTeamProfile}
                disabled={loadingProfile || !teamName.trim()}
                className="w-full bg-blue-600 text-white py-3 px-6 rounded-md font-semibold hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
              >
                {loadingProfile ? "Generating..." : "Generate Team Profile"}
              </button>

              <button
                onClick={handleHistoricalPerformance}
                disabled={loadingHistorical || !teamName.trim()}
                className="w-full bg-green-600 text-white py-3 px-6 rounded-md font-semibold hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
              >
                {loadingHistorical ? "Generating..." : "Generate Historical Performance"}
              </button>
            </div>
          </div>

          {error && (
            <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-md">
              <p className="text-red-800 text-sm">{error}</p>
            </div>
          )}
        </div>

        {/* Results Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Team Profile Section */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-2xl font-bold text-gray-900">Team Profile</h2>
              {profileImage && (
                <button
                  onClick={() =>
                    handleDownload(profileImage, `${teamName}_profile.png`)
                  }
                  className="text-blue-600 hover:text-blue-800 font-medium text-sm"
                >
                  Download
                </button>
              )}
            </div>

            {loadingProfile ? (
              <div className="flex items-center justify-center h-96">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
              </div>
            ) : profileImage ? (
              <div className="relative w-full">
                <Image
                  src={profileImage}
                  alt="Team Profile"
                  width={800}
                  height={1000}
                  className="w-full h-auto rounded-lg"
                  unoptimized
                />
              </div>
            ) : (
              <div className="flex items-center justify-center h-96 bg-gray-100 rounded-lg">
                <p className="text-gray-500">
                  Select a team and click "Generate Team Profile"
                </p>
              </div>
            )}
          </div>

          {/* Historical Performance Section */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-2xl font-bold text-gray-900">
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
                  className="text-green-600 hover:text-green-800 font-medium text-sm"
                >
                  Download
                </button>
              )}
            </div>

            {loadingHistorical ? (
              <div className="flex items-center justify-center h-96">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-green-600"></div>
              </div>
            ) : historicalImage ? (
              <div className="relative w-full">
                <Image
                  src={historicalImage}
                  alt="Historical Performance"
                  width={1000}
                  height={700}
                  className="w-full h-auto rounded-lg"
                  unoptimized
                />
              </div>
            ) : (
              <div className="flex items-center justify-center h-96 bg-gray-100 rounded-lg">
                <p className="text-gray-500">
                  Select a team and click "Generate Historical Performance"
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Info Section */}
        <div className="mt-8 bg-blue-50 border border-blue-200 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-blue-900 mb-2">
            About These Visualizations
          </h3>
          <div className="text-sm text-blue-800 space-y-2">
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
      </div>
    </div>
  );
}