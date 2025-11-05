'use client';

import { useState, useEffect } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import { ArrowLeft, Trophy, TrendingUp, Calendar } from 'lucide-react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || '';

interface TeamProfileData {
  // Add your data structure here when you create the API
  teamName: string;
  conference?: string;
  record?: string;
  ranking?: number;
  // Add more fields as needed
}

export default function TeamProfilePage() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const teamName = searchParams.get('team');
  
  const [profileData, setProfileData] = useState<TeamProfileData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (teamName) {
      fetchTeamProfile(teamName);
    }
  }, [teamName]);

  const fetchTeamProfile = async (team: string) => {
    try {
      setLoading(true);
      setError(null);
      
      // TODO: Replace this with your actual API call
      // const response = await fetch(`${API_URL}/api/cbase/team-profile?team=${encodeURIComponent(team)}`);
      // const data = await response.json();
      // setProfileData(data);
      
      // Simulated delay for demonstration
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Mock data - remove this when you implement your API
      setProfileData({
        teamName: team,
        conference: 'ACC',
        record: '25-5',
        ranking: 15,
      });
      
    } catch (err) {
      console.error('Error fetching team profile:', err);
      setError('Failed to load team profile');
    } finally {
      setLoading(false);
    }
  };

  if (!teamName) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800 flex items-center justify-center">
        <div className="text-center">
          <p className="text-xl text-gray-600 dark:text-gray-400">No team selected</p>
          <button
            onClick={() => router.back()}
            className="mt-4 px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Go Back
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Back Button */}
        <button
          onClick={() => router.back()}
          className="flex items-center gap-2 text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white transition-colors mb-6"
        >
          <ArrowLeft size={20} />
          <span className="font-medium">Back to Rankings</span>
        </button>

        {/* Header */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 mb-6">
          <div className="flex items-center gap-4">
            <img 
              src={`${API_URL}/api/baseball-logo/${encodeURIComponent(teamName)}`}
              alt={`${teamName} logo`}
              className="w-20 h-20 object-contain"
              onError={(e) => {
                e.currentTarget.style.display = 'none';
              }}
            />
            <div>
              <h1 className="text-4xl font-bold text-gray-900 dark:text-white">
                {teamName}
              </h1>
              <p className="text-lg text-gray-600 dark:text-gray-400 mt-1">
                Team Profile
              </p>
            </div>
          </div>
        </div>

        {/* Content Area */}
        {loading ? (
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-12">
            <div className="flex flex-col items-center justify-center">
              <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600 dark:border-blue-400"></div>
              <p className="mt-4 text-gray-600 dark:text-gray-400 text-lg">
                Loading team profile...
              </p>
            </div>
          </div>
        ) : error ? (
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-12">
            <div className="text-center">
              <p className="text-red-600 dark:text-red-400 text-lg">{error}</p>
            </div>
          </div>
        ) : (
          <div className="space-y-6">
            {/* Stats Overview - Placeholder */}
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
                Season Overview
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 rounded-lg p-4">
                  <div className="flex items-center gap-3 mb-2">
                    <Trophy className="text-blue-600 dark:text-blue-400" size={24} />
                    <h3 className="font-semibold text-gray-700 dark:text-gray-300">
                      Record
                    </h3>
                  </div>
                  <p className="text-3xl font-bold text-gray-900 dark:text-white">
                    {profileData?.record || 'N/A'}
                  </p>
                </div>

                <div className="bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 rounded-lg p-4">
                  <div className="flex items-center gap-3 mb-2">
                    <TrendingUp className="text-purple-600 dark:text-purple-400" size={24} />
                    <h3 className="font-semibold text-gray-700 dark:text-gray-300">
                      National Rank
                    </h3>
                  </div>
                  <p className="text-3xl font-bold text-gray-900 dark:text-white">
                    #{profileData?.ranking || 'N/A'}
                  </p>
                </div>

                <div className="bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 rounded-lg p-4">
                  <div className="flex items-center gap-3 mb-2">
                    <Calendar className="text-green-600 dark:text-green-400" size={24} />
                    <h3 className="font-semibold text-gray-700 dark:text-gray-300">
                      Conference
                    </h3>
                  </div>
                  <p className="text-3xl font-bold text-gray-900 dark:text-white">
                    {profileData?.conference || 'N/A'}
                  </p>
                </div>
              </div>
            </div>

            {/* Detailed Stats Section - Placeholder */}
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
                Detailed Statistics
              </h2>
              <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-8 text-center">
                <p className="text-gray-600 dark:text-gray-400 text-lg">
                  This section will be populated with data from your API
                </p>
                <p className="text-gray-500 dark:text-gray-500 text-sm mt-2">
                  Add charts, tables, and other visualizations here
                </p>
              </div>
            </div>

            {/* Schedule Section - Placeholder */}
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
                Recent Games
              </h2>
              <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-8 text-center">
                <p className="text-gray-600 dark:text-gray-400 text-lg">
                  Game history and upcoming schedule will appear here
                </p>
              </div>
            </div>

            {/* Additional Sections - Placeholder */}
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
                Advanced Metrics
              </h2>
              <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-8 text-center">
                <p className="text-gray-600 dark:text-gray-400 text-lg">
                  Advanced analytics and metrics will be displayed here
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}