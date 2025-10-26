'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Trophy, ChevronDown } from 'lucide-react';
import { useState } from 'react';

export default function Navbar() {
  const pathname = usePathname();
  const [sportDropdownOpen, setSportDropdownOpen] = useState(false);

  // Determine current sport from pathname
  const isCBASE = pathname.startsWith('/cbase');
  const currentSport = isCBASE ? 'baseball' : 'football';

  // Navigation items for each sport
  const footballNavItems = [
    { name: 'Ratings', path: '/' },
    { name: 'Stats', path: '/stats' },
    { name: 'Spreads', path: '/spreads' },
    { name: 'Matchups', path: '/matchups' },
    { name: 'Previews', path: '/previews' },
    { name: 'Archive', path: '/archive' },
  ];

  const baseballNavItems = [
    { name: 'Ratings', path: '/cbase' },
    { name: 'Stats', path: '/cbase/stats' },
    { name: 'Tournament', path: '/cbase/tournament' },
    { name: 'Matchups', path: '/cbase/matchups' },
    { name: 'Profiles', path: '/cbase/profiles' },
    { name: 'Conferences', path: '/cbase/conferences' },
  ];

  const navItems = isCBASE ? baseballNavItems : footballNavItems;

  return (
    <nav className="bg-gradient-to-r from-blue-900 via-blue-800 to-blue-900 text-white shadow-lg sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo/Brand with Sport Switcher */}
          <div className="flex items-center space-x-4">
            <Link 
              href={isCBASE ? '/cbase' : '/'} 
              className="flex items-center space-x-3 hover:opacity-80 transition-opacity"
            >
              <Trophy className="w-8 h-8 text-yellow-400" />
              <div>
                <div className="text-xl font-bold">
                  {isCBASE ? 'CBASE PEAR' : 'CFB PEAR'}
                </div>
                <div className="text-xs text-blue-200">
                  {isCBASE ? 'College Baseball Analytics' : 'College Football Analytics'}
                </div>
              </div>
            </Link>

            {/* Sport Switcher Dropdown */}
            <div className="relative">
              <button
                onClick={() => setSportDropdownOpen(!sportDropdownOpen)}
                className="flex items-center space-x-1 px-3 py-2 bg-blue-800 hover:bg-blue-700 rounded-lg transition-colors"
              >
                <span className="text-sm font-semibold">
                  {isCBASE ? 'Baseball' : 'Football'}
                </span>
                <ChevronDown className="w-4 h-4" />
              </button>

              {sportDropdownOpen && (
                <div className="absolute top-full left-0 mt-2 w-48 bg-white rounded-lg shadow-xl overflow-hidden z-50">
                  <Link
                    href="/"
                    onClick={() => setSportDropdownOpen(false)}
                    className="block px-4 py-3 text-gray-700 hover:bg-blue-50 transition-colors"
                  >
                    <div className="font-semibold">Football</div>
                    <div className="text-xs text-gray-500">CFB PEAR</div>
                  </Link>
                  <Link
                    href="/cbase"
                    onClick={() => setSportDropdownOpen(false)}
                    className="block px-4 py-3 text-gray-700 hover:bg-blue-50 transition-colors border-t border-gray-100"
                  >
                    <div className="font-semibold">Baseball</div>
                    <div className="text-xs text-gray-500">CBASE PEAR</div>
                  </Link>
                </div>
              )}
            </div>
          </div>

          {/* Navigation Links */}
          <div className="hidden md:flex space-x-1">
            {navItems.map((item) => {
              const isActive = pathname === item.path;
              return (
                <Link
                  key={item.path}
                  href={item.path}
                  className={`px-4 py-2 rounded-lg font-semibold transition-all ${
                    isActive
                      ? 'bg-blue-700 text-white'
                      : 'text-blue-100 hover:bg-blue-800 hover:text-white'
                  }`}
                >
                  {item.name}
                </Link>
              );
            })}
          </div>

          {/* Mobile Menu Button */}
          <div className="md:hidden">
            <button className="text-blue-100 hover:text-white p-2">
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
          </div>
        </div>

        {/* Mobile Menu */}
        <div className="md:hidden pb-4">
          <div className="flex flex-col space-y-1">
            {navItems.map((item) => {
              const isActive = pathname === item.path;
              return (
                <Link
                  key={item.path}
                  href={item.path}
                  className={`px-4 py-2 rounded-lg font-semibold transition-all ${
                    isActive
                      ? 'bg-blue-700 text-white'
                      : 'text-blue-100 hover:bg-blue-800 hover:text-white'
                  }`}
                >
                  {item.name}
                </Link>
              );
            })}
          </div>
        </div>
      </div>
    </nav>
  );
}