'use client';

import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import { DollarSign, Moon, Sun } from 'lucide-react';
import Image from 'next/image';
import { useTheme } from '../app/contexts/ThemeContext';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function Navbar() {
  const pathname = usePathname();
  const router = useRouter();
  const { darkMode, toggleDarkMode } = useTheme();

  // Determine current sport from pathname
  const isCBASE = pathname.startsWith('/cbase');

  const toggleSport = () => {
    if (isCBASE) {
      router.push('/');
    } else {
      router.push('/cbase');
    }
  };

  // Navigation items for each sport
  const footballNavItems = [
    { name: 'Ratings', path: '/' },
    { name: 'Stats', path: '/stats' },
    { name: 'Spreads', path: '/spreads' },
    { name: 'Matchups', path: '/matchups' },
    { name: 'Previews', path: '/previews' },
  ];

  const baseballNavItems = [
    { name: 'Ratings', path: '/cbase' },
    { name: 'Stats', path: '/cbase/stats' },
    { name: 'Tournament', path: '/cbase/tournament' },
    { name: 'Matchups', path: '/cbase/matchups' },
    { name: 'Profiles', path: '/cbase/profiles' },
  ];

  const navItems = isCBASE ? baseballNavItems : footballNavItems;

  return (
    <nav className="text-gray-900 dark:text-gray-100 shadow-lg fixed top-0 left-0 right-0 z-50 bg-[#CECEB2] dark:bg-gray-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo/Brand with Sport Switcher */}
          <div className="flex items-center space-x-4">
            <Link 
              href={isCBASE ? '/cbase' : '/'} 
              className="flex items-center space-x-3 hover:opacity-80 transition-opacity"
            >
              <Image 
                src={`${API_URL}/api/logo`}
                alt="PEAR Logo"
                width={40}
                height={40}
                className="rounded"
                unoptimized
              />
              <div>
                <div className="text-xl font-bold text-gray-900 dark:text-gray-100">
                  {isCBASE ? 'CBASE PEAR' : 'CFB PEAR'}
                </div>
                <div className="text-xs text-gray-700 dark:text-gray-300">
                  {isCBASE ? 'College Baseball Analytics' : 'College Football Analytics'}
                </div>
              </div>
            </Link>

            {/* Sport Switcher Toggle Button */}
            <button
              onClick={toggleSport}
              className="px-4 py-2 rounded-lg transition-colors text-white font-semibold bg-[#fc8884] hover:bg-[#f5645f]"
            >
              {isCBASE ? 'Baseball' : 'Football'}
            </button>
          </div>

          {/* Navigation Links */}
          <div className="hidden md:flex items-center space-x-1">
            {navItems.map((item) => {
              const isActive = pathname === item.path;
              return (
                <Link
                  key={item.path}
                  href={item.path}
                  className={`px-4 py-2 rounded-lg font-semibold transition-all ${
                    isActive
                      ? 'bg-gray-700 text-white'
                      : 'text-gray-700 dark:text-gray-200 hover:bg-gray-600 hover:text-white'
                  }`}
                >
                  {item.name}
                </Link>
              );
            })}
            
            {/* Dark Mode Toggle */}
            <button
              onClick={toggleDarkMode}
              className="ml-2 p-2 rounded-lg bg-gray-700 hover:bg-gray-600 text-white transition-all"
              aria-label="Toggle dark mode"
            >
              {darkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
            </button>
            
            {/* Support Button */}
            <a
              href="https://buymeacoffee.com/PEARatings"
              target="_blank"
              rel="noopener noreferrer"
              className="p-2 rounded-lg bg-yellow-500 hover:bg-yellow-600 text-gray-900 transition-all"
              aria-label="Support on Buy Me a Coffee"
            >
              <DollarSign className="w-5 h-5" />
            </a>
          </div>

          {/* Mobile Menu Button */}
          <div className="md:hidden">
            <button className="text-gray-700 dark:text-gray-200 hover:text-gray-900 dark:hover:text-white p-2">
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
                      ? 'bg-gray-700 text-white'
                      : 'text-gray-700 dark:text-gray-200 hover:bg-gray-600 hover:text-white'
                  }`}
                >
                  {item.name}
                </Link>
              );
            })}
            
            {/* Sport Switcher - Mobile */}
            <button
              onClick={toggleSport}
              className="flex items-center justify-center space-x-2 px-4 py-2 rounded-lg font-semibold transition-colors text-white bg-[#fc8884] hover:bg-[#f5645f]"
            >
              <span>{isCBASE ? '⚾ Switch to Football' : '🏈 Switch to Baseball'}</span>
            </button>
            
            {/* Dark Mode Toggle - Mobile */}
            <button
              onClick={toggleDarkMode}
              className="flex items-center space-x-2 px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg font-semibold transition-all"
            >
              {darkMode ? (
                <>
                  <Sun className="w-4 h-4" />
                  <span>Light Mode</span>
                </>
              ) : (
                <>
                  <Moon className="w-4 h-4" />
                  <span>Dark Mode</span>
                </>
              )}
            </button>
            
            {/* Support Button - Mobile */}
            <a
              href="https://buymeacoffee.com/PEARatings"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center space-x-2 px-4 py-2 bg-yellow-500 hover:bg-yellow-600 text-gray-900 rounded-lg font-semibold transition-all"
            >
              <DollarSign className="w-4 h-4" />
              <span>Support on Buy Me a Coffee</span>
            </a>
          </div>
        </div>
      </div>
    </nav>
  );
}