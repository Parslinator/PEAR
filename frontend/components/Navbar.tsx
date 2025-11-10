'use client';

import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import { DollarSign, Moon, Sun, Menu, X, ChevronDown } from 'lucide-react';
import Image from 'next/image';
import { useTheme } from '../app/contexts/ThemeContext';
import { useState, useRef, useEffect } from 'react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || '';

export default function Navbar() {
  const pathname = usePathname();
  const router = useRouter();
  const { darkMode, toggleDarkMode } = useTheme();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [sportDropdownOpen, setSportDropdownOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Determine current sport from pathname
  const isCBASE = pathname.startsWith('/cbase');
  const isCSOFT = pathname.startsWith('/csoft');

  const getCurrentSport = () => {
    if (isCBASE) return 'Baseball';
    if (isCSOFT) return 'Softball';
    return 'Football';
  };

  const navigateToSport = (sport: string) => {
    switch (sport) {
      case 'Football':
        router.push('/');
        break;
      case 'Baseball':
        router.push('/cbase');
        break;
      case 'Softball':
        router.push('/csoft');
        break;
    }
    setSportDropdownOpen(false);
    setMobileMenuOpen(false);
  };

  const toggleMobileMenu = () => {
    setMobileMenuOpen(!mobileMenuOpen);
  };

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setSportDropdownOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  // Navigation items for each sport
  const footballNavItems = [
    { name: 'Ratings', path: '/' },
    { name: 'Stats', path: '/stats' },
    { name: 'Games', path: '/games' },
    { name: 'Matchups', path: '/matchups' },
    { name: 'Previews', path: '/previews' },
    { name: 'Teams', path: '/teams' },
  ];

  const baseballNavItems = [
    { name: 'Ratings', path: '/cbase' },
    { name: 'Stats', path: '/cbase/stats' },
    { name: 'Games', path: '/cbase/games' },
    { name: 'Tournament', path: '/cbase/tournament' },
    { name: 'Matchups', path: '/cbase/matchups' },
    { name: 'Teams', path: '/cbase/teams' },
  ];

  const softballNavItems = [
    { name: 'Ratings', path: '/csoft' },
    { name: 'Stats', path: '/csoft/stats' },
    { name: 'Games', path: '/csoft/games' },
    { name: 'Tournament', path: '/csoft/tournament' },
    { name: 'Matchups', path: '/csoft/matchups' },
    { name: 'Teams', path: '/csoft/teams' },
  ];

  const getNavItems = () => {
    if (isCBASE) return baseballNavItems;
    if (isCSOFT) return softballNavItems;
    return footballNavItems;
  };

  const getBrandName = () => {
    if (isCBASE) return 'CBASE PEAR';
    if (isCSOFT) return 'CSOFT PEAR';
    return 'CFB PEAR';
  };

  const getBrandDescription = () => {
    if (isCBASE) return 'College Baseball Analytics';
    if (isCSOFT) return 'College Softball Analytics';
    return 'College Football Analytics';
  };

  const getHomePath = () => {
    if (isCBASE) return '/cbase';
    if (isCSOFT) return '/csoft';
    return '/';
  };

  const navItems = getNavItems();
  const sports = ['Football', 'Baseball', 'Softball'];
  const currentSport = getCurrentSport();

  return (
    <nav className="text-gray-900 shadow-lg fixed top-0 left-0 right-0 z-50 bg-[#CECEB2]">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo/Brand with Sport Switcher */}
          <div className="flex items-center space-x-4">
            <Link 
              href={getHomePath()} 
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
                <div className="text-xl font-bold text-gray-900">
                  {getBrandName()}
                </div>
                <div className="text-xs text-gray-700">
                  {getBrandDescription()}
                </div>
              </div>
            </Link>

            {/* Sport Switcher Dropdown Button - Desktop */}
            <div className="hidden md:block relative" ref={dropdownRef}>
              <button
                onClick={() => setSportDropdownOpen(!sportDropdownOpen)}
                className="px-4 py-2 rounded-lg transition-colors text-white font-semibold bg-[#fc8884] hover:bg-[#f5645f] flex items-center space-x-2"
              >
                <span>{currentSport}</span>
                <ChevronDown className={`w-4 h-4 transition-transform ${sportDropdownOpen ? 'rotate-180' : ''}`} />
              </button>
              
              {/* Dropdown Menu */}
              {sportDropdownOpen && (
                <div className="absolute top-full mt-1 left-0 bg-white rounded-lg shadow-lg border border-gray-200 py-1 min-w-[150px]">
                  {sports.map((sport) => (
                    <button
                      key={sport}
                      onClick={() => navigateToSport(sport)}
                      className={`w-full text-left px-4 py-2 hover:bg-gray-100 transition-colors ${
                        sport === currentSport ? 'bg-gray-50 font-semibold text-[#fc8884]' : 'text-gray-700'
                      }`}
                    >
                      {sport}
                    </button>
                  ))}
                </div>
              )}
            </div>
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
                      : 'text-gray-700 hover:bg-gray-600 hover:text-white'
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
            <button 
              onClick={toggleMobileMenu}
              className="text-gray-700 hover:text-gray-900 p-2"
              aria-label="Toggle mobile menu"
            >
              {mobileMenuOpen ? (
                <X className="w-6 h-6" />
              ) : (
                <Menu className="w-6 h-6" />
              )}
            </button>
          </div>
        </div>

        {/* Mobile Menu */}
        {mobileMenuOpen && (
          <div className="md:hidden pb-4">
            <div className="flex flex-col space-y-1">
              {navItems.map((item) => {
                const isActive = pathname === item.path;
                return (
                  <Link
                    key={item.path}
                    href={item.path}
                    onClick={() => setMobileMenuOpen(false)}
                    className={`px-4 py-2 rounded-lg font-semibold transition-all ${
                      isActive
                        ? 'bg-gray-700 text-white'
                        : 'text-gray-700 hover:bg-gray-600 hover:text-white'
                    }`}
                  >
                    {item.name}
                  </Link>
                );
              })}
              
              {/* Sport Switcher - Mobile */}
              <div className="border-t border-gray-300 pt-2 mt-2">
                <div className="px-4 py-2 text-xs font-semibold text-gray-600 uppercase">
                  Switch Sport
                </div>
                {sports.map((sport) => (
                  <button
                    key={sport}
                    onClick={() => navigateToSport(sport)}
                    className={`w-full text-left px-4 py-2 rounded-lg font-semibold transition-colors ${
                      sport === currentSport
                        ? 'bg-[#fc8884] text-white'
                        : 'text-gray-700 hover:bg-gray-600 hover:text-white'
                    }`}
                  >
                    {sport}
                  </button>
                ))}
              </div>
              
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
        )}
      </div>
    </nav>
  );
}