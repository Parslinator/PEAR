'use client';

import { useState, useEffect, useRef } from 'react';
import Image from 'next/image';
import axios from 'axios';
import { ChevronLeft, ChevronRight } from 'lucide-react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || '';

interface Props {
  title: string;
  type: 'games' | 'profiles';
  year: number;
  week: number;
}

export default function ImageGallery({ title, type, year, week }: Props) {
  const [images, setImages] = useState<string[]>([]);
  const [selectedImage, setSelectedImage] = useState('');
  const [currentIndex, setCurrentIndex] = useState(0);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [filteredImages, setFilteredImages] = useState<string[]>([]);
  const [showDropdown, setShowDropdown] = useState(false);
  const [selectedDropdownIndex, setSelectedDropdownIndex] = useState(-1);
  const [fullscreenImage, setFullscreenImage] = useState<string | null>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetchImages();
  }, [year, week, type]);

  // Filter images based on search term
  useEffect(() => {
    if (searchTerm.trim() === '') {
      setFilteredImages(images);
    } else {
      const filtered = images.filter((image) => {
        const imageName = image.toLowerCase();
        const searchLower = searchTerm.toLowerCase();
        
        if (type === 'games') {
          // For games, split by common separators (vs, @, -, _)
          const parts = imageName.split(/[\s_-]+(?:vs|@)[\s_-]+/i);
          
          // Check if search term matches any part of the matchup
          return parts.some(part => part.includes(searchLower)) || imageName.includes(searchLower);
        } else {
          // For profiles, simple search
          return imageName.includes(searchLower);
        }
      });
      setFilteredImages(filtered);
    }
    setSelectedDropdownIndex(-1);
  }, [searchTerm, images, type]);

  // Handle click outside dropdown
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setShowDropdown(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Handle escape key for fullscreen modal
  useEffect(() => {
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && fullscreenImage) {
        setFullscreenImage(null);
      }
    };

    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [fullscreenImage]);

  const fetchImages = async () => {
    setLoading(true);
    try {
      const endpoint = type === 'games' ? 'game-images' : 'stat-profiles';
      const response = await axios.get(`${API_URL}/api/${endpoint}/${year}/${week}`);
      const imageList = response.data.images;
      setImages(imageList);
      setFilteredImages(imageList);
      if (imageList.length > 0) {
        setSelectedImage(imageList[0]);
        setCurrentIndex(0);
        setSearchTerm('');
      }
    } catch (error) {
      console.error('Error fetching images:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleImageChange = (image: string, index: number) => {
    setSelectedImage(image);
    setCurrentIndex(index);
  };

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchTerm(e.target.value);
    setShowDropdown(true);
  };

  const handleImageSelect = (image: string) => {
    const index = images.indexOf(image);
    setSelectedImage(image);
    setCurrentIndex(index);
    setSearchTerm(image.replace('.png', '').replace(/_/g, ' '));
    setShowDropdown(false);
    setSelectedDropdownIndex(-1);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (!showDropdown) {
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        setShowDropdown(true);
      }
      return;
    }

    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        setSelectedDropdownIndex((prev) =>
          prev < filteredImages.length - 1 ? prev + 1 : prev
        );
        break;
      case 'ArrowUp':
        e.preventDefault();
        setSelectedDropdownIndex((prev) => (prev > 0 ? prev - 1 : -1));
        break;
      case 'Enter':
        e.preventDefault();
        if (selectedDropdownIndex >= 0 && selectedDropdownIndex < filteredImages.length) {
          handleImageSelect(filteredImages[selectedDropdownIndex]);
        } else if (filteredImages.length === 1) {
          handleImageSelect(filteredImages[0]);
        }
        break;
      case 'Escape':
        setShowDropdown(false);
        setSelectedDropdownIndex(-1);
        break;
    }
  };

  const handlePrevious = () => {
    if (currentIndex > 0) {
      const newIndex = currentIndex - 1;
      setSelectedImage(images[newIndex]);
      setCurrentIndex(newIndex);
    }
  };

  const handleNext = () => {
    if (currentIndex < images.length - 1) {
      const newIndex = currentIndex + 1;
      setSelectedImage(images[newIndex]);
      setCurrentIndex(newIndex);
    }
  };

  if (loading) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg overflow-hidden">
        <div className="bg-gradient-to-r from-blue-600 to-blue-700 dark:from-blue-700 dark:to-blue-800 px-6 py-4">
          <h2 className="text-xl font-bold text-white">{title}</h2>
        </div>
        <div className="p-6 text-center">
          <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 dark:border-blue-400"></div>
          <p className="mt-4 text-gray-600 dark:text-gray-400">Loading images...</p>
        </div>
      </div>
    );
  }

  if (images.length === 0) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg overflow-hidden">
        <div className="bg-gradient-to-r from-blue-600 to-blue-700 dark:from-blue-700 dark:to-blue-800 px-6 py-4">
          <h2 className="text-xl font-bold text-white">{title}</h2>
        </div>
        <div className="p-6 text-center text-gray-600 dark:text-gray-400">
          No images available for this week
        </div>
      </div>
    );
  }

  const imageUrl = `${API_URL}/api/image/${year}/${week}/${type}/${selectedImage}`;

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg overflow-hidden">
      <div className="bg-gradient-to-r from-blue-600 to-blue-700 dark:from-blue-700 dark:to-blue-800 px-6 py-4">
        <h2 className="text-xl font-bold text-white">{title}</h2>
      </div>
      
      <div className="p-6">
        {/* Searchable dropdown for both games and profiles */}
        <div className="mb-4 relative" ref={dropdownRef}>
          <input
            ref={inputRef}
            type="text"
            value={searchTerm}
            onChange={handleSearchChange}
            onFocus={() => setShowDropdown(true)}
            onKeyDown={handleKeyDown}
            placeholder={type === 'games' ? "Type to search games..." : "Type to search team profiles..."}
            className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400"
            autoComplete="off"
          />
          
          {/* Dropdown menu */}
          {showDropdown && filteredImages.length > 0 && (
            <div className="absolute z-10 w-full mt-1 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg shadow-lg max-h-60 overflow-auto">
              {filteredImages.map((image, index) => (
                <div
                  key={image}
                  onClick={() => handleImageSelect(image)}
                  className={`px-4 py-2 cursor-pointer text-gray-900 dark:text-white ${
                    index === selectedDropdownIndex
                      ? 'bg-blue-100 dark:bg-blue-900/50'
                      : 'hover:bg-gray-100 dark:hover:bg-gray-600'
                  }`}
                >
                  {image.replace('.png', '').replace(/_/g, ' ')}
                </div>
              ))}
            </div>
          )}

          {/* No results message */}
          {showDropdown && filteredImages.length === 0 && searchTerm && (
            <div className="absolute z-10 w-full mt-1 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg shadow-lg">
              <div className="px-4 py-2 text-gray-500 dark:text-gray-400">
                {type === 'games' ? 'No games found' : 'No profiles found'}
              </div>
            </div>
          )}
        </div>

        <div className="relative bg-gray-100 dark:bg-gray-700 rounded-lg overflow-hidden">
          <div 
            className="cursor-pointer"
            onClick={() => setFullscreenImage(imageUrl)}
          >
            <img
              src={imageUrl}
              alt={selectedImage}
              className="w-full h-auto hover:opacity-90 transition-opacity"
              onError={(e) => {
                e.currentTarget.src = '/placeholder-image.png';
              }}
            />
          </div>
          
          {images.length > 1 && (
            <>
              <button
                onClick={handlePrevious}
                disabled={currentIndex === 0}
                className="absolute left-2 top-1/2 transform -translate-y-1/2 bg-white/90 dark:bg-gray-800/90 hover:bg-white dark:hover:bg-gray-800 p-2 rounded-full shadow-lg disabled:opacity-50 disabled:cursor-not-allowed transition-all z-10"
              >
                <ChevronLeft className="w-6 h-6 text-gray-700 dark:text-gray-200" />
              </button>
              
              <button
                onClick={handleNext}
                disabled={currentIndex === images.length - 1}
                className="absolute right-2 top-1/2 transform -translate-y-1/2 bg-white/90 dark:bg-gray-800/90 hover:bg-white dark:hover:bg-gray-800 p-2 rounded-full shadow-lg disabled:opacity-50 disabled:cursor-not-allowed transition-all z-10"
              >
                <ChevronRight className="w-6 h-6 text-gray-700 dark:text-gray-200" />
              </button>
            </>
          )}
        </div>

        <div className="mt-3 text-center text-sm text-gray-600 dark:text-gray-400">
          Image {currentIndex + 1} of {images.length}
        </div>
      </div>

      {/* Fullscreen Modal */}
      {fullscreenImage && (
        <div
          className="fixed inset-0 z-50 bg-black bg-opacity-90 dark:bg-opacity-95 flex items-center justify-center p-4"
          onClick={() => setFullscreenImage(null)}
        >
          <button
            onClick={() => setFullscreenImage(null)}
            className="absolute top-4 right-4 text-white hover:text-gray-300 dark:hover:text-gray-400 text-4xl font-bold z-10"
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
  );
}