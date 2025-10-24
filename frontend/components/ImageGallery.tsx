'use client';

import { useState, useEffect } from 'react';
import axios from 'axios';
import { ChevronLeft, ChevronRight } from 'lucide-react';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

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

  useEffect(() => {
    fetchImages();
  }, [year, week, type]);

  const fetchImages = async () => {
    setLoading(true);
    try {
      const endpoint = type === 'games' ? 'game-images' : 'stat-profiles';
      const response = await axios.get(`${API_URL}/api/${endpoint}/${year}/${week}`);
      const imageList = response.data.images;
      setImages(imageList);
      if (imageList.length > 0) {
        setSelectedImage(imageList[0]);
        setCurrentIndex(0);
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
      <div className="bg-white rounded-xl shadow-lg overflow-hidden">
        <div className="bg-gradient-to-r from-blue-600 to-blue-700 px-6 py-4">
          <h2 className="text-xl font-bold text-white">{title}</h2>
        </div>
        <div className="p-6 text-center">
          <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
          <p className="mt-4 text-gray-600">Loading images...</p>
        </div>
      </div>
    );
  }

  if (images.length === 0) {
    return (
      <div className="bg-white rounded-xl shadow-lg overflow-hidden">
        <div className="bg-gradient-to-r from-blue-600 to-blue-700 px-6 py-4">
          <h2 className="text-xl font-bold text-white">{title}</h2>
        </div>
        <div className="p-6 text-center text-gray-600">
          No images available for this week
        </div>
      </div>
    );
  }

  const imageUrl = `${API_URL}/api/image/${year}/${week}/${type}/${selectedImage}`;

  return (
    <div className="bg-white rounded-xl shadow-lg overflow-hidden">
      <div className="bg-gradient-to-r from-blue-600 to-blue-700 px-6 py-4">
        <h2 className="text-xl font-bold text-white">{title}</h2>
      </div>
      
      <div className="p-6">
        <div className="mb-4">
          <select
            value={selectedImage}
            onChange={(e) => handleImageChange(e.target.value, images.indexOf(e.target.value))}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            {images.map((image, index) => (
              <option key={image} value={image}>
                {image.replace('.png', '').replace(/_/g, ' ')}
              </option>
            ))}
          </select>
        </div>

        <div className="relative bg-gray-100 rounded-lg overflow-hidden">
          <img
            src={imageUrl}
            alt={selectedImage}
            className="w-full h-auto"
            onError={(e) => {
              e.currentTarget.src = '/placeholder-image.png';
            }}
          />
          
          {images.length > 1 && (
            <>
              <button
                onClick={handlePrevious}
                disabled={currentIndex === 0}
                className="absolute left-2 top-1/2 transform -translate-y-1/2 bg-white/90 hover:bg-white p-2 rounded-full shadow-lg disabled:opacity-50 disabled:cursor-not-allowed transition-all"
              >
                <ChevronLeft className="w-6 h-6 text-gray-700" />
              </button>
              
              <button
                onClick={handleNext}
                disabled={currentIndex === images.length - 1}
                className="absolute right-2 top-1/2 transform -translate-y-1/2 bg-white/90 hover:bg-white p-2 rounded-full shadow-lg disabled:opacity-50 disabled:cursor-not-allowed transition-all"
              >
                <ChevronRight className="w-6 h-6 text-gray-700" />
              </button>
            </>
          )}
        </div>

        <div className="mt-3 text-center text-sm text-gray-600">
          Image {currentIndex + 1} of {images.length}
        </div>
      </div>
    </div>
  );
}