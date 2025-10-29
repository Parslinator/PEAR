/** @type {import('next').NextConfig} */
const nextConfig = {
  typescript: {
    ignoreBuildErrors: true,
  },
  eslint: {
    ignoreDuringBuilds: true,
  },
  images: {
    domains: ['localhost', '129.212.189.128', 'pearatings.com', 'www.pearatings.com'],
    remotePatterns: [
      {
        protocol: 'http',
        hostname: 'localhost',
        port: '8000',
        pathname: '/api/image/**',
      },
      {
        protocol: 'http',
        hostname: '129.212.189.128',
        pathname: '/api/image/**',
      },
      {
        protocol: 'http',
        hostname: 'pearatings.com',
        pathname: '/api/**',
      },
      {
        protocol: 'https',
        hostname: 'pearatings.com',
        pathname: '/api/**',
      },
      {
        protocol: 'http',
        hostname: 'www.pearatings.com',
        pathname: '/api/**',
      },
      {
        protocol: 'https',
        hostname: 'www.pearatings.com',
        pathname: '/api/**',
      },
    ],
  },
}
module.exports = nextConfig