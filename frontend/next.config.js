/** @type {import('next').NextConfig} */
const nextConfig = {
  typescript: {
    ignoreBuildErrors: true,
  },
  eslint: {
    ignoreDuringBuilds: true,
  },
  images: {
    domains: ['localhost', '129.212.189.128'],
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
    ],
  },
}
module.exports = nextConfig