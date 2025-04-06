import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/pytorch/:path*', // Matches all requests to /api/torch/*
        destination: 'http://127.0.0.1:5000/api/pytorch/:path*', // Replace with your Flask server URL
      },
    ];
  },
};

export default nextConfig;
