/** @type {import('next').NextConfig} */
const nextConfig = {
    reactStrictMode: true,
    swcMinify: true,
    images: {
        domains: ['localhost'],
    },
    async rewrites() {
        return [
            {
                source: '/api/:path*',
                destination: 'http://localhost:8000/api/:path*',
            },
        ]
    },
    webpack: (config) => {
        config.resolve.alias.canvas = false;

        // Add this rule for pdf.worker.js
        config.module.rules.push({
            test: /pdf\.worker\.(min\.)?js/,
            type: 'asset/resource',
            generator: {
                filename: 'static/worker/[hash][ext][query]',
            },
        });

        return config;
    },
}

module.exports = nextConfig 