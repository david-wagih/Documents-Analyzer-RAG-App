/** @type {import('tailwindcss').Config} */
module.exports = {
    content: [
        "./app/**/*.{js,ts,jsx,tsx,mdx}",
        "./pages/**/*.{js,ts,jsx,tsx,mdx}",
        "./components/**/*.{js,ts,jsx,tsx,mdx}",
    ],
    theme: {
        extend: {
            typography: {
                DEFAULT: {
                    css: {
                        maxWidth: 'none',
                        color: '#374151',
                        h2: {
                            color: '#1f2937',
                        },
                        h3: {
                            color: '#374151',
                        },
                        strong: {
                            color: '#4b5563',
                        },
                        a: {
                            color: '#2563eb',
                            '&:hover': {
                                color: '#1d4ed8',
                            },
                        },
                        hr: {
                            borderColor: '#e5e7eb',
                            marginTop: '2rem',
                            marginBottom: '2rem',
                        },
                        ul: {
                            listStyleType: 'none',
                            paddingLeft: '0',
                        },
                        li: {
                            marginTop: '0.5rem',
                            marginBottom: '0.5rem',
                        },
                    },
                },
            },
        },
    },
    plugins: [
        require('@tailwindcss/typography'),
    ],
} 