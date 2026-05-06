/** @type {import('tailwindcss').Config} */
export default {
    content: ['./index.html', './src/**/*.{js,jsx}'],
    theme: {
        extend: {
            fontFamily: {
                code: ['"JetBrains Mono"', 'monospace'],
                ui: ['Inter', 'sans-serif'],
            },
            keyframes: {
                'rocm-pulse': {
                    '0%, 100%': { opacity: '1' },
                    '50%': { opacity: '0.3' },
                },
            },
            animation: {
                'rocm-pulse': 'rocm-pulse 1.2s ease-in-out infinite',
            },
        },
    },
    plugins: [],
}
