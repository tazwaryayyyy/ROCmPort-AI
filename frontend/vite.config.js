import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
    plugins: [react()],
    server: {
        port: 5173,
        proxy: {
            '/port': 'http://localhost:8000',
            '/health': 'http://localhost:8000',
            '/demo-kernels': 'http://localhost:8000',
            '/benchmark-report': 'http://localhost:8000',
            '/cold-start': 'http://localhost:8000',
            '/aggregate-metric': 'http://localhost:8000',
            '/recompile': 'http://localhost:8000',
            '/export': 'http://localhost:8000',
        },
    },
})
