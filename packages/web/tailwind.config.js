/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./app/**/*.{js,ts,jsx,tsx,mdx}', './components/**/*.{js,ts,jsx,tsx,mdx}'],
  theme: {
    extend: {
      colors: {
        // Regime colors
        regime: {
          democracy: '#3B82F6',
          state: '#EF4444',
          social: '#10B981',
          authoritarian: '#6B7280',
          transitional: '#F59E0B',
        },
        // Risk gradient
        risk: {
          low: '#10B981',
          medium: '#F59E0B',
          high: '#EF4444',
        },
      },
    },
  },
  plugins: [],
};
