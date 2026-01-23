/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'dark-blue': '#0C3A67',
        'light-blue': '#51B6FF',
        'accent-blue': '#2F80ED',
      },
    },
  },
  plugins: [],
}
