wAI Imaging Interface
====================

React-based frontend interface for the wAI imaging system.

## Setup

Prerequisites: Node 18+

Install dependencies:
```bash
cd interface
npm install
```

Run development server:
```bash
npm run dev
```

Open `http://localhost:5174` in your browser.

## Build

Build for production:
```bash
npm run build
```

## Structure

- `src/main.tsx` - Application entry point
- `src/ui/App.tsx` - Main application component
- `src/ui/components/` - React components
  - `Header.tsx` - Top navigation bar with logo and icons
  - `ImagingView.tsx` - Main imaging interface with form inputs
  - `ProgressIndicator.tsx` - Step progress indicator

## Technologies

- React 18.3.1
- TypeScript
- Vite
- TailwindCSS
