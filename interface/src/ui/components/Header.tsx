import React from 'react'

export const Header: React.FC = () => {
  return (
    <header className="bg-dark-blue w-full px-6 py-4 flex items-center justify-between">
      {/* Settings Icon */}
      <button className="text-white hover:opacity-80 transition-opacity">
        <svg
          xmlns="http://www.w3.org/2000/svg"
          className="h-6 w-6"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={2}
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
          />
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
          />
        </svg>
      </button>

      {/* Logo */}
      <div className="flex items-center gap-3">
        <div className="relative">
          {/* Wheelchair person icon with mesh pattern */}
          <svg
            width="48"
            height="48"
            viewBox="0 0 48 48"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            {/* Wheelchair base */}
            <circle cx="24" cy="36" r="6" fill="white" />
            <circle cx="24" cy="36" r="4" fill="none" stroke="white" strokeWidth="1.5" />
            <circle cx="20" cy="36" r="1.5" fill="white" />
            <circle cx="28" cy="36" r="1.5" fill="white" />
            
            {/* Person body */}
            <circle cx="24" cy="20" r="4" fill="white" />
            <rect x="20" y="24" width="8" height="10" fill="white" rx="1" />
            
            {/* Mesh pattern on body */}
            <path
              d="M20 26 L24 28 L28 26 M20 30 L24 32 L28 30 M22 28 L22 32 M26 28 L26 32"
              stroke="dark-blue"
              strokeWidth="0.8"
              fill="none"
            />
            
            {/* Arms */}
            <line x1="16" y1="26" x2="20" y2="28" stroke="white" strokeWidth="2" strokeLinecap="round" />
            <line x1="28" y1="28" x2="32" y2="26" stroke="white" strokeWidth="2" strokeLinecap="round" />
          </svg>
        </div>
        <span className="text-white text-2xl font-semibold">wai</span>
      </div>

      {/* Help Icon */}
      <button className="text-white hover:opacity-80 transition-opacity">
        <svg
          xmlns="http://www.w3.org/2000/svg"
          className="h-6 w-6"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={2}
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
          />
        </svg>
      </button>
    </header>
  )
}
