import React from 'react'

interface ImagingViewProps {
  sex: 'female' | 'male' | ''
  height: string
  heightUnit: 'cm' | 'in'
  streamUrl: string
  onSexChange: (sex: 'female' | 'male' | '') => void
  onHeightChange: (height: string) => void
  onHeightUnitChange: (unit: 'cm' | 'in') => void
  onCapture: () => void
  busy: boolean
  lastCapture: { rgb_path?: string; depth_path?: string; timestamp?: string } | null
}

export const ImagingView: React.FC<ImagingViewProps> = ({
  sex,
  height,
  heightUnit,
  streamUrl,
  onSexChange,
  onHeightChange,
  onHeightUnitChange,
  onCapture,
  busy,
  lastCapture,
}) => {
  return (
    <div className="w-full max-w-6xl">
      {/* Instruction Text */}
      <p className="text-center text-gray-700 mb-8 text-lg">
        Please centre the user in the outline & input sex and height values
      </p>

      <div className="flex gap-8 items-start justify-center">
        {/* Image Display Area with Overlays */}
        <div className="relative flex-shrink-0">
          <div className="relative w-[640px] h-[480px] bg-gray-100 rounded-lg overflow-hidden border-2 border-light-blue">
            {/* RealSense Video Stream */}
            <img
              src={streamUrl}
              alt="RealSense RGB Stream"
              className="w-full h-full object-cover"
              style={{ imageRendering: 'auto' }}
            />

            {/* Corner Brackets Overlay */}
            <div className="absolute inset-0 pointer-events-none">
              {/* Top-left bracket */}
              <div className="absolute top-4 left-4 w-12 h-12">
                <div className="absolute top-0 left-0 w-8 h-1 bg-light-blue"></div>
                <div className="absolute top-0 left-0 w-1 h-8 bg-light-blue"></div>
              </div>
              {/* Top-right bracket */}
              <div className="absolute top-4 right-4 w-12 h-12">
                <div className="absolute top-0 right-0 w-8 h-1 bg-light-blue"></div>
                <div className="absolute top-0 right-0 w-1 h-8 bg-light-blue"></div>
              </div>
              {/* Bottom-left bracket */}
              <div className="absolute bottom-4 left-4 w-12 h-12">
                <div className="absolute bottom-0 left-0 w-8 h-1 bg-light-blue"></div>
                <div className="absolute bottom-0 left-0 w-1 h-8 bg-light-blue"></div>
              </div>
              {/* Bottom-right bracket */}
              <div className="absolute bottom-4 right-4 w-12 h-12">
                <div className="absolute bottom-0 right-0 w-8 h-1 bg-light-blue"></div>
                <div className="absolute bottom-0 right-0 w-1 h-8 bg-light-blue"></div>
              </div>
            </div>

            {/* Crosshair Overlay */}
            <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 pointer-events-none">
              <div className="w-8 h-8 relative">
                <div className="absolute top-1/2 left-0 w-full h-0.5 bg-light-blue transform -translate-y-1/2"></div>
                <div className="absolute left-1/2 top-0 w-0.5 h-full bg-light-blue transform -translate-x-1/2"></div>
              </div>
            </div>
          </div>
        </div>

        {/* Data Input Section */}
        <div className="flex flex-col gap-6 min-w-[280px]">
          {/* Sex Input */}
          <div>
            <label className="block text-gray-700 mb-2">
              Sex <span className="text-red-500">required</span>
            </label>
            <div className="flex gap-6">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={sex === 'female'}
                  onChange={(e) => {
                    if (e.target.checked) {
                      onSexChange('female')
                    } else {
                      onSexChange('')
                    }
                  }}
                  className="w-5 h-5 text-light-blue border-gray-300 rounded focus:ring-light-blue"
                />
                <span className="text-gray-700">Female</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={sex === 'male'}
                  onChange={(e) => {
                    if (e.target.checked) {
                      onSexChange('male')
                    } else {
                      onSexChange('')
                    }
                  }}
                  className="w-5 h-5 text-light-blue border-gray-300 rounded focus:ring-light-blue"
                />
                <span className="text-gray-700">Male</span>
              </label>
            </div>
          </div>

          {/* Height Input */}
          <div>
            <label className="block text-gray-700 mb-2">Height</label>
            <div className="flex gap-2">
              <input
                type="number"
                value={height}
                onChange={(e) => onHeightChange(e.target.value)}
                placeholder=""
                className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-light-blue focus:border-transparent"
              />
              <select
                value={heightUnit}
                onChange={(e) => onHeightUnitChange(e.target.value as 'cm' | 'in')}
                className="px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-light-blue focus:border-transparent bg-white"
              >
                <option value="cm">(cm)</option>
                <option value="in">(in)</option>
              </select>
            </div>
          </div>
        </div>
      </div>

      {/* Capture Button */}
      <div className="flex flex-col items-center gap-4 mt-12">
        <button
          onClick={onCapture}
          disabled={busy}
          className={`w-20 h-20 rounded-full transition-colors flex items-center justify-center shadow-lg ${
            busy
              ? 'bg-gray-400 cursor-not-allowed'
              : 'bg-gray-200 hover:bg-gray-300'
          }`}
          aria-label="Capture image"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-10 w-10 text-white"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z"
            />
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M15 13a3 3 0 11-6 0 3 3 0 016 0z"
            />
          </svg>
        </button>
        {lastCapture && (
          <div className="text-sm text-gray-600 text-center">
            <p className="font-semibold text-green-600">✓ Captured successfully!</p>
            <p className="text-xs mt-1">
              RGB: {lastCapture.rgb_path?.split(/[/\\]/).pop()}
            </p>
            <p className="text-xs">
              Depth: {lastCapture.depth_path?.split(/[/\\]/).pop()}
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
