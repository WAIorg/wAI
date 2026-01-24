import React, { useState, useEffect } from 'react'
import { NumberPad } from './NumberPad'

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
  const [showNumberPad, setShowNumberPad] = useState(false)
  const [tempHeight, setTempHeight] = useState(height)
  const [showSexError, setShowSexError] = useState(false)

  const handleCaptureClick = () => {
    if (!sex) {
      setShowSexError(true)
      // Hide error after 3 seconds
      setTimeout(() => setShowSexError(false), 3000)
      return
    }
    setShowSexError(false)
    onCapture()
  }

  // Clear error when sex is selected
  useEffect(() => {
    if (sex) {
      setShowSexError(false)
    }
  }, [sex])
  return (
    <div className="w-full max-w-6xl">
      <div className="flex gap-20 items-start justify-center w-full max-w-7xl">
        {/* Image Display Area with Overlays */}
        <div className="relative flex-shrink-0">
          {/* Instruction Text for Image */}
          <p className="text-center text-dark-blue mb-6 text-3xl font-bold">
            Please centre the user in the outline
          </p>
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
        <div className="flex flex-col gap-10 flex-1 max-w-md overflow-visible">
          {/* Instruction Text for Inputs */}
          <p className="text-dark-blue text-3xl font-bold mb-2">
            Input sex and height values
          </p>
          
          {/* Sex Input */}
          <div>
            <label className="block text-dark-blue mb-5 text-2xl font-semibold">
              Sex <span className="text-red-500 ml-3">required</span>
            </label>
            <div className="flex gap-12">
              <label className="flex items-center gap-5 cursor-pointer touch-manipulation">
                <div className="relative">
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
                    className="w-8 h-8 border-2 border-gray-300 rounded-lg focus:ring-light-blue appearance-none checked:bg-light-blue checked:border-light-blue"
                  />
                  {sex === 'female' && (
                    <svg
                      className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-5 h-5 text-white pointer-events-none"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                      strokeWidth={3}
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                    </svg>
                  )}
                </div>
                <span className="text-dark-blue text-2xl font-medium">Female</span>
              </label>
              <label className="flex items-center gap-5 cursor-pointer touch-manipulation">
                <div className="relative">
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
                    className="w-8 h-8 border-2 border-gray-300 rounded-lg focus:ring-light-blue appearance-none checked:bg-light-blue checked:border-light-blue"
                  />
                  {sex === 'male' && (
                    <svg
                      className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-5 h-5 text-white pointer-events-none"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                      strokeWidth={3}
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                    </svg>
                  )}
                </div>
                <span className="text-dark-blue text-2xl font-medium">Male</span>
              </label>
            </div>
          </div>

          {/* Height Input */}
          <div>
            <label className="block text-dark-blue mb-5 text-2xl font-semibold">Height</label>
            <div className="flex gap-4">
              <input
                type="text"
                inputMode="none"
                readOnly
                value={height}
                onClick={() => {
                  setTempHeight(height)
                  setShowNumberPad(true)
                }}
                placeholder="Tap to enter"
                className="flex-1 px-6 py-5 border-2 border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-light-blue focus:border-transparent cursor-pointer text-2xl touch-manipulation"
              />
              <select
                value={heightUnit}
                onChange={(e) => onHeightUnitChange(e.target.value as 'cm' | 'in')}
                className="px-6 py-5 border-2 border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-light-blue focus:border-transparent bg-white text-2xl touch-manipulation w-[120px]"
                style={{ minWidth: '120px' }}
              >
                <option value="cm">cm</option>
                <option value="in">in</option>
              </select>
            </div>
          </div>

          {/* Capture Button */}
          <div className="mt-4">
            <button
              onClick={handleCaptureClick}
              disabled={busy || !sex}
              className={`w-full px-6 py-5 rounded-xl transition-colors flex items-center justify-center gap-3 shadow-lg touch-manipulation ${
                busy || !sex
                  ? 'bg-gray-400 cursor-not-allowed text-white'
                  : 'bg-light-blue hover:bg-dark-blue text-white'
              }`}
              aria-label="Capture image"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-8 w-8"
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
              <span className="text-3xl font-bold">Capture Image</span>
            </button>
            {showSexError && !sex && (
              <p className="text-base text-red-500 font-medium mt-3 text-center">Please select a sex</p>
            )}
            {lastCapture && (
              <div className="text-base text-dark-blue text-center mt-3">
                <p className="font-semibold text-green-600">✓ Captured successfully!</p>
                <p className="text-sm mt-1">
                  RGB: {lastCapture.rgb_path?.split(/[/\\]/).pop()}
                </p>
                <p className="text-sm">
                  Depth: {lastCapture.depth_path?.split(/[/\\]/).pop()}
                </p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Number Pad Modal */}
      {showNumberPad && (
        <NumberPad
          value={tempHeight}
          unit={heightUnit}
          onInput={setTempHeight}
          onClose={() => {
            setShowNumberPad(false)
            setTempHeight(height)
          }}
          onConfirm={() => {
            onHeightChange(tempHeight)
            setShowNumberPad(false)
          }}
        />
      )}
    </div>
  )
}
