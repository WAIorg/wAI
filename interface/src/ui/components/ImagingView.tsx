import React from 'react'

interface ImagingViewProps {
  sex: 'female' | 'male' | ''
  height: string
  heightUnit: 'cm' | 'in'
  streamUrl: string
  busy: boolean
  processing: boolean
  captureMessage: string
  logs: Array<{timestamp: string, type: string, message: string}>
  weight: string | null
  onSexChange: (sex: 'female' | 'male' | '') => void
  onHeightChange: (height: string) => void
  onHeightUnitChange: (unit: 'cm' | 'in') => void
  onCapture: () => void
}

export const ImagingView: React.FC<ImagingViewProps> = ({
  sex,
  height,
  heightUnit,
  streamUrl,
  busy,
  processing,
  captureMessage,
  logs,
  weight,
  onSexChange,
  onHeightChange,
  onHeightUnitChange,
  onCapture,
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
            {/* RealSense RGB Stream */}
            <img
              src={streamUrl}
              alt="RealSense RGB Stream"
              className="w-full h-full object-cover"
              onLoad={() => {
                console.log('Stream image loaded successfully')
              }}
              onError={(e) => {
                console.error('Stream image error:', e)
                // Fallback to placeholder if stream fails
                const target = e.currentTarget
                target.style.display = 'none'
                const placeholder = target.nextElementSibling as HTMLElement
                if (placeholder) {
                  placeholder.style.display = 'flex'
                }
              }}
            />
            {/* Fallback placeholder (hidden by default, shown on error) */}
            <div className="w-full h-full bg-gradient-to-br from-blue-50 to-blue-100 flex items-center justify-center absolute inset-0" style={{ display: 'none' }}>
              <div className="text-gray-400 text-center">
                <svg
                  className="w-24 h-24 mx-auto mb-4"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                  />
                </svg>
                <p className="text-sm">Waiting for stream...</p>
              </div>
            </div>

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

      {/* Capture Button and Message */}
      <div className="flex flex-col items-center mt-12 gap-4 w-full max-w-4xl">
        <button
          type="button"
          onClick={(e) => {
            e.preventDefault()
            e.stopPropagation()
            console.log('Button clicked, calling onCapture')
            try {
              onCapture()
              console.log('onCapture called successfully')
            } catch (err) {
              console.error('Error calling onCapture:', err)
            }
          }}
          disabled={busy || processing}
          className={`w-20 h-20 rounded-full transition-colors flex items-center justify-center shadow-lg ${
            busy || processing
              ? 'bg-gray-400 cursor-not-allowed'
              : 'bg-gray-200 hover:bg-gray-300 cursor-pointer active:scale-95'
          }`}
          aria-label="Capture image"
        >
          {processing ? (
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white"></div>
          ) : (
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
          )}
        </button>
        
        {/* Processing Status */}
        {processing && (
          <div className="w-full">
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                <p className="text-blue-800 font-semibold">Processing 3D model...</p>
              </div>
            </div>
          </div>
        )}
        
        {/* Weight Display */}
        {weight && (
          <div className="w-full">
            <div className="bg-green-50 border-2 border-green-500 rounded-lg p-6 text-center">
              <p className="text-sm text-green-700 mb-2">Estimated Weight</p>
              <p className="text-4xl font-bold text-green-800">{weight}</p>
            </div>
          </div>
        )}
        
        {/* Capture Message */}
        {captureMessage && !processing && (
          <p
            className={`text-sm px-4 py-2 rounded ${
              captureMessage.startsWith('Error')
                ? 'text-red-600 bg-red-50'
                : 'text-green-600 bg-green-50'
            }`}
          >
            {captureMessage}
          </p>
        )}
        
        {/* Processing Logs */}
        {logs.length > 0 && (
          <div className="w-full mt-4">
            <div className="bg-gray-900 rounded-lg p-4 max-h-96 overflow-y-auto">
              <div className="text-xs font-mono space-y-1">
                {logs.map((log, idx) => (
                  <div
                    key={idx}
                    className={`${
                      log.type === 'error'
                        ? 'text-red-400'
                        : log.type === 'success'
                        ? 'text-green-400'
                        : log.type === 'weight'
                        ? 'text-yellow-400 font-bold'
                        : 'text-gray-300'
                    }`}
                  >
                    <span className="text-gray-500">[{new Date(log.timestamp).toLocaleTimeString()}]</span>{' '}
                    {log.message}
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
