import React, { useEffect, useRef } from 'react'

interface ProcessingViewProps {
  isProcessing: boolean
  logs: string[]
  result: {
    success: boolean
    volume?: number
    weight?: number
    sex?: string
    height?: number
    error?: string
  } | null
  progress?: number
  currentStep?: string
  onClose?: () => void
  showLogs?: boolean
}

export const ProcessingView: React.FC<ProcessingViewProps> = ({
  isProcessing,
  logs,
  result,
  progress = 0,
  currentStep = '',
  onClose,
  showLogs = false,
}) => {
  const logsEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    // Auto-scroll to bottom when new logs arrive
    if (logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [logs])

  return (
    <div className="flex-1 flex flex-col items-center justify-center px-8 py-12 bg-white">
      <div className="w-full max-w-6xl">
      {/* Processing Status with Progress Bar */}
      {isProcessing && (
        <div className="bg-blue-50 border border-light-blue rounded-lg p-6 mb-6">
          <div className="mb-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-dark-blue font-semibold text-xl">Processing 3D model...</span>
              <span className="text-dark-blue font-medium text-sm">{Math.round(progress)}%</span>
            </div>
            {/* Progress Bar */}
            <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
              <div 
                className="bg-light-blue h-3 rounded-full transition-all duration-300 ease-out"
                style={{ width: `${progress}%` }}
              ></div>
            </div>
          </div>
          {/* Current Step Message */}
          {currentStep && (
            <div className="mt-3">
              <p className="text-center text-dark-blue font-medium">{currentStep}</p>
            </div>
          )}
          {!currentStep && (
            <p className="text-center text-dark-blue mt-2 opacity-70">This may take a few minutes</p>
          )}
        </div>
      )}

      {/* Results Display - Only show if still processing (for logs view) */}
      {result && result.success && isProcessing && (
        <div className="bg-green-50 border border-green-400 rounded-lg p-6 mb-4">
          <h3 className="text-xl font-bold text-green-800 mb-4">Processing Complete!</h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-sm text-dark-blue">Estimated Weight</p>
              <p className="text-3xl font-bold text-green-700">
                {result.weight?.toFixed(2)} kg
              </p>
              <p className="text-lg text-dark-blue opacity-70">
                ({(result.weight! * 2.20462).toFixed(2)} lbs)
              </p>
            </div>
            <div>
              <p className="text-sm text-dark-blue">Volume</p>
              <p className="text-2xl font-bold text-green-700">
                {result.volume?.toFixed(2)} cm³
              </p>
            </div>
            {result.sex && (
              <div>
                <p className="text-sm text-dark-blue">Sex</p>
                <p className="text-lg font-semibold text-dark-blue capitalize">{result.sex}</p>
              </div>
            )}
            {result.height && (
              <div>
                <p className="text-sm text-dark-blue">Height</p>
                <p className="text-lg font-semibold text-dark-blue">{result.height} cm</p>
              </div>
            )}
          </div>
        </div>
      )}

      {result && !result.success && (
        <div className="bg-red-50 border border-red-400 rounded-lg p-4 mb-4">
          <h3 className="text-lg font-bold text-red-800 mb-2">Processing Failed</h3>
          <p className="text-red-700">{result.error || 'Unknown error occurred'}</p>
        </div>
      )}

      {/* Logs Display - only when developer mode is on */}
      {showLogs && logs.length > 0 && (
        <div className="bg-gray-900 rounded-lg p-6 mb-6 max-h-[500px] overflow-y-auto shadow-lg">
          <div className="text-green-400 font-mono text-sm space-y-1">
            {logs.map((log, index) => (
              <div key={index} className="whitespace-pre-wrap">
                {log}
              </div>
            ))}
            <div ref={logsEndRef} />
          </div>
        </div>
      )}

      {/* Close/Back button when processing is complete */}
      {result && !isProcessing && onClose && (
        <div className="flex justify-center mt-6">
          <button
            onClick={onClose}
            className="px-6 py-3 bg-dark-blue text-white rounded-lg hover:bg-opacity-90 transition-colors font-semibold"
          >
            Back to Imaging
          </button>
        </div>
      )}
    </div>
    </div>
  )
}
