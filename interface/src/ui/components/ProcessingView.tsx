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
}

export const ProcessingView: React.FC<ProcessingViewProps> = ({
  isProcessing,
  logs,
  result,
}) => {
  const logsEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    // Auto-scroll to bottom when new logs arrive
    if (logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [logs])

  if (!isProcessing && !result && logs.length === 0) {
    return null
  }

  return (
    <div className="w-full max-w-4xl mt-8">
      {/* Processing Status */}
      {isProcessing && (
        <div className="bg-blue-50 border border-light-blue rounded-lg p-4 mb-4">
          <div className="flex items-center gap-3">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-light-blue"></div>
            <span className="text-dark-blue font-semibold">Processing 3D model...</span>
          </div>
        </div>
      )}

      {/* Results Display */}
      {result && result.success && (
        <div className="bg-green-50 border border-green-400 rounded-lg p-6 mb-4">
          <h3 className="text-xl font-bold text-green-800 mb-4">Processing Complete!</h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-sm text-gray-600">Estimated Weight</p>
              <p className="text-3xl font-bold text-green-700">
                {result.weight?.toFixed(2)} kg
              </p>
              <p className="text-lg text-gray-500">
                ({(result.weight! * 2.20462).toFixed(2)} lbs)
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Volume</p>
              <p className="text-2xl font-bold text-green-700">
                {result.volume?.toFixed(2)} cm³
              </p>
            </div>
            {result.sex && (
              <div>
                <p className="text-sm text-gray-600">Sex</p>
                <p className="text-lg font-semibold text-gray-700 capitalize">{result.sex}</p>
              </div>
            )}
            {result.height && (
              <div>
                <p className="text-sm text-gray-600">Height</p>
                <p className="text-lg font-semibold text-gray-700">{result.height} cm</p>
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

      {/* Logs Display */}
      {logs.length > 0 && (
        <div className="bg-gray-900 rounded-lg p-4 max-h-96 overflow-y-auto">
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
    </div>
  )
}
