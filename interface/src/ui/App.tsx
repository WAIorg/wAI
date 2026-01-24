import React, { useState, useMemo, useEffect } from 'react'
import { Header } from './components/Header'
import { ImagingView } from './components/ImagingView'
import { ProgressIndicator } from './components/ProgressIndicator'
import { ProcessingView } from './components/ProcessingView'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

export const App: React.FC = () => {
  const [currentStep, setCurrentStep] = useState<'imaging' | 'data-processing' | 'weight-output'>('imaging')
  const [sex, setSex] = useState<'female' | 'male' | ''>('')
  const [height, setHeight] = useState('')
  const [heightUnit, setHeightUnit] = useState<'cm' | 'in'>('cm')
  const [busy, setBusy] = useState(false)
  const [lastCapture, setLastCapture] = useState<{ rgb_path?: string; depth_path?: string; timestamp?: string } | null>(null)
  
  // Processing state
  const [isProcessing, setIsProcessing] = useState(false)
  const [processingLogs, setProcessingLogs] = useState<string[]>([])
  const [processingResult, setProcessingResult] = useState<{
    success: boolean
    volume?: number
    weight?: number
    sex?: string
    height?: number
    error?: string
  } | null>(null)

  const streamUrl = useMemo(() => `${API_BASE}/realsense_stream/rgb`, [])

  const handleCapture = async () => {
    if (busy) return
    
    setBusy(true)
    setProcessingLogs([])
    setProcessingResult(null)
    try {
      // Format height with unit
      const heightValue = height ? `${height} ${heightUnit}` : null
      
      const res = await fetch(`${API_BASE}/realsense_capture/image`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          height: heightValue,
          sex: sex || null,
        }),
      })
      const json = await res.json()
      if (json.success) {
        setLastCapture({
          rgb_path: json.rgb_path,
          depth_path: json.depth_path,
          timestamp: json.timestamp,
        })
        console.log('Image captured:', json)
        
        // Automatically start processing after capture
        startProcessing()
      } else {
        console.error('Capture failed')
        alert('Failed to capture image. Make sure the RealSense camera is connected.')
      }
    } catch (error) {
      console.error('Error capturing image:', error)
      alert('Error capturing image. Check backend connection.')
    } finally {
      setBusy(false)
    }
  }

  const startProcessing = async () => {
    setIsProcessing(true)
    setProcessingLogs([])
    setProcessingResult(null)
    setCurrentStep('data-processing') // Update progress indicator
    
    // Extract numeric height if provided
    const heightValue = height ? parseFloat(height) : null
    
    try {
      const response = await fetch(`${API_BASE}/processing/run/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          use_most_recent: true,
          sex: sex || null,
          height: heightValue,
        }),
      })

      if (!response.ok) {
        throw new Error('Failed to start processing')
      }

      const reader = response.body?.getReader()
      const decoder = new TextDecoder()

      if (!reader) {
        throw new Error('No response body')
      }

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value)
        const lines = chunk.split('\n')

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6) // Remove 'data: ' prefix
            
            // Check if this is a result message
            if (data.startsWith('RESULT:')) {
              try {
                const resultJson = JSON.parse(data.slice(7)) // Remove 'RESULT:' prefix
                setProcessingResult(resultJson)
                setIsProcessing(false)
              } catch (e) {
                console.error('Error parsing result:', e)
              }
            } else {
              // Regular log line
              setProcessingLogs(prev => [...prev, data])
            }
          }
        }
      }
    } catch (error) {
      console.error('Error during processing:', error)
      setProcessingResult({
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      })
      setIsProcessing(false)
      setCurrentStep('imaging') // Return to imaging step on error
    }
  }

  // Update step based on processing state
  useEffect(() => {
    if (processingResult) {
      if (processingResult.success) {
        setCurrentStep('weight-output')
      } else {
        setCurrentStep('imaging')
      }
    }
  }, [processingResult])

  return (
    <div className="min-h-screen bg-white flex flex-col">
      <Header />
      {isProcessing || processingResult ? (
        // Full-screen processing view
        <ProcessingView
          isProcessing={isProcessing}
          logs={processingLogs}
          result={processingResult}
          onClose={() => {
            setIsProcessing(false)
            setProcessingLogs([])
            setProcessingResult(null)
            setCurrentStep('imaging')
          }}
        />
      ) : (
        // Normal imaging view
        <div className="flex-1 flex flex-col items-center justify-center px-8 py-12">
          <ImagingView
            sex={sex}
            height={height}
            heightUnit={heightUnit}
            streamUrl={streamUrl}
            onSexChange={setSex}
            onHeightChange={setHeight}
            onHeightUnitChange={setHeightUnit}
            onCapture={handleCapture}
            busy={busy}
            lastCapture={lastCapture}
          />
        </div>
      )}
      <ProgressIndicator currentStep={currentStep} />
    </div>
  )
}
