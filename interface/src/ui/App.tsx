import React, { useState, useMemo, useEffect } from 'react'
import { Header } from './components/Header'
import { ImagingView } from './components/ImagingView'
import { ProgressIndicator } from './components/ProgressIndicator'
import { ProcessingView } from './components/ProcessingView'
import { WeightOutputView } from './components/WeightOutputView'
import { DataCollectionView } from './components/DataCollectionView'
import { SettingsModal } from './components/SettingsModal'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

export const App: React.FC = () => {
  // Check for data collection mode from URL query parameter
  const isDataCollectionMode = useMemo(() => {
    const params = new URLSearchParams(window.location.search)
    return params.get('mode') === 'data-collection'
  }, [])

  const [showDataCollection, setShowDataCollection] = useState(isDataCollectionMode)
  const [currentStep, setCurrentStep] = useState<'imaging' | 'data-processing' | 'weight-output'>('imaging')
  const [sex, setSex] = useState<'female' | 'male' | ''>('')
  const [height, setHeight] = useState('')
  const [heightUnit, setHeightUnit] = useState<'cm' | 'in'>('cm')
  const [busy, setBusy] = useState(false)
  const [lastCapture, setLastCapture] = useState<{ rgb_path?: string; depth_path?: string; timestamp?: string } | null>(null)
  
  // Data collection fields
  const [weight, setWeight] = useState('')
  const [raceEthnicity, setRaceEthnicity] = useState('')
  const [activityLevel, setActivityLevel] = useState('')
  const [notes, setNotes] = useState('')
  
  // Processing state
  const [isProcessing, setIsProcessing] = useState(false)
  const [processingLogs, setProcessingLogs] = useState<string[]>([])
  const [processingProgress, setProcessingProgress] = useState(0)
  const [processingStep, setProcessingStep] = useState('')
  const [processingResult, setProcessingResult] = useState<{
    success: boolean
    volume?: number
    weight?: number
    std_dev_kg?: number
    std_dev_percent?: number
    sex?: string
    height?: number
    error?: string
  } | null>(null)

  // Settings: persist in localStorage
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [developerMode, setDeveloperMode] = useState(() => {
    try {
      const stored = localStorage.getItem('wai_developer_mode')
      return stored === 'true'
    } catch {
      return false
    }
  })
  const [audioCueEnabled, setAudioCueEnabled] = useState(() => {
    try {
      const stored = localStorage.getItem('wai_audio_cue')
      return stored !== 'false' // default on
    } catch {
      return true
    }
  })
  const [streamAutoOn, setStreamAutoOn] = useState(() => {
    try {
      const stored = localStorage.getItem('wai_stream_auto')
      return stored !== 'false' // default on = show stream automatically
    } catch {
      return true
    }
  })
  const [streamOnManual, setStreamOnManual] = useState(false)
  useEffect(() => {
    try {
      localStorage.setItem('wai_developer_mode', String(developerMode))
    } catch {
      // ignore
    }
  }, [developerMode])
  useEffect(() => {
    try {
      localStorage.setItem('wai_audio_cue', String(audioCueEnabled))
    } catch {
      // ignore
    }
  }, [audioCueEnabled])
  useEffect(() => {
    try {
      localStorage.setItem('wai_stream_auto', String(streamAutoOn))
    } catch {
      // ignore
    }
  }, [streamAutoOn])

  const streamUrl = useMemo(() => `${API_BASE}/realsense_stream/rgb`, [])
  const showStream = streamAutoOn || streamOnManual

  const handleCapture = async () => {
    if (busy) return
    
    setBusy(true)
    setProcessingLogs([])
    setProcessingResult(null)
    const captureStartTime = Date.now() / 1000 // Unix timestamp in seconds
    try {
      // Format height with unit, always storing height in cm for consistency
      let heightValue: string | null = null
      if (height) {
        const numericHeight = parseFloat(height)
        if (!isNaN(numericHeight)) {
          if (heightUnit === 'in') {
            const heightCm = numericHeight * 2.54
            heightValue = `${heightCm.toFixed(2)} cm`
          } else {
            heightValue = `${numericHeight} cm`
          }
        }
      }
      
      const res = await fetch(`${API_BASE}/realsense_capture/image`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          height: heightValue,
          sex: sex || null,
          weight: weight || null,
          race_ethnicity: raceEthnicity || null,
          activity_level: activityLevel || null,
          notes: notes || null,
        }),
      })
      const json = await res.json()
      if (json.success) {
        console.log('Image captured:', json)
        if (isDataCollectionMode) {
          // Data collection mode: save only, no processing. Return to first screen and clear values.
          setWeight('')
          setRaceEthnicity('')
          setActivityLevel('')
          setNotes('')
          setLastCapture(null)
          setSex('')
          setHeight('')
          setShowDataCollection(true)
        } else {
          setLastCapture({
            rgb_path: json.rgb_path,
            depth_path: json.depth_path,
            timestamp: json.timestamp,
          })
          // Automatically start processing after capture, passing capture start time
          startProcessing(captureStartTime)
        }
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

  const startProcessing = async (captureStartTime?: number) => {
    setIsProcessing(true)
    setProcessingLogs([])
    setProcessingResult(null)
    setProcessingProgress(0)
    setProcessingStep('')
    setCurrentStep('data-processing') // Update progress indicator
    
    // Extract numeric height in cm if provided
    let heightValue: number | null = null
    if (height) {
      const numericHeight = parseFloat(height)
      if (!isNaN(numericHeight)) {
        heightValue = heightUnit === 'in' ? numericHeight * 2.54 : numericHeight
      }
    }
    
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
          capture_start_time: captureStartTime || null,
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
            
            // Check if this is a progress update
            if (data.startsWith('PROGRESS:')) {
              try {
                const progressData = JSON.parse(data.slice(9)) // Remove 'PROGRESS:' prefix
                setProcessingProgress(progressData.progress || 0)
                setProcessingStep(progressData.step || '')
              } catch (e) {
                console.error('Error parsing progress:', e)
              }
            }
            // Check if this is a result message
            else if (data.startsWith('RESULT:')) {
              try {
                const resultJson = JSON.parse(data.slice(7)) // Remove 'RESULT:' prefix
                setProcessingResult(resultJson)
                setProcessingProgress(100)
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

  // Show data collection view first if in data collection mode
  if (showDataCollection) {
    return (
      <DataCollectionView
        weight={weight}
        raceEthnicity={raceEthnicity}
        activityLevel={activityLevel}
        notes={notes}
        onWeightChange={setWeight}
        onRaceEthnicityChange={setRaceEthnicity}
        onActivityLevelChange={setActivityLevel}
        onNotesChange={setNotes}
        onContinue={() => setShowDataCollection(false)}
      />
    )
  }

  const handleTakeAnotherPhoto = () => {
    setIsProcessing(false)
    setProcessingLogs([])
    setProcessingResult(null)
    setProcessingProgress(0)
    setProcessingStep('')
    setLastCapture(null) // Clear "Captured successfully" so new session starts fresh
    setHeight('') // Clear height input
    setSex('') // Clear sex input
    if (!streamAutoOn) {
      setStreamOnManual(false)
    }
    setCurrentStep('imaging')
  }

  return (
    <div className="min-h-screen bg-white flex flex-col">
      <Header onSettingsClick={() => setSettingsOpen(true)} />
      <SettingsModal
        isOpen={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        developerMode={developerMode}
        onDeveloperModeChange={setDeveloperMode}
        audioCueEnabled={audioCueEnabled}
        onAudioCueChange={setAudioCueEnabled}
        streamAutoOn={streamAutoOn}
        onStreamAutoChange={setStreamAutoOn}
      />
      {isProcessing ? (
        // Processing view (with logs when developer mode is on)
        <ProcessingView
          isProcessing={isProcessing}
          logs={processingLogs}
          result={processingResult}
          progress={processingProgress}
          currentStep={processingStep}
          onClose={handleTakeAnotherPhoto}
          showLogs={developerMode}
        />
      ) : processingResult && processingResult.success ? (
        // Weight output view
        <WeightOutputView
          weight={processingResult.weight || 0}
          stdDevKg={processingResult.std_dev_kg || 0}
          onTakeAnotherPhoto={handleTakeAnotherPhoto}
          audioCueEnabled={audioCueEnabled}
        />
      ) : (
        // Normal imaging view
        <div className="flex-1 flex flex-col items-center justify-center px-8 py-12">
          <ImagingView
            sex={sex}
            height={height}
            heightUnit={heightUnit}
            streamUrl={streamUrl}
            showStream={showStream}
            onTurnStreamOn={() => setStreamOnManual(true)}
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
