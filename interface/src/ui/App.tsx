import React, { useState, useMemo, useEffect, useRef } from 'react'
import { Header } from './components/Header'
import { ImagingView } from './components/ImagingView'
import { ProgressIndicator } from './components/ProgressIndicator'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

console.log('API_BASE:', API_BASE)
console.log('Environment:', import.meta.env)

// Expose test function to window for manual testing
if (typeof window !== 'undefined') {
  (window as any).testRealsenseCapture = async () => {
    const url = `${API_BASE}/realsense_capture/image`
    console.log('Manual test - Fetching:', url)
    try {
      const res = await fetch(url, { 
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        mode: 'cors'
      })
      console.log('Manual test - Response:', res)
      const json = await res.json()
      console.log('Manual test - JSON:', json)
      return json
    } catch (err) {
      console.error('Manual test - Error:', err)
      throw err
    }
  }
  console.log('Test function available: window.testRealsenseCapture()')
}

export const App: React.FC = () => {
  const [currentStep, setCurrentStep] = useState<'calibration' | 'imaging' | 'data-processing'>('imaging')
  const [sex, setSex] = useState<'female' | 'male' | ''>('')

  const handleSexChange = (newSex: 'female' | 'male' | '') => {
    setSex(newSex)
  }
  const [height, setHeight] = useState('')
  const [heightUnit, setHeightUnit] = useState<'cm' | 'in'>('cm')
  const [busy, setBusy] = useState(false)
  const [captureMessage, setCaptureMessage] = useState<string>('')
  const [processing, setProcessing] = useState(false)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [logs, setLogs] = useState<Array<{timestamp: string, type: string, message: string}>>([])
  const [weight, setWeight] = useState<string | null>(null)

  const streamUrl = useMemo(() => {
    const url = `${API_BASE}/realsense_stream/rgb`
    console.log('Stream URL:', url)
    return url
  }, [])

  // Test backend connection on mount
  useEffect(() => {
    console.log('Testing backend connection...')
    const healthUrl = `${API_BASE}/health`
    console.log('Health check URL:', healthUrl)
    
    // Test if fetch is available
    console.log('Fetch available:', typeof fetch !== 'undefined')
    
    const testFetch = async () => {
      try {
        console.log('Starting health check fetch...')
        const controller = new AbortController()
        const timeoutId = setTimeout(() => controller.abort(), 5000)
        
        const res = await fetch(healthUrl, { 
          mode: 'cors',
          signal: controller.signal
        })
        clearTimeout(timeoutId)
        
        console.log('Health check response status:', res.status)
        const data = await res.json()
        console.log('Backend health check success:', data)
      } catch (err) {
        console.error('Backend connection failed:', err)
        console.error('Error details:', err)
        if (err instanceof Error) {
          console.error('Error name:', err.name)
          console.error('Error message:', err.message)
        }
      }
    }
    
    testFetch()
  }, [])

  const handleCapture = async () => {
    console.log('Capture button clicked')
    console.log('API_BASE:', API_BASE)
    setBusy(true)
    setCaptureMessage('')
    
    const url = `${API_BASE}/realsense_capture/image`
    console.log('Fetching URL:', url)
    console.log('About to call fetch...')
    
    // Prepare request body with sex and height
    const requestBody = {
      sex: sex || null,
      height: height || null
    }
    console.log('Request body:', requestBody)
    
    try {
      console.log('Calling fetch now...')
      const fetchPromise = fetch(url, { 
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        mode: 'cors',
        body: JSON.stringify(requestBody)
      })
      console.log('Fetch promise created:', fetchPromise)
      
      const res = await fetchPromise
      console.log('Response received:', res)
      console.log('Response status:', res.status)
      console.log('Response headers:', res.headers)
      
      if (!res.ok) {
        const errorText = await res.text()
        console.error('Response error:', errorText)
        setCaptureMessage(`Error: ${res.status} ${res.statusText}`)
        setBusy(false)
        return
      }
      
      const json = await res.json()
      console.log('Response JSON:', json)
      
      if (json.success) {
        setCaptureMessage(`Image captured! Starting 3D processing...`)
        setProcessing(true)
        setSessionId(json.session_id)
        setLogs([])
        setWeight(null)
        
        // Start listening to logs if session_id is provided
        if (json.session_id) {
          startLogStream(json.session_id)
        }
      } else {
        setCaptureMessage(`Error: ${json.message || 'Failed to capture image'}`)
        setBusy(false)
      }
    } catch (error) {
      console.error('Fetch error caught:', error)
      console.error('Error type:', typeof error)
      console.error('Error details:', {
        name: error instanceof Error ? error.name : 'unknown',
        message: error instanceof Error ? error.message : String(error),
        stack: error instanceof Error ? error.stack : 'no stack'
      })
      setCaptureMessage(`Error: ${error instanceof Error ? error.message : 'Failed to capture image'}`)
    } finally {
      console.log('Finally block - setting busy to false')
      // Don't set busy to false here - let the log stream handle it
    }
  }

  const startLogStream = (sessionId: string) => {
    console.log(`[LOG STREAM] Starting log stream for session: ${sessionId}`)
    // Close existing connection if any
    if (eventSourceRef.current) {
      console.log('[LOG STREAM] Closing existing EventSource')
      eventSourceRef.current.close()
    }
    
    const logUrl = `${API_BASE}/realsense_capture/logs/${sessionId}`
    console.log(`[LOG STREAM] Connecting to: ${logUrl}`)
    const eventSource = new EventSource(logUrl)
    eventSourceRef.current = eventSource
    
    eventSource.onopen = () => {
      console.log('[LOG STREAM] EventSource connection opened')
    }
    
    eventSource.onmessage = (event) => {
      console.log('[LOG STREAM] Received message:', event.data)
      try {
        const logEntry = JSON.parse(event.data)
        
        if (logEntry.type === 'end') {
          console.log('[LOG STREAM] End signal received, closing connection')
          eventSource.close()
          eventSourceRef.current = null
          setProcessing(false)
          setBusy(false)
          return
        }
        
        // Add log entry
        setLogs(prev => [...prev, logEntry])
        
        // Extract weight from log messages
        const message = logEntry.message || ''
        // Look for weight patterns from weight_formula output
        // Pattern: "The estimated weight is: 70.50 kg or 155.43 lbs"
        const weightMatch = message.match(/estimated weight is:\s*(\d+\.?\d*)\s*kg/i)
        if (weightMatch) {
          setWeight(`${weightMatch[1]} kg`)
        }
        
        // Check for success completion
        if (logEntry.type === 'success') {
          setProcessing(false)
          setBusy(false)
        }
        
        // Check for errors
        if (logEntry.type === 'error') {
          // Don't stop processing on errors, just log them
        }
      } catch (err) {
        console.error('Error parsing log entry:', err)
      }
    }
    
    eventSource.onerror = (error) => {
      console.error('[LOG STREAM] EventSource error:', error)
      console.error('[LOG STREAM] EventSource readyState:', eventSource.readyState)
      // readyState: 0 = CONNECTING, 1 = OPEN, 2 = CLOSED
      if (eventSource.readyState === EventSource.CLOSED) {
        console.error('[LOG STREAM] Connection closed unexpectedly')
      }
      eventSource.close()
      eventSourceRef.current = null
      setProcessing(false)
      setBusy(false)
    }
  }
  
  // Cleanup effect
  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close()
        eventSourceRef.current = null
      }
    }
  }, [])

  return (
    <div className="min-h-screen bg-white flex flex-col">
      <Header />
      <div className="flex-1 flex flex-col items-center justify-center px-8 py-12">
        <ImagingView
          sex={sex}
          height={height}
          heightUnit={heightUnit}
          streamUrl={streamUrl}
          busy={busy}
          processing={processing}
          captureMessage={captureMessage}
          logs={logs}
          weight={weight}
          onSexChange={handleSexChange}
          onHeightChange={setHeight}
          onHeightUnitChange={setHeightUnit}
          onCapture={handleCapture}
        />
      </div>
      <ProgressIndicator currentStep={currentStep} />
    </div>
  )
}
