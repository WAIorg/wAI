import React, { useMemo, useRef, useState } from 'react'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

export const App: React.FC = () => {
  const streamUrl = useMemo(() => `${API_BASE}/stream/rgb`, [])
  const [name, setName] = useState('')
  const [weight, setWeight] = useState('')
  const [age, setAge] = useState('')
  const [sex, setSex] = useState('')
  const [mediaPath, setMediaPath] = useState<string>('')
  const [busy, setBusy] = useState(false)

  async function startRecording() {
    setBusy(true)
    try {
      const res = await fetch(`${API_BASE}/capture/record/start`, { method: 'POST' })
      const json = await res.json()
      setMediaPath(json.take_dir || '')
    } finally {
      setBusy(false)
    }
  }

  async function stopRecording() {
    setBusy(true)
    try {
      const res = await fetch(`${API_BASE}/capture/record/stop`, { method: 'POST' })
      const json = await res.json()
      setMediaPath(json.take_dir || '')
    } finally {
      setBusy(false)
    }
  }

  async function captureImage() {
    setBusy(true)
    try {
      const res = await fetch(`${API_BASE}/capture/image`, { method: 'POST' })
      const json = await res.json()
      setMediaPath(json.img_dir || '')
    } finally {
      setBusy(false)
    }
  }

  async function saveMetadata() {
    setBusy(true)
    try {
      await fetch(`${API_BASE}/metadata`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name,
          weight: Number(weight),
          age: Number(age),
          sex,
          media_path: mediaPath,
        }),
      })
      alert('Saved metadata')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div style={{ maxWidth: 960, margin: '0 auto', padding: 16, fontFamily: 'sans-serif' }}>
      <h1>Kinect Data Collection</h1>
      <div style={{ display: 'flex', gap: 16 }}>
        <div style={{ flex: 1 }}>
          <div style={{ border: '1px solid #ccc', borderRadius: 8, overflow: 'hidden', width: 640, height: 480 }}>
            <img src={streamUrl} alt="RGB Stream" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
          </div>
          <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
            <button onClick={startRecording} disabled={busy}>Start Recording</button>
            <button onClick={stopRecording} disabled={busy}>Stop Recording</button>
            <button onClick={captureImage} disabled={busy}>Capture Image</button>
          </div>
          <div style={{ marginTop: 8, color: '#555' }}>Saved path: {mediaPath || '—'}</div>
        </div>
        <div style={{ flex: 1 }}>
          <h3>Participant Metadata</h3>
          <div style={{ display: 'grid', gridTemplateColumns: '120px 1fr', gap: 8, alignItems: 'center' }}>
            <label>Name</label>
            <input value={name} onChange={e => setName(e.target.value)} />
            <label>Weight</label>
            <input value={weight} onChange={e => setWeight(e.target.value)} type="number" step="0.1" />
            <label>Age</label>
            <input value={age} onChange={e => setAge(e.target.value)} type="number" />
            <label>Sex</label>
            <select value={sex} onChange={e => setSex(e.target.value)}>
              <option value="">Select</option>
              <option value="male">Male</option>
              <option value="female">Female</option>
              <option value="other">Other</option>
            </select>
            <label>Media Path</label>
            <input value={mediaPath} onChange={e => setMediaPath(e.target.value)} placeholder="autofilled on capture" />
          </div>
          <div style={{ marginTop: 12 }}>
            <button onClick={saveMetadata} disabled={busy || !name || !sex || !mediaPath}>Save CSV Row</button>
          </div>
        </div>
      </div>
    </div>
  )
}


