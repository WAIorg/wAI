import React, { useMemo, useState } from 'react'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

const LIGHT_BLUE = '#51B6FF'
const DARK_BLUE = '#0C3A67'
const ACCENT_BLUE = '#2F80ED'

export const App: React.FC = () => {
  const streamUrl = useMemo(() => `${API_BASE}/stream/rgb`, [])
  const [name, setName] = useState('')
  const [weight, setWeight] = useState('')
  const [age, setAge] = useState('')
  const [sex, setSex] = useState('')
  const [mediaPath, setMediaPath] = useState<string>('')
  const [busy, setBusy] = useState(false)
  const [participantIndex, setParticipantIndex] = useState(1)

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

  function nextParticipant() {
    // Clear all form fields and media path; the backend will create next folders automatically
    setName('')
    setWeight('')
    setAge('')
    setSex('')
    setMediaPath('')
    setParticipantIndex(i => i + 1)
  }

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: 16, fontFamily: 'Inter, system-ui, Arial, sans-serif', color: DARK_BLUE as any }}>
      <header style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
        {/* Place your logo image at frontend/public/logo.png */}
        <img src="/logo.png" alt="WAI Logo" style={{ height: 56 }} />
        <div>
          <div style={{ fontSize: 24, fontWeight: 800, color: DARK_BLUE as any }}>Kinect Data Collection</div>
          <div style={{ fontSize: 13, color: '#3B628A' }}>Participant #{participantIndex}</div>
        </div>
      </header>
      <div style={{ display: 'flex', gap: 16 }}>
        <div style={{ flex: 1 }}>
          <div style={{ border: `2px solid ${LIGHT_BLUE}`, borderRadius: 12, overflow: 'hidden', width: 640, height: 480, background: '#f7fbff' }}>
            <img src={streamUrl} alt="RGB Stream" style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }} />
          </div>
          <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
            <button onClick={startRecording} disabled={busy} style={{ background: ACCENT_BLUE, color: 'white', border: 0, padding: '10px 14px', borderRadius: 8, cursor: 'pointer' }}>Start Recording</button>
            <button onClick={stopRecording} disabled={busy} style={{ background: DARK_BLUE, color: 'white', border: 0, padding: '10px 14px', borderRadius: 8, cursor: 'pointer' }}>Stop Recording</button>
            <button onClick={captureImage} disabled={busy} style={{ background: LIGHT_BLUE, color: 'white', border: 0, padding: '10px 14px', borderRadius: 8, cursor: 'pointer' }}>Capture Image</button>
          </div>
          <div style={{ marginTop: 8, color: '#555' }}>Saved path: {mediaPath || '—'}</div>
        </div>
        <div style={{ flex: 1 }}>
          <h3 style={{ color: DARK_BLUE as any }}>Participant Metadata</h3>
          <div style={{ background: '#F0F7FF', border: `1px solid ${LIGHT_BLUE}`, borderRadius: 12, padding: 16 }}>
            <div style={{ display: 'grid', gridTemplateColumns: '140px 1fr', gap: 10, alignItems: 'center' }}>
              <label style={{ color: DARK_BLUE as any }}>Name</label>
              <input value={name} onChange={e => setName(e.target.value)} style={{ padding: 10, borderRadius: 8, border: `1px solid ${LIGHT_BLUE}` }} />
              <label style={{ color: DARK_BLUE as any }}>Weight</label>
              <input value={weight} onChange={e => setWeight(e.target.value)} type="number" step="0.1" style={{ padding: 10, borderRadius: 8, border: `1px solid ${LIGHT_BLUE}` }} />
              <label style={{ color: DARK_BLUE as any }}>Age</label>
              <input value={age} onChange={e => setAge(e.target.value)} type="number" style={{ padding: 10, borderRadius: 8, border: `1px solid ${LIGHT_BLUE}` }} />
              <label style={{ color: DARK_BLUE as any }}>Sex</label>
              <select value={sex} onChange={e => setSex(e.target.value)} style={{ padding: 10, borderRadius: 8, border: `1px solid ${LIGHT_BLUE}` }}>
                <option value="">Select</option>
                <option value="male">Male</option>
                <option value="female">Female</option>
                <option value="other">Other</option>
              </select>
              <label style={{ color: DARK_BLUE as any }}>Media Path</label>
              <input value={mediaPath} onChange={e => setMediaPath(e.target.value)} placeholder="autofilled on capture" style={{ padding: 10, borderRadius: 8, border: `1px solid ${LIGHT_BLUE}` }} />
            </div>
            <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
              <button onClick={saveMetadata} disabled={busy || !name || !sex || !mediaPath} style={{ background: ACCENT_BLUE, color: 'white', border: 0, padding: '10px 14px', borderRadius: 8, cursor: 'pointer' }}>Save CSV Row</button>
              <button onClick={nextParticipant} disabled={busy} style={{ background: '#eaf4ff', color: DARK_BLUE, border: `1px solid ${LIGHT_BLUE}`, padding: '10px 14px', borderRadius: 8, cursor: 'pointer' }}>Next Participant</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}


