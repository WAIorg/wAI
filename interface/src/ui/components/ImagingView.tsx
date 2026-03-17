import React, { useState, useEffect } from 'react'
import { NumberPad } from './NumberPad'

interface ImagingViewProps {
  sex: 'female' | 'male' | ''
  height: string
  heightUnit: 'cm' | 'in'
  streamUrl: string
  showStream: boolean
  onTurnStreamOn: () => void
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
  showStream,
  onTurnStreamOn,
  onSexChange,
  onHeightChange,
  onHeightUnitChange,
  onCapture,
  busy,
  lastCapture,
}) => {
  const [showNumberPad, setShowNumberPad] = useState(false)
  const [tempHeight, setTempHeight] = useState(height)
  const [ftInRaw, setFtInRaw] = useState('')
  const [captureAttempted, setCaptureAttempted] = useState(false)
  const [showSexError, setShowSexError] = useState(false)

  const parseFtInRaw = (raw: string) => {
    const digitsOnly = raw.replace(/[^\d]/g, '')
    if (!digitsOnly) return { raw: '', feetStr: '', inchesStr: '' }
    if (digitsOnly.length === 1) {
      return { raw: digitsOnly, feetStr: digitsOnly, inchesStr: '' }
    }
    if (digitsOnly.length === 2) {
      return { raw: digitsOnly, feetStr: digitsOnly.slice(0, 1), inchesStr: digitsOnly.slice(1) }
    }
    return {
      raw: digitsOnly,
      feetStr: digitsOnly.slice(0, -2),
      inchesStr: digitsOnly.slice(-2),
    }
  }

  const normalizeFeetInches = (feetStr: string, inchesStr: string) => {
    const feetVal = parseInt(feetStr || '0', 10) || 0
    const inchesValRaw = parseInt(inchesStr || '0', 10) || 0
    if (feetVal <= 0 && inchesValRaw <= 0) {
      return { feetVal: 0, inchesVal: 0, totalInches: 0 }
    }
    // Keep inches in [0, 11] by carrying overflow into feet.
    const carryFeet = Math.floor(inchesValRaw / 12)
    const inchesVal = inchesValRaw % 12
    const normalizedFeet = feetVal + carryFeet
    const totalInches = normalizedFeet * 12 + inchesVal
    return { feetVal: normalizedFeet, inchesVal, totalInches }
  }

  // Synchronize display raw string from total height when using inches
  useEffect(() => {
    if (heightUnit === 'in') {
      const totalInches = parseFloat(height)
      if (!isNaN(totalInches) && totalInches > 0) {
        let wholeFeet = Math.floor(totalInches / 12)
        let remainingInches = Math.round(totalInches - wholeFeet * 12)
        if (remainingInches >= 12) {
          wholeFeet += 1
          remainingInches -= 12
        }
        const raw = `${wholeFeet}${String(remainingInches).padStart(2, '0')}`
        setFtInRaw(raw)
      } else {
        setFtInRaw('')
      }
    } else {
      setFtInRaw('')
    }
  }, [height, heightUnit])

  const handleNumberPadClose = () => {
    setShowNumberPad(false)
    if (heightUnit === 'in') {
      setTempHeight(ftInRaw)
    } else {
      setTempHeight(height)
    }
  }

  const handleNumberPadConfirm = () => {
    if (heightUnit === 'in') {
      const parsed = parseFtInRaw(tempHeight)
      const normalized = normalizeFeetInches(parsed.feetStr, parsed.inchesStr)
      const rebuiltRaw =
        normalized.totalInches > 0
          ? `${normalized.feetVal}${String(normalized.inchesVal).padStart(2, '0')}`
          : ''
      setFtInRaw(rebuiltRaw)
      onHeightChange(normalized.totalInches > 0 ? String(normalized.totalInches) : '')
    } else {
      onHeightChange(tempHeight)
    }
    setShowNumberPad(false)
  }

  const handleCaptureClick = () => {
    const heightVal = parseFloat(height)
    const hasHeight = !isNaN(heightVal) && heightVal > 0
    const hasSex = Boolean(sex)

    if (!hasSex || !hasHeight) {
      setCaptureAttempted(true)
      setShowSexError(!hasSex)
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

  // If user fixes missing fields after attempting capture, clear the inline warnings.
  useEffect(() => {
    if (!captureAttempted) return
    const heightVal = parseFloat(height)
    const hasHeight = !isNaN(heightVal) && heightVal > 0
    const hasSex = Boolean(sex)
    if (hasSex && hasHeight) {
      setCaptureAttempted(false)
      setShowSexError(false)
    }
  }, [captureAttempted, height, sex])
  return (
    <div className="w-full max-w-6xl">
      <div className="flex gap-24 items-start justify-center w-full max-w-7xl">
        {/* Image Display Area with Overlays */}
        <div className="relative flex-shrink-0">
          {/* Instruction Text for Image */}
          <p className="text-center text-dark-blue mb-6 text-3xl font-bold">
            Please centre the user in the outline
          </p>
          <div className="relative w-[640px] h-[480px] bg-gray-100 rounded-lg overflow-hidden border-2 border-light-blue">
            {/* RealSense Video Stream or Turn stream on button */}
            {showStream ? (
              <img
                src={streamUrl}
                alt="RealSense RGB Stream"
                className="w-full h-full object-cover"
                style={{ imageRendering: 'auto' }}
              />
            ) : (
              <button
                type="button"
                onClick={onTurnStreamOn}
                className="w-full h-full flex flex-col items-center justify-center gap-4 text-dark-blue hover:bg-gray-200 transition-colors"
                aria-label="Turn stream on"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-24 w-24 text-light-blue" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z" />
                </svg>
                <span className="text-2xl font-bold">Turn stream on</span>
              </button>
            )}

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

            {/* Crosshair Overlay - only when stream is visible */}
            {showStream && (
              <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 pointer-events-none">
                <div className="w-8 h-8 relative">
                  <div className="absolute top-1/2 left-0 w-full h-0.5 bg-light-blue transform -translate-y-1/2"></div>
                  <div className="absolute left-1/2 top-0 w-0.5 h-full bg-light-blue transform -translate-x-1/2"></div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Data Input Section */}
        <div className="flex flex-col gap-10 flex-1 max-w-md overflow-visible">
          {/* Instruction Text for Inputs */}
          <p className="text-dark-blue text-4xl font-bold mb-4">
            Input sex and height values
          </p>
          
          {/* Sex Input */}
          <div>
            <label className="block text-dark-blue mb-6 text-3xl font-semibold">
              Sex
              {captureAttempted && !sex && <span className="text-red-500 ml-3">required</span>}
            </label>
            <div className="flex gap-16">
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
                    className="w-10 h-10 border-2 border-gray-300 rounded-xl focus:ring-4 focus:ring-light-blue appearance-none checked:bg-light-blue checked:border-light-blue"
                  />
                  {sex === 'female' && (
                    <svg
                      className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-6 h-6 text-white pointer-events-none"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                      strokeWidth={3}
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                    </svg>
                  )}
                </div>
                <span className="text-dark-blue text-3xl font-medium">Female</span>
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
                    className="w-10 h-10 border-2 border-gray-300 rounded-xl focus:ring-4 focus:ring-light-blue appearance-none checked:bg-light-blue checked:border-light-blue"
                  />
                  {sex === 'male' && (
                    <svg
                      className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-6 h-6 text-white pointer-events-none"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                      strokeWidth={3}
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                    </svg>
                  )}
                </div>
                <span className="text-dark-blue text-3xl font-medium">Male</span>
              </label>
            </div>
          </div>

          {/* Height Input */}
          <div>
            <label className="block text-dark-blue mb-6 text-3xl font-semibold">
              Height
              {(() => {
                const heightVal = parseFloat(height)
                const hasHeight = !isNaN(heightVal) && heightVal > 0
                return captureAttempted && !hasHeight ? <span className="text-red-500 ml-3">required</span> : null
              })()}
            </label>
            <div className="flex gap-5">
              {heightUnit === 'cm' ? (
                <div className="flex-1">
                  <div
                    className={`flex items-center px-7 py-7 border-2 rounded-xl focus-within:ring-4 focus-within:ring-light-blue focus-within:border-transparent bg-white text-3xl touch-manipulation ${
                      captureAttempted && !(parseFloat(height) > 0) ? 'border-red-500' : 'border-gray-300'
                    }`}
                  >
                    <input
                      type="text"
                      inputMode="none"
                      readOnly
                      value={height ? `${height} cm` : ''}
                      onClick={() => {
                        setTempHeight(height)
                        setShowNumberPad(true)
                      }}
                      placeholder="Tap to enter"
                      className="flex-1 min-w-0 outline-none bg-transparent cursor-pointer"
                      aria-label="Height in centimeters"
                    />
                  </div>
                </div>
              ) : (
                <div className="flex-1">
                  <div
                    className={`flex items-center px-7 py-7 border-2 rounded-xl focus-within:ring-4 focus-within:ring-light-blue focus-within:border-transparent bg-white text-3xl touch-manipulation ${
                      captureAttempted && !(parseFloat(height) > 0) ? 'border-red-500' : 'border-gray-300'
                    }`}
                  >
                    <input
                      type="text"
                      inputMode="none"
                      readOnly
                      value={(() => {
                        const { feetStr, inchesStr } = parseFtInRaw(ftInRaw)
                        if (!feetStr && !inchesStr) return ''
                        const normalized = normalizeFeetInches(feetStr, inchesStr)
                        return `${normalized.feetVal} ft ${normalized.inchesVal} in`
                      })()}
                      onClick={() => {
                        setTempHeight(ftInRaw)
                        setShowNumberPad(true)
                      }}
                      placeholder="Tap to enter"
                      className="flex-1 min-w-0 outline-none bg-transparent cursor-pointer"
                      aria-label="Height in feet and inches"
                    />
                  </div>
                </div>
              )}
              <select
                value={heightUnit}
                onChange={(e) => {
                  const nextUnit = e.target.value as 'cm' | 'in'
                  // Clear height when switching units so users always start fresh.
                  setTempHeight('')
                  setFtInRaw('')
                  onHeightChange('')
                  onHeightUnitChange(nextUnit)
                }}
                className="px-7 py-7 border-2 border-gray-300 rounded-xl focus:outline-none focus:ring-4 focus:ring-light-blue focus:border-transparent bg-white text-3xl touch-manipulation w-[150px]"
                style={{ minWidth: '150px' }}
              >
                <option value="cm">cm</option>
                <option value="in">ft</option>
              </select>
            </div>
          </div>

          {/* Capture Button */}
          <div className="mt-6">
            <button
              onClick={handleCaptureClick}
              disabled={busy}
              className={`w-full px-8 py-7 rounded-2xl transition-colors flex items-center justify-center gap-4 shadow-2xl touch-manipulation ${
                busy
                  ? 'bg-gray-400 cursor-not-allowed text-white'
                  : 'bg-light-blue hover:bg-dark-blue text-white'
              }`}
              aria-label="Capture image"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-10 w-10"
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
              <span className="text-4xl font-bold">Capture Image</span>
            </button>
            {lastCapture && (
              <div className="text-lg text-dark-blue text-center mt-4">
                <p className="font-semibold text-green-600 text-xl">✓ Captured successfully!</p>
                <p className="text-base mt-2">
                  RGB: {lastCapture.rgb_path?.split(/[/\\]/).pop()}
                </p>
                <p className="text-base">
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
          unit={
            heightUnit === 'in' ? 'ft' : heightUnit
          }
          title={
            heightUnit === 'in' ? 'Enter height (ft/in)' : undefined
          }
          displayNode={
            heightUnit === 'in'
              ? (() => {
                  const digitsOnly = tempHeight.replace(/[^\d]/g, '')
                  const parsed = parseFtInRaw(digitsOnly)
                  const normalized = normalizeFeetInches(parsed.feetStr, parsed.inchesStr)
                  return (
                    <span className="flex items-baseline gap-3">
                      <span>{normalized.feetVal}</span>
                      <span className="text-3xl text-dark-blue font-normal opacity-60">ft</span>
                      <span>{normalized.inchesVal}</span>
                      <span className="text-3xl text-dark-blue font-normal opacity-60">in</span>
                    </span>
                  )
                })()
              : undefined
          }
          showUnit={heightUnit !== 'in'}
          onInput={setTempHeight}
          onClose={handleNumberPadClose}
          onConfirm={handleNumberPadConfirm}
        />
      )}
    </div>
  )
}
