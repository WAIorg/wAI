import React, { useState, useEffect } from 'react'

interface WeightOutputViewProps {
  weight: number
  onTakeAnotherPhoto: () => void
  audioCueEnabled?: boolean
}

function playAlertLouder() {
  const audio = new Audio('/alert.mp3')
  const ctx = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)()
  const src = ctx.createMediaElementSource(audio)
  const gainNode = ctx.createGain()
  gainNode.gain.value = 2.0 // 2x louder (adjust 1.5–3.0 as needed)
  src.connect(gainNode)
  gainNode.connect(ctx.destination)
  if (ctx.state === 'suspended') ctx.resume()
  audio.play().catch((e) => console.warn('Could not play alert sound:', e))
}

export const WeightOutputView: React.FC<WeightOutputViewProps> = ({
  weight,
  onTakeAnotherPhoto,
  audioCueEnabled = true,
}) => {
  const [unit, setUnit] = useState<'kg' | 'lbs'>('kg')

  // Play notification when weight result is shown (only if audio cue is enabled)
  useEffect(() => {
    if (audioCueEnabled && weight && weight > 0) {
      playAlertLouder()
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps -- play once when view is shown

  // Ensure weight is valid
  const weightInKg = weight && weight > 0 ? weight : 0
  const weightInLbs = weightInKg * 2.20462
  const errorMargin = 5 // kg error margin
  
  const displayWeight = unit === 'kg' ? weightInKg : weightInLbs
  const displayError = unit === 'kg' ? errorMargin : errorMargin * 2.20462
  const displayUnit = unit

  return (
    <div className="flex-1 flex flex-col items-center justify-center px-4 sm:px-6 lg:px-8 py-8 sm:py-10 bg-white">
      <div className="w-full max-w-7xl">
        {/* Title */}
        <h1 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-black text-center mb-8 sm:mb-10 lg:mb-12">
          Total Body Weight Results
        </h1>

        {/* Weight Display Section */}
        <div className="flex flex-col items-center justify-center mb-8 sm:mb-10">
          {/* Weight Display Card with Unit Selector */}
          <div className="flex gap-6 items-center justify-center w-full max-w-2xl">
            {/* Weight Display Card */}
            <div className="flex-1">
              <div className="bg-white border-4 border-light-blue rounded-3xl p-8 sm:p-10 lg:p-12 xl:p-14 shadow-lg">
                {/* Weight Label */}
                <div className="text-base sm:text-lg lg:text-xl text-black mb-4 sm:mb-5 font-semibold text-center">
                  Weight*
                </div>
                
                {/* Weight Value */}
                <div className="flex items-baseline justify-center gap-3 sm:gap-4 lg:gap-5">
                  <span className="text-5xl sm:text-6xl lg:text-7xl xl:text-8xl font-bold text-black">
                    {displayWeight.toFixed(0)}
                  </span>
                  <span className="text-2xl sm:text-3xl lg:text-4xl text-black font-semibold">±</span>
                  <span className="text-5xl sm:text-6xl lg:text-7xl xl:text-8xl font-bold text-black">
                    {displayError.toFixed(0)}
                  </span>
                </div>
              </div>
            </div>

            {/* Unit Selector - Aligned to midline of weight display box */}
            <div className="flex items-center">
              <select
                value={unit}
                onChange={(e) => setUnit(e.target.value as 'kg' | 'lbs')}
                className="px-7 py-7 border-2 border-gray-300 rounded-2xl focus:outline-none focus:ring-4 focus:ring-light-blue focus:border-transparent bg-white text-3xl touch-manipulation w-[150px]"
                style={{ minWidth: '150px' }}
              >
                <option value="kg">kg</option>
                <option value="lbs">lbs</option>
              </select>
            </div>
          </div>

          {/* Take Another Photo Button - On next row, centered */}
          <div className="w-full flex justify-center mt-8 sm:mt-10">
            <button
              onClick={onTakeAnotherPhoto}
              className="px-10 sm:px-12 lg:px-14 py-5 sm:py-6 lg:py-7 bg-dark-blue text-white rounded-2xl hover:bg-opacity-90 transition-all duration-200 font-semibold text-xl sm:text-2xl lg:text-3xl shadow-lg hover:shadow-xl transform hover:scale-105 active:scale-100"
            >
              Take Another Photo
            </button>
          </div>
        </div>

        {/* Disclaimer Text */}
        <div className="max-w-4xl mx-auto px-4">
          <p className="text-xs sm:text-sm lg:text-base text-gray-600 text-center leading-relaxed">
            * The photo you acquired resulted in the above total body weight. This system can reliably report the total body weight within a possible error range of {errorMargin} kg.
          </p>
        </div>
      </div>
    </div>
  )
}
