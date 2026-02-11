import React, { useState } from 'react'

interface WeightOutputViewProps {
  weight: number
  onTakeAnotherPhoto: () => void
}

export const WeightOutputView: React.FC<WeightOutputViewProps> = ({
  weight,
  onTakeAnotherPhoto,
}) => {
  const [unit, setUnit] = useState<'kg' | 'lbs'>('kg')
  
  // Ensure weight is valid
  const weightInKg = weight && weight > 0 ? weight : 0
  const weightInLbs = weightInKg * 2.20462
  const errorMargin = 5 // kg error margin
  
  const displayWeight = unit === 'kg' ? weightInKg : weightInLbs
  const displayError = unit === 'kg' ? errorMargin : errorMargin * 2.20462
  const displayUnit = unit

  return (
    <div className="flex-1 flex flex-col items-center justify-center px-4 sm:px-6 lg:px-8 py-6 sm:py-8 bg-white">
      <div className="w-full max-w-7xl">
        {/* Title */}
        <h1 className="text-2xl sm:text-3xl lg:text-4xl font-bold text-black text-center mb-6 sm:mb-8 lg:mb-10">
          Total Body Weight Results
        </h1>

        {/* Weight Display Section */}
        <div className="flex flex-col items-center justify-center mb-6 sm:mb-8">
          {/* Weight Display Card with Unit Selector */}
          <div className="flex gap-4 items-center justify-center w-full max-w-2xl">
            {/* Weight Display Card */}
            <div className="flex-1">
              <div className="bg-white border-4 border-light-blue rounded-3xl p-6 sm:p-8 lg:p-10 xl:p-12 shadow-lg">
                {/* Weight Label */}
                <div className="text-sm sm:text-base lg:text-lg text-black mb-3 sm:mb-4 font-medium text-center">
                  Weight*
                </div>
                
                {/* Weight Value */}
                <div className="flex items-baseline justify-center gap-2 sm:gap-3 lg:gap-4">
                  <span className="text-4xl sm:text-5xl lg:text-6xl xl:text-7xl font-bold text-black">
                    {displayWeight.toFixed(0)}
                  </span>
                  <span className="text-xl sm:text-2xl lg:text-3xl text-black font-semibold">±</span>
                  <span className="text-4xl sm:text-5xl lg:text-6xl xl:text-7xl font-bold text-black">
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
                className="px-6 py-5 border-2 border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-light-blue focus:border-transparent bg-white text-2xl touch-manipulation w-[120px]"
                style={{ minWidth: '120px' }}
              >
                <option value="kg">kg</option>
                <option value="lbs">lbs</option>
              </select>
            </div>
          </div>

          {/* Take Another Photo Button - On next row, centered */}
          <div className="w-full flex justify-center mt-6 sm:mt-8">
            <button
              onClick={onTakeAnotherPhoto}
              className="px-8 sm:px-10 lg:px-12 py-3 sm:py-4 lg:py-5 bg-dark-blue text-white rounded-xl sm:rounded-2xl hover:bg-opacity-90 transition-all duration-200 font-semibold text-base sm:text-lg lg:text-xl shadow-lg hover:shadow-xl transform hover:scale-105 active:scale-100"
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
