import React, { useState } from 'react'
import { NumberPad } from './NumberPad'

interface DataCollectionViewProps {
  weight: string
  raceEthnicity: string
  activityLevel: string
  onWeightChange: (weight: string) => void
  onRaceEthnicityChange: (race: string) => void
  onActivityLevelChange: (level: string) => void
  onContinue: () => void
}

const RACE_ETHNICITY_OPTIONS = [
  'American Indian or Alaska Native',
  'Asian',
  'Black or African American',
  'Hispanic or Latino',
  'Native Hawaiian or Other Pacific Islander',
  'White',
  'Other',
  'Prefer not to answer',
]

const ACTIVITY_LEVEL_OPTIONS = [
  'Sedentary (little or no exercise)',
  'Lightly active (light exercise 1-3 days/week)',
  'Moderately active (moderate exercise 3-5 days/week)',
  'Very active (hard exercise 6-7 days/week)',
]

export const DataCollectionView: React.FC<DataCollectionViewProps> = ({
  weight,
  raceEthnicity,
  activityLevel,
  onWeightChange,
  onRaceEthnicityChange,
  onActivityLevelChange,
  onContinue,
}) => {
  const [showNumberPad, setShowNumberPad] = useState(false)
  const [tempWeight, setTempWeight] = useState(weight)
  const [errors, setErrors] = useState<{ weight?: boolean; race?: boolean; activity?: boolean }>({})

  const handleContinue = () => {
    const newErrors: { weight?: boolean; race?: boolean; activity?: boolean } = {}
    
    if (!weight) {
      newErrors.weight = true
    }
    if (!raceEthnicity) {
      newErrors.race = true
    }
    if (!activityLevel) {
      newErrors.activity = true
    }

    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors)
      setTimeout(() => setErrors({}), 3000)
      return
    }

    setErrors({})
    onContinue()
  }

  return (
    <div className="min-h-screen bg-dark-blue flex flex-col items-center justify-center px-8 py-12">
      <div className="w-full max-w-4xl">
        {/* Title */}
        <h1 className="text-white text-4xl font-bold text-center mb-12">
          Data Collection
        </h1>

        <div className="space-y-10">
          {/* Weight Input */}
          <div>
            <label className="block text-white mb-5 text-2xl font-semibold">
              Weight <span className="text-red-300 ml-3">required</span>
            </label>
            <div className="flex gap-4">
              <input
                type="text"
                inputMode="none"
                readOnly
                value={weight}
                onClick={() => {
                  setTempWeight(weight)
                  setShowNumberPad(true)
                }}
                placeholder="Tap to enter"
                className={`flex-1 px-6 py-5 border-2 rounded-lg focus:outline-none focus:ring-2 focus:ring-light-blue focus:border-transparent cursor-pointer text-2xl touch-manipulation bg-white text-dark-blue ${
                  errors.weight ? 'border-red-400' : 'border-gray-300'
                }`}
              />
              <div className="px-6 py-5 bg-white border-2 border-gray-300 rounded-lg text-2xl text-dark-blue font-semibold flex items-center">
                lbs
              </div>
            </div>
            {errors.weight && (
              <p className="text-red-300 text-base font-medium mt-2">Please enter weight</p>
            )}
          </div>

          {/* Race/Ethnicity Input */}
          <div>
            <label className="block text-white mb-5 text-2xl font-semibold">
              Race/Ethnicity <span className="text-red-300 ml-3">required</span>
            </label>
            <select
              value={raceEthnicity}
              onChange={(e) => {
                onRaceEthnicityChange(e.target.value)
                if (errors.race) {
                  setErrors({ ...errors, race: false })
                }
              }}
              className={`w-full px-6 py-5 border-2 rounded-lg focus:outline-none focus:ring-2 focus:ring-light-blue focus:border-transparent bg-white text-2xl touch-manipulation text-dark-blue ${
                errors.race ? 'border-red-400' : 'border-gray-300'
              }`}
            >
              <option value="">Select race/ethnicity</option>
              {RACE_ETHNICITY_OPTIONS.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
            {errors.race && (
              <p className="text-red-300 text-base font-medium mt-2">Please select race/ethnicity</p>
            )}
          </div>

          {/* Activity Level Input */}
          <div>
            <label className="block text-white mb-5 text-2xl font-semibold">
              Activity Level <span className="text-red-300 ml-3">required</span>
            </label>
            <select
              value={activityLevel}
              onChange={(e) => {
                onActivityLevelChange(e.target.value)
                if (errors.activity) {
                  setErrors({ ...errors, activity: false })
                }
              }}
              className={`w-full px-6 py-5 border-2 rounded-lg focus:outline-none focus:ring-2 focus:ring-light-blue focus:border-transparent bg-white text-2xl touch-manipulation text-dark-blue ${
                errors.activity ? 'border-red-400' : 'border-gray-300'
              }`}
            >
              <option value="">Select activity level</option>
              {ACTIVITY_LEVEL_OPTIONS.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
            {errors.activity && (
              <p className="text-red-300 text-base font-medium mt-2">Please select activity level</p>
            )}
          </div>

          {/* Continue Button */}
          <div className="mt-8">
            <button
              onClick={handleContinue}
              className="w-full px-6 py-5 rounded-xl transition-colors flex items-center justify-center gap-3 shadow-lg touch-manipulation bg-light-blue hover:bg-accent-blue text-white"
              aria-label="Continue to imaging"
            >
              <span className="text-3xl font-bold">Continue</span>
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-8 w-8"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M13 7l5 5m0 0l-5 5m5-5H6"
                />
              </svg>
            </button>
          </div>
        </div>
      </div>

      {/* Number Pad Modal for Weight */}
      {showNumberPad && (
        <NumberPad
          value={tempWeight}
          unit="lbs"
          title="Enter Weight"
          onInput={setTempWeight}
          onClose={() => {
            setShowNumberPad(false)
            setTempWeight(weight)
          }}
          onConfirm={() => {
            onWeightChange(tempWeight)
            setShowNumberPad(false)
            if (errors.weight) {
              setErrors({ ...errors, weight: false })
            }
          }}
        />
      )}
    </div>
  )
}
