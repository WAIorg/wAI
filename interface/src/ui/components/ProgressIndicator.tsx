import React from 'react'

interface ProgressIndicatorProps {
  currentStep: 'calibration' | 'imaging' | 'data-processing'
}

export const ProgressIndicator: React.FC<ProgressIndicatorProps> = ({ currentStep }) => {
  const steps = [
    { id: 'calibration', label: 'Calibration' },
    { id: 'imaging', label: 'Imaging' },
    { id: 'data-processing', label: 'Data Processing' },
  ] as const

  return (
    <div className="w-full px-8 py-6">
      {/* Dashed line */}
      <div className="border-t-2 border-dashed border-gray-300 mb-4"></div>
      
      {/* Step labels */}
      <div className="flex justify-between items-center">
        {steps.map((step) => {
          const isActive = currentStep === step.id
          return (
            <div key={step.id} className="flex flex-col items-center">
              {/* Vertical bar marker */}
              <div
                className={`w-1 h-8 mb-2 ${
                  isActive ? 'bg-light-blue' : 'bg-gray-200'
                }`}
              ></div>
              {/* Step label */}
              <span
                className={`text-sm ${
                  isActive
                    ? 'text-dark-blue font-semibold underline'
                    : 'text-gray-500'
                }`}
              >
                {step.label}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
