import React from 'react'

interface ProgressIndicatorProps {
  currentStep: 'imaging' | 'data-processing' | 'weight-output'
}

export const ProgressIndicator: React.FC<ProgressIndicatorProps> = ({ currentStep }) => {
  const steps = [
    { id: 'imaging', label: 'Imaging' },
    { id: 'data-processing', label: 'Data Processing' },
    { id: 'weight-output', label: 'Weight Output' },
  ] as const

  const getStepIndex = () => {
    return steps.findIndex(step => step.id === currentStep)
  }

  const currentIndex = getStepIndex()

  return (
    <div className="w-full px-8 py-8">
      <div className="max-w-4xl mx-auto">
        {/* Step labels with indicators */}
        <div className="flex justify-between items-center relative">
          {/* Progress bar - positioned behind circles */}
          <div className="absolute top-6 left-6 right-6 h-1 bg-gray-200 rounded-full"></div>
          {/* Progress fill */}
          <div 
            className="absolute top-6 left-6 h-1 bg-light-blue rounded-full transition-all duration-300"
            style={{ width: `calc(${(currentIndex / (steps.length - 1)) * 100}% - 3rem)` }}
          ></div>
          
          {steps.map((step, index) => {
            const isActive = currentStep === step.id
            const isCompleted = index < currentIndex
            const isUpcoming = index > currentIndex
            
            return (
              <div key={step.id} className="flex flex-col items-center relative z-10">
                {/* Step circle indicator */}
                <div className="relative mb-3">
                  <div
                    className={`w-12 h-12 rounded-full flex items-center justify-center transition-all duration-300 ${
                      isActive
                        ? 'bg-light-blue border-4 border-light-blue shadow-lg scale-110'
                        : isCompleted
                        ? 'bg-light-blue border-4 border-light-blue'
                        : 'bg-white border-4 border-gray-300'
                    }`}
                  >
                    {isCompleted && (
                      <svg
                        className="w-6 h-6 text-white"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={3}
                          d="M5 13l4 4L19 7"
                        />
                      </svg>
                    )}
                    {isActive && !isCompleted && (
                      <div className="w-3 h-3 bg-white rounded-full"></div>
                    )}
                  </div>
                </div>
                {/* Step label */}
                <span
                  className={`text-2xl font-medium ${
                    isActive
                      ? 'text-dark-blue font-bold'
                      : isCompleted
                      ? 'text-light-blue'
                      : 'text-gray-400'
                  }`}
                >
                  {step.label}
                </span>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
