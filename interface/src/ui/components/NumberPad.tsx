import React from 'react'

interface NumberPadProps {
  value: string
  unit: 'cm' | 'in' | 'lbs' | 'ft'
  title?: string
  displayValue?: string
  displayNode?: React.ReactNode
  showUnit?: boolean
  onInput: (value: string) => void
  onClose: () => void
  onConfirm: () => void
}

export const NumberPad: React.FC<NumberPadProps> = ({
  value,
  unit,
  title,
  displayValue,
  displayNode,
  showUnit = true,
  onInput,
  onClose,
  onConfirm,
}) => {
  const handleNumberClick = (num: string) => {
    onInput(value + num)
  }

  const handleBackspace = () => {
    onInput(value.slice(0, -1))
  }

  const handleClear = () => {
    onInput('')
  }

  const handleDecimal = () => {
    if (!value.includes('.')) {
      onInput(value + '.')
    }
  }

  return (
    <div 
      className="fixed inset-0 bg-black bg-opacity-50 flex items-end justify-center z-50"
      onClick={onClose}
    >
      <div 
        className="bg-white w-full max-w-2xl rounded-t-2xl shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <button
            onClick={onClose}
            className="text-dark-blue hover:opacity-80 font-semibold text-lg px-4 py-2 touch-manipulation"
          >
            Cancel
          </button>
          <span className="text-xl font-semibold text-dark-blue">{title || `Enter ${unit === 'lbs' ? 'Weight' : 'Height'}`}</span>
          <button
            onClick={onConfirm}
            className="text-light-blue hover:opacity-80 font-semibold text-lg px-4 py-2 touch-manipulation"
          >
            Done
          </button>
        </div>

        {/* Display */}
        <div className="p-8 bg-gray-50">
          <div className="text-5xl font-bold text-dark-blue text-center py-6 bg-white rounded-lg border-2 border-light-blue flex items-center justify-center gap-3 relative">
            <span>{displayNode ?? displayValue ?? (value || '0')}</span>
            {showUnit && (
              <span className="text-3xl text-dark-blue font-normal opacity-60">{unit}</span>
            )}
            {value && (
              <button
                onClick={handleClear}
                className="absolute right-6 top-1/2 transform -translate-y-1/2 text-dark-blue hover:opacity-80 transition-opacity touch-manipulation p-2"
                aria-label="Clear"
              >
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
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              </button>
            )}
          </div>
        </div>

        {/* Number Pad */}
        <div className="p-6 pb-10">
          <div className="grid grid-cols-3 gap-4">
            {/* Row 1 */}
            <button
              onClick={() => handleNumberClick('1')}
              className="h-20 bg-gray-100 hover:bg-gray-200 active:bg-gray-300 rounded-xl text-3xl font-semibold text-dark-blue transition-colors touch-manipulation"
            >
              1
            </button>
            <button
              onClick={() => handleNumberClick('2')}
              className="h-20 bg-gray-100 hover:bg-gray-200 active:bg-gray-300 rounded-xl text-3xl font-semibold text-dark-blue transition-colors touch-manipulation"
            >
              2
            </button>
            <button
              onClick={() => handleNumberClick('3')}
              className="h-20 bg-gray-100 hover:bg-gray-200 active:bg-gray-300 rounded-xl text-3xl font-semibold text-dark-blue transition-colors touch-manipulation"
            >
              3
            </button>

            {/* Row 2 */}
            <button
              onClick={() => handleNumberClick('4')}
              className="h-20 bg-gray-100 hover:bg-gray-200 active:bg-gray-300 rounded-xl text-3xl font-semibold text-dark-blue transition-colors touch-manipulation"
            >
              4
            </button>
            <button
              onClick={() => handleNumberClick('5')}
              className="h-20 bg-gray-100 hover:bg-gray-200 active:bg-gray-300 rounded-xl text-3xl font-semibold text-dark-blue transition-colors touch-manipulation"
            >
              5
            </button>
            <button
              onClick={() => handleNumberClick('6')}
              className="h-20 bg-gray-100 hover:bg-gray-200 active:bg-gray-300 rounded-xl text-3xl font-semibold text-dark-blue transition-colors touch-manipulation"
            >
              6
            </button>

            {/* Row 3 */}
            <button
              onClick={() => handleNumberClick('7')}
              className="h-20 bg-gray-100 hover:bg-gray-200 active:bg-gray-300 rounded-xl text-3xl font-semibold text-dark-blue transition-colors touch-manipulation"
            >
              7
            </button>
            <button
              onClick={() => handleNumberClick('8')}
              className="h-20 bg-gray-100 hover:bg-gray-200 active:bg-gray-300 rounded-xl text-3xl font-semibold text-dark-blue transition-colors touch-manipulation"
            >
              8
            </button>
            <button
              onClick={() => handleNumberClick('9')}
              className="h-20 bg-gray-100 hover:bg-gray-200 active:bg-gray-300 rounded-xl text-3xl font-semibold text-dark-blue transition-colors touch-manipulation"
            >
              9
            </button>

            {/* Row 4 */}
            <button
              onClick={handleDecimal}
              className="h-20 bg-gray-100 hover:bg-gray-200 active:bg-gray-300 rounded-xl text-3xl font-semibold text-dark-blue transition-colors touch-manipulation"
            >
              .
            </button>
            <button
              onClick={() => handleNumberClick('0')}
              className="h-20 bg-gray-100 hover:bg-gray-200 active:bg-gray-300 rounded-xl text-3xl font-semibold text-dark-blue transition-colors touch-manipulation"
            >
              0
            </button>
            <button
              onClick={handleBackspace}
              className="h-20 bg-red-100 hover:bg-red-200 active:bg-red-300 rounded-xl text-2xl font-semibold text-red-700 transition-colors touch-manipulation flex items-center justify-center"
            >
              ⌫
            </button>
          </div>

          {/* Done button */}
          <button
            onClick={onConfirm}
            className="w-full h-16 mt-4 bg-light-blue hover:bg-dark-blue active:bg-dark-blue text-white rounded-xl text-xl font-semibold transition-colors touch-manipulation"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  )
}
