import React from 'react'

interface TextKeyboardProps {
  value: string
  title?: string
  onInput: (value: string) => void
  onClose: () => void
  onConfirm: () => void
}

const ROWS: string[][] = [
  ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
  ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
  ['Z', 'X', 'C', 'V', 'B', 'N', 'M'],
]

const EXTRA_KEYS: string[] = [',', '.', '?', '!']

export const TextKeyboard: React.FC<TextKeyboardProps> = ({
  value,
  title,
  onInput,
  onClose,
  onConfirm,
}) => {
  const handleKeyClick = (key: string) => {
    onInput(value + key)
  }

  const handleBackspace = () => {
    onInput(value.slice(0, -1))
  }

  const handleClear = () => {
    onInput('')
  }

  const handleSpace = () => {
    onInput(value + ' ')
  }

  return (
    <div
      className="fixed inset-0 bg-black bg-opacity-50 flex items-end justify-center z-50"
      onClick={onClose}
    >
      <div
        className="bg-white w-full max-w-5xl rounded-t-2xl shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-10 py-7 border-b border-gray-200">
          <button
            onClick={onClose}
            className="text-dark-blue hover:opacity-80 font-semibold text-2xl px-5 py-3 touch-manipulation"
          >
            Cancel
          </button>
          <span className="text-3xl font-semibold text-dark-blue">
            {title || 'Enter Notes'}
          </span>
          <button
            onClick={onConfirm}
            className="text-light-blue hover:opacity-80 font-semibold text-2xl px-5 py-3 touch-manipulation"
          >
            Done
          </button>
        </div>

        {/* Display */}
        <div className="p-7 bg-gray-50">
          <div className="min-h-[140px] max-h-60 overflow-y-auto text-3xl text-left py-6 px-6 bg-white rounded-2xl border-2 border-light-blue text-dark-blue relative">
            <span className={value ? '' : 'text-gray-400'}>
              {value || 'Tap keys below to type notes'}
            </span>
            {value && (
              <button
                onClick={handleClear}
                className="absolute right-3 top-3 text-dark-blue hover:opacity-80 transition-opacity touch-manipulation p-1"
                aria-label="Clear text"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-6 w-6"
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

        {/* Keyboard */}
        <div className="p-6 pb-10">
          <div className="space-y-5">
            {ROWS.map((row, rowIndex) => (
              <div
                key={rowIndex}
                className="flex justify-center gap-4"
              >
                {row.map((key) => (
                  <button
                    key={key}
                    onClick={() => handleKeyClick(key.toLowerCase())}
                    className="h-24 min-w-[72px] px-6 bg-gray-100 hover:bg-gray-200 active:bg-gray-300 rounded-2xl text-3xl font-semibold text-dark-blue transition-colors touch-manipulation"
                  >
                    {key}
                  </button>
                ))}
              </div>
            ))}

            {/* Extra keys row */}
            <div className="flex justify-center gap-4">
              {EXTRA_KEYS.map((key) => (
                <button
                  key={key}
                  onClick={() => handleKeyClick(key)}
                  className="h-24 min-w-[72px] px-6 bg-gray-100 hover:bg-gray-200 active:bg-gray-300 rounded-2xl text-3xl font-semibold text-dark-blue transition-colors touch-manipulation"
                >
                  {key}
                </button>
              ))}
              <button
                onClick={handleBackspace}
                className="h-24 px-8 bg-red-100 hover:bg-red-200 active:bg-red-300 rounded-2xl text-3xl font-semibold text-red-700 transition-colors touch-manipulation flex items-center justify-center"
              >
                ⌫
              </button>
            </div>

            {/* Space bar */}
            <div className="flex justify-center mt-2">
              <button
                onClick={handleSpace}
                className="h-24 w-11/12 bg-gray-100 hover:bg-gray-200 active:bg-gray-300 rounded-2xl text-3xl font-semibold text-dark-blue transition-colors touch-manipulation"
              >
                Space
              </button>
            </div>
          </div>

          {/* Done button at bottom */}
          <button
            onClick={onConfirm}
            className="w-full h-20 mt-6 bg-light-blue hover:bg-dark-blue active:bg-dark-blue text-white rounded-2xl text-3xl font-semibold transition-colors touch-manipulation"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  )
}

