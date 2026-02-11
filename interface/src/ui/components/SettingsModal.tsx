import React from 'react'

interface SettingsModalProps {
  isOpen: boolean
  onClose: () => void
  developerMode: boolean
  onDeveloperModeChange: (value: boolean) => void
}

export const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  developerMode,
  onDeveloperModeChange,
}) => {
  if (!isOpen) return null

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black bg-opacity-50 z-40"
        onClick={onClose}
        aria-hidden="true"
      />
      {/* Modal - Positioned at top right under settings icon */}
      <div
        className="fixed top-20 right-4 z-50"
        role="dialog"
        aria-modal="true"
        aria-labelledby="settings-modal-title"
        onClick={(e) => e.stopPropagation()}
      >
        <div
          className="bg-white rounded-2xl shadow-xl max-w-md w-80 p-6 border-2 border-light-blue"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="flex items-center justify-between mb-6">
            <h2 id="settings-modal-title" className="text-2xl font-bold text-dark-blue">
              Settings
            </h2>
            <button
              onClick={onClose}
              className="text-gray-500 hover:text-dark-blue transition-colors p-1"
              aria-label="Close settings"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Developer mode toggle */}
          <label className="flex items-center gap-4 cursor-pointer select-none">
            <span className="text-lg font-medium text-dark-blue">Developer mode</span>
            <div className="relative flex-shrink-0">
              <input
                type="checkbox"
                checked={developerMode}
                onChange={(e) => onDeveloperModeChange(e.target.checked)}
                className="sr-only peer"
              />
              <div
                className={`w-14 h-8 rounded-full transition-colors ${developerMode ? 'bg-light-blue' : 'bg-gray-300'}`}
              />
              <div
                className={`absolute left-1 top-1 w-6 h-6 bg-white rounded-full shadow transition-transform ${developerMode ? 'translate-x-6' : 'translate-x-0'}`}
              />
            </div>
          </label>
          <p className="text-sm text-gray-600 mt-2">
            When on, shows detailed logs under the loading bar during processing.
          </p>
        </div>
      </div>
    </>
  )
}
