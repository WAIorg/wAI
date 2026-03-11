import React from 'react'

interface SettingsModalProps {
  isOpen: boolean
  onClose: () => void
  developerMode: boolean
  onDeveloperModeChange: (value: boolean) => void
  audioCueEnabled: boolean
  onAudioCueChange: (value: boolean) => void
  streamAutoOn: boolean
  onStreamAutoChange: (value: boolean) => void
}

export const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  developerMode,
  onDeveloperModeChange,
  audioCueEnabled,
  onAudioCueChange,
  streamAutoOn,
  onStreamAutoChange,
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
      {/* Modal - Centered, large panel */}
      <div
        className="fixed inset-0 z-50 flex items-center justify-center p-8"
        role="dialog"
        aria-modal="true"
        aria-labelledby="settings-modal-title"
        onClick={(e) => e.stopPropagation()}
      >
        <div
          className="bg-white rounded-3xl shadow-2xl max-w-2xl w-full p-12 border-4 border-light-blue"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="flex items-center justify-between mb-10">
            <h2 id="settings-modal-title" className="text-4xl font-bold text-dark-blue">
              Settings
            </h2>
            <button
              onClick={onClose}
              className="text-gray-500 hover:text-dark-blue transition-colors p-2 rounded-xl hover:bg-gray-100"
              aria-label="Close settings"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          <div className="space-y-10">
            {/* Audio cue toggle */}
            <div>
              <label className="flex items-center gap-6 cursor-pointer select-none">
                <span className="text-2xl font-semibold text-dark-blue">Audio cue</span>
                <div className="relative flex-shrink-0">
                  <input
                    type="checkbox"
                    checked={audioCueEnabled}
                    onChange={(e) => onAudioCueChange(e.target.checked)}
                    className="sr-only peer"
                  />
                  <div
                    className={`w-20 h-11 rounded-full transition-colors ${audioCueEnabled ? 'bg-light-blue' : 'bg-gray-300'}`}
                  />
                  <div
                    className={`absolute left-1.5 top-1.5 w-8 h-8 bg-white rounded-full shadow transition-transform ${audioCueEnabled ? 'translate-x-9' : 'translate-x-0'}`}
                  />
                </div>
              </label>
              <p className="text-xl text-gray-600 mt-3">
                Play a sound when weight results are ready.
              </p>
            </div>

            {/* Show stream automatically toggle */}
            <div>
              <label className="flex items-center gap-6 cursor-pointer select-none">
                <span className="text-2xl font-semibold text-dark-blue">Show stream automatically</span>
                <div className="relative flex-shrink-0">
                  <input
                    type="checkbox"
                    checked={streamAutoOn}
                    onChange={(e) => onStreamAutoChange(e.target.checked)}
                    className="sr-only peer"
                  />
                  <div
                    className={`w-20 h-11 rounded-full transition-colors ${streamAutoOn ? 'bg-light-blue' : 'bg-gray-300'}`}
                  />
                  <div
                    className={`absolute left-1.5 top-1.5 w-8 h-8 bg-white rounded-full shadow transition-transform ${streamAutoOn ? 'translate-x-9' : 'translate-x-0'}`}
                  />
                </div>
              </label>
              <p className="text-xl text-gray-600 mt-3">
                When off, the camera stream area shows a &quot;Turn stream on&quot; button instead of the live feed until you click it.
              </p>
            </div>

            {/* Developer mode toggle */}
            <div>
              <label className="flex items-center gap-6 cursor-pointer select-none">
                <span className="text-2xl font-semibold text-dark-blue">Developer mode</span>
                <div className="relative flex-shrink-0">
                  <input
                    type="checkbox"
                    checked={developerMode}
                    onChange={(e) => onDeveloperModeChange(e.target.checked)}
                    className="sr-only peer"
                  />
                  <div
                    className={`w-20 h-11 rounded-full transition-colors ${developerMode ? 'bg-light-blue' : 'bg-gray-300'}`}
                  />
                  <div
                    className={`absolute left-1.5 top-1.5 w-8 h-8 bg-white rounded-full shadow transition-transform ${developerMode ? 'translate-x-9' : 'translate-x-0'}`}
                  />
                </div>
              </label>
              <p className="text-xl text-gray-600 mt-3">
                When on, shows detailed logs under the loading bar during processing.
              </p>
            </div>

            {/* Refresh and Quit */}
            <div className="flex flex-wrap gap-4">
              <button
                type="button"
                onClick={() => window.location.reload()}
                className="flex items-center gap-3 px-6 py-4 rounded-2xl bg-light-blue text-dark-blue font-semibold text-xl hover:bg-opacity-90 transition-colors"
                aria-label="Refresh page"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                Refresh page
              </button>
              <button
                type="button"
                onClick={() => window.close()}
                className="flex items-center gap-3 px-6 py-4 rounded-2xl bg-gray-200 text-gray-800 font-semibold text-xl hover:bg-gray-300 transition-colors"
                aria-label="Quit and close browser"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
                </svg>
                Quit
              </button>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}
