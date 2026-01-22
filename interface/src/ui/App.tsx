import React, { useState } from 'react'

const LIGHT_BLUE = '#51B6FF'
const DARK_BLUE = '#0C3A67'
const LIGHT_GREY = '#E5E5E5'
const DARK_GREY = '#333333'
const BACKGROUND_BLUE = '#E8F4FD'

export const App: React.FC = () => {
  const [sex, setSex] = useState<'female' | 'male' | null>(null)
  const [height, setHeight] = useState('')

  const handleBack = () => {
    // Handle back navigation
    console.log('Back clicked')
  }

  const handleHelp = () => {
    // Handle help
    console.log('Help clicked')
  }

  const handleCapture = () => {
    // Handle capture action
    console.log('Capture clicked', { sex, height })
  }

  return (
    <div style={{ 
      minHeight: '100vh', 
      backgroundColor: BACKGROUND_BLUE,
      fontFamily: 'Inter, system-ui, Arial, sans-serif',
      display: 'flex',
      flexDirection: 'column'
    }}>
      {/* Header */}
      <header style={{
        backgroundColor: DARK_BLUE,
        padding: '16px 24px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        color: 'white'
      }}>
        <button
          onClick={handleBack}
          style={{
            background: 'none',
            border: 'none',
            color: 'white',
            cursor: 'pointer',
            fontSize: '24px',
            padding: '4px 8px',
            display: 'flex',
            alignItems: 'center'
          }}
        >
          ←
        </button>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '4px',
          fontSize: '24px',
          fontWeight: 600
        }}>
          <span>w</span>
          <span style={{ color: LIGHT_BLUE }}>a</span>
          <span>i</span>
          <span style={{
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            backgroundColor: LIGHT_BLUE,
            marginLeft: '2px'
          }} />
        </div>
        <button
          onClick={handleHelp}
          style={{
            background: 'white',
            border: 'none',
            borderRadius: '50%',
            width: '32px',
            height: '32px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'pointer',
            color: DARK_BLUE,
            fontSize: '18px',
            fontWeight: 'bold'
          }}
        >
          ?
        </button>
      </header>

      {/* Main Content */}
      <main style={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '40px 24px',
        gap: '40px'
      }}>
        {/* Instruction Text */}
        <div style={{
          color: DARK_GREY,
          fontSize: '20px',
          fontWeight: 500,
          textAlign: 'center'
        }}>
          Please centre the user & input values
        </div>

        {/* Content Container */}
        <div style={{
          display: 'flex',
          gap: '60px',
          alignItems: 'center',
          maxWidth: '1200px',
          width: '100%',
          justifyContent: 'center'
        }}>
          {/* Illustration (Left Side) */}
          <div style={{
            flex: 1,
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center'
          }}>
            <div style={{
              width: '400px',
              height: '500px',
              backgroundColor: 'white',
              borderRadius: '16px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
              position: 'relative',
              overflow: 'hidden'
            }}>
              {/* Placeholder for illustration - you can replace this with an actual image */}
              <div style={{
                width: '100%',
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)',
                color: DARK_GREY,
                fontSize: '16px',
                textAlign: 'center',
                padding: '20px'
              }}>
                <div style={{ fontSize: '48px', marginBottom: '16px' }}>🪑</div>
                <div>User Illustration</div>
                <div style={{ fontSize: '12px', marginTop: '8px', opacity: 0.7 }}>
                  (Replace with actual illustration)
                </div>
              </div>
            </div>
          </div>

          {/* Input Fields (Right Side) */}
          <div style={{
            flex: 1,
            display: 'flex',
            flexDirection: 'column',
            gap: '32px',
            maxWidth: '400px'
          }}>
            {/* Sex Field */}
            <div>
              <label style={{
                display: 'block',
                color: DARK_GREY,
                fontSize: '18px',
                fontWeight: 500,
                marginBottom: '12px'
              }}>
                Sex <span style={{ color: '#e74c3c' }}>*</span>
              </label>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                <label style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '12px',
                  cursor: 'pointer',
                  fontSize: '16px',
                  color: DARK_GREY
                }}>
                  <input
                    type="checkbox"
                    checked={sex === 'female'}
                    onChange={() => setSex(sex === 'female' ? null : 'female')}
                    style={{
                      width: '20px',
                      height: '20px',
                      cursor: 'pointer',
                      accentColor: LIGHT_BLUE
                    }}
                  />
                  Female
                </label>
                <label style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '12px',
                  cursor: 'pointer',
                  fontSize: '16px',
                  color: DARK_GREY
                }}>
                  <input
                    type="checkbox"
                    checked={sex === 'male'}
                    onChange={() => setSex(sex === 'male' ? null : 'male')}
                    style={{
                      width: '20px',
                      height: '20px',
                      cursor: 'pointer',
                      accentColor: LIGHT_BLUE
                    }}
                  />
                  Male
                </label>
              </div>
            </div>

            {/* Height Field */}
            <div>
              <label style={{
                display: 'block',
                color: DARK_GREY,
                fontSize: '18px',
                fontWeight: 500,
                marginBottom: '12px'
              }}>
                Height
              </label>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <input
                  type="number"
                  value={height}
                  onChange={(e) => setHeight(e.target.value)}
                  placeholder=""
                  style={{
                    padding: '12px 16px',
                    borderRadius: '8px',
                    border: `1px solid ${LIGHT_BLUE}`,
                    fontSize: '16px',
                    width: '120px',
                    outline: 'none'
                  }}
                />
                <span style={{ color: DARK_GREY, fontSize: '16px' }}>(cm)</span>
              </div>
            </div>
          </div>
        </div>

        {/* Capture Button */}
        <button
          onClick={handleCapture}
          disabled={!sex}
          style={{
            width: '80px',
            height: '80px',
            borderRadius: '50%',
            backgroundColor: sex ? DARK_GREY : LIGHT_GREY,
            border: 'none',
            cursor: sex ? 'pointer' : 'not-allowed',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 4px 12px rgba(0,0,0,0.2)',
            transition: 'all 0.2s',
            opacity: sex ? 1 : 0.6
          }}
          onMouseEnter={(e) => {
            if (sex) {
              e.currentTarget.style.transform = 'scale(1.05)'
            }
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.transform = 'scale(1)'
          }}
        >
          <span style={{ color: 'white', fontSize: '32px' }}>📷</span>
        </button>
      </main>

      {/* Footer Progress Indicator */}
      <footer style={{
        padding: '24px',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        gap: '16px'
      }}>
        <div style={{
          color: LIGHT_GREY,
          fontSize: '14px',
          fontWeight: 500
        }}>
          Calibration
        </div>
        <div style={{
          width: '40px',
          height: '1px',
          borderTop: `2px dashed ${LIGHT_GREY}`,
          borderBottom: `2px dashed ${LIGHT_GREY}`
        }} />
        <div style={{
          color: DARK_GREY,
          fontSize: '14px',
          fontWeight: 600
        }}>
          Imaging
        </div>
        <div style={{
          width: '40px',
          height: '1px',
          borderTop: `2px dashed ${LIGHT_GREY}`,
          borderBottom: `2px dashed ${LIGHT_GREY}`
        }} />
        <div style={{
          color: LIGHT_GREY,
          fontSize: '14px',
          fontWeight: 500
        }}>
          Data Processing
        </div>
      </footer>
    </div>
  )
}

