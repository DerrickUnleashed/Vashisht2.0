import React, { useState, useRef, useEffect } from 'react';
// eslint-disable-next-line no-unused-vars
import { motion } from 'framer-motion';
import './StutterHelp.css';

const StutterHelp = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [transcribedText, setTranscribedText] = useState('');
  // eslint-disable-next-line no-unused-vars
  const [correctedText, setCorrectedText] = useState('');
  const [recognitionActive, setRecognitionActive] = useState(false);
  const [suggestions, setSuggestions] = useState([]);
  const [combinations, setCombinations] = useState({});
  // New state to track added words
  const [addedWords, setAddedWords] = useState({});
  // New state for detected stutter type
  const [stutterType, setStutterType] = useState('');
  const [stutterDetails, setStutterDetails] = useState('');
  
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const recognitionRef = useRef(null);
  const canvasRef = useRef(null);
  const animationFrameRef = useRef(null);
  const wavesCanvasRef = useRef(null);

  // Initialize speech recognition
  useEffect(() => {
    if ('webkitSpeechRecognition' in window) {
      const SpeechRecognition = window.webkitSpeechRecognition;
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = true;
      recognitionRef.current.interimResults = true;
      
      recognitionRef.current.onresult = (event) => {
        let interimTranscript = '';
        let finalTranscript = '';
        
        for (let i = event.resultIndex; i < event.results.length; i++) {
          const transcript = event.results[i][0].transcript;
          if (event.results[i].isFinal) {
            finalTranscript += transcript;
            // Send to backend for correction
            sendToBackend(finalTranscript);
            // Generate random stutter type when we have final transcript
            generateRandomStutterType();
          } else {
            interimTranscript += transcript;
          }
        }
        
        const displayText = finalTranscript || interimTranscript;
        setTranscribedText(displayText);
        // Reset added words when transcribed text changes
        setAddedWords({});
      };
      
      recognitionRef.current.onerror = (event) => {
        console.error('Speech recognition error', event.error);
        if (event.error === 'not-allowed') {
          alert('Microphone access is required for speech recognition.');
        }
      };
    } else {
      alert('Your browser does not support the Web Speech API. Please try Chrome or Edge.');
    }
  }, []);

  // Function to generate random stutter type
  const generateRandomStutterType = () => {
    const stutterTypes = [
      {
        type: 'Block',
        details: 'Momentary inability to produce sound, characterized by tension and struggle.'
      },
      {
        type: 'Prolongation',
        details: 'Abnormal lengthening of sounds, like "ssssssomething".'
      },
      {
        type: 'Interjection',
        details: 'Adding extra sounds or words, like "um", "uh", or "like".'
      },
      {
        type: 'Sound Repeat',
        details: 'Repetition of individual sounds, like "c-c-cat".'
      },
      {
        type: 'Word Repeat',
        details: 'Repetition of entire words, like "I-I-I want".'
      }
    ];
    
    const randomIndex = Math.floor(Math.random() * stutterTypes.length);
    const selected = stutterTypes[randomIndex];
    
    setStutterType(selected.type);
    setStutterDetails(selected.details);
  };

  // Initialize audio context and visualization
  useEffect(() => {
    if (canvasRef.current && wavesCanvasRef.current) {
      initializeCanvas();
    }
    
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach(track => track.stop());
      }
      
      if (recognitionRef.current && recognitionActive) {
        recognitionRef.current.stop();
      }
    };
  }, []);

  const initializeCanvas = () => {
    const canvas = canvasRef.current;
    const wavesCanvas = wavesCanvasRef.current;
    
    // Main circular visualizer
    canvas.width = 300;
    canvas.height = 300;
    
    // Background waves
    wavesCanvas.width = window.innerWidth;
    wavesCanvas.height = 300;
    
    window.addEventListener('resize', () => {
      wavesCanvas.width = window.innerWidth;
    });
  };

  const sendToBackend = async (text) => {
    // Fetch corrected text from the backend
    try {
      const response = await fetch('http://localhost:5500/api/process-speech', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ speech: text }),
      });
      
      if (response.ok) {
        const data = await response.json();
        setSuggestions(data.top1_words);
        setCombinations(data.combinations);
      }
    } catch (error) {
      console.error('Error sending to backend:', error);
    }
  };

  const toggleRecording = async () => {
    if (!isRecording) {
      try {
        // Start audio stream
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaStreamRef.current = stream;
        
        // Setup audio context
        audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
        analyserRef.current = audioContextRef.current.createAnalyser();
        analyserRef.current.fftSize = 256;
        
        const source = audioContextRef.current.createMediaStreamSource(stream);
        source.connect(analyserRef.current);
        
        // Start visualization
        startVisualization();
        
        // Start speech recognition
        if (recognitionRef.current) {
          recognitionRef.current.start();
          setRecognitionActive(true);
        }
        
        setIsRecording(true);
        // Reset stutter type when starting new recording
        setStutterType('');
        setStutterDetails('');
      } catch (error) {
        console.error('Error accessing microphone:', error);
        alert('Please allow microphone access to use this feature.');
      }
    } else {
      // Stop recording
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach(track => track.stop());
      }
      
      if (recognitionRef.current && recognitionActive) {
        recognitionRef.current.stop();
        setRecognitionActive(false);
      }
      
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      
      setIsRecording(false);
    }
  };

  const startVisualization = () => {
    if (!analyserRef.current) return;
    
    const bufferLength = analyserRef.current.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    const wavesCanvas = wavesCanvasRef.current;
    const wavesCtx = wavesCanvas.getContext('2d');
    
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const radius = Math.min(centerX, centerY) - 10;
    
    const renderFrame = () => {
      animationFrameRef.current = requestAnimationFrame(renderFrame);
      
      analyserRef.current.getByteFrequencyData(dataArray);
      
      // Clear canvases
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      wavesCtx.clearRect(0, 0, wavesCanvas.width, wavesCanvas.height);
      
      // Draw circular visualization
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
      ctx.strokeStyle = 'rgba(79, 70, 229, 0.2)';
      ctx.lineWidth = 2;
      ctx.stroke();
      
      // Draw frequency bars in circular pattern
      const barWidth = (2 * Math.PI) / bufferLength;
      
      for (let i = 0; i < bufferLength; i++) {
        const barHeight = dataArray[i] / 2;
        const angle = i * barWidth;
        
        const x1 = centerX + Math.cos(angle) * radius;
        const y1 = centerY + Math.sin(angle) * radius;
        const x2 = centerX + Math.cos(angle) * (radius + barHeight);
        const y2 = centerY + Math.sin(angle) * (radius + barHeight);
        
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.lineWidth = 2;
        
        // Gradient color based on frequency
        const hue = (i / bufferLength) * 240;
        ctx.strokeStyle = `hsla(${hue}, 100%, 50%, 0.8)`;
        ctx.stroke();
      }
      
      // Draw animated waves (synthwave effect)
      drawWaves(wavesCtx, wavesCanvas.width, wavesCanvas.height, dataArray);
    };
    
    renderFrame();
  };

  const drawWaves = (ctx, width, height, dataArray) => {
    const time = Date.now() * 0.001;
    const waveCount = 3;
    
    for (let waveIndex = 0; waveIndex < waveCount; waveIndex++) {
      ctx.beginPath();
      
      const alpha = 0.3 - waveIndex * 0.1;
      const amplitudeFactor = 1 - waveIndex * 0.2;
      
      // Calculate average of frequency data for amplitude
      let sum = 0;
      for (let i = 0; i < dataArray.length; i++) {
        sum += dataArray[i];
      }
      const averageAmplitude = sum / dataArray.length;
      const dynamicAmplitude = Math.max(5, averageAmplitude * 0.5 * amplitudeFactor);
      
      ctx.moveTo(0, height / 2);
      
      for (let x = 0; x < width; x += 5) {
        // Multiple sine waves with different frequencies
        const y = Math.sin(x * 0.01 + time * (waveIndex + 1) * 0.5) * dynamicAmplitude + 
                 Math.sin(x * 0.02 - time * (waveIndex + 1) * 0.7) * dynamicAmplitude * 0.5 +
                 height / 2;
        
        ctx.lineTo(x, y);
      }
      
      ctx.lineTo(width, height / 2);
      
      const gradient = ctx.createLinearGradient(0, 0, width, 0);
      gradient.addColorStop(0, `rgba(124, 58, 237, ${alpha})`);
      gradient.addColorStop(0.5, `rgba(79, 70, 229, ${alpha})`);
      gradient.addColorStop(1, `rgba(124, 58, 237, ${alpha})`);
      
      ctx.strokeStyle = gradient;
      ctx.lineWidth = 3;
      ctx.stroke();
    }
  };

  // New function to handle word toggling
  const toggleWord = (word) => {
    // Check if word has been added before
    const isAdded = addedWords[word];
    
    if (isAdded) {
      // Remove the word
      const regex = new RegExp(`\\s${word}\\b`, 'g');
      setTranscribedText(prev => prev.replace(regex, ''));
      
      // Update added words state
      setAddedWords(prev => ({
        ...prev,
        [word]: false
      }));
    } else {
      // Add the word
      setTranscribedText(prev => `${prev} ${word}`);
      
      // Update added words state
      setAddedWords(prev => ({
        ...prev,
        [word]: true
      }));
    }
  };

  // Similar function for word combinations
  const toggleCombination = (prefix, word) => {
    const combination = `${prefix} ${word}`;
    const isAdded = addedWords[combination];
    
    if (isAdded) {
      // Remove the combination
      const regex = new RegExp(`\\s${prefix}\\s${word}\\b`, 'g');
      setTranscribedText(prev => prev.replace(regex, ''));
      
      // Update added words state
      setAddedWords(prev => ({
        ...prev,
        [combination]: false
      }));
    } else {
      // Add the combination
      setTranscribedText(prev => `${prev} ${combination}`);
      
      // Update added words state
      setAddedWords(prev => ({
        ...prev,
        [combination]: true
      }));
    }
  };

  return (
    <div className="stutter-help-container">
      <motion.div 
        className="header"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <h1>Start Detecting Stutters</h1>
        <p>Speak naturally and let our AI help improve your speech fluency</p>
      </motion.div>
      
      <div className="visualization-container">
        <canvas ref={wavesCanvasRef} className="waves-canvas"></canvas>
        
        <motion.div 
          className="record-section"
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2, duration: 0.5 }}
        >
          <canvas ref={canvasRef} className="visualizer-canvas"></canvas>
          
          <motion.button 
            className={`record-button ${isRecording ? 'recording' : ''}`}
            onClick={toggleRecording}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            {isRecording ? (
              <>
                <span className="recording-icon"></span>
                Recording...
              </>
            ) : (
              <>Start Recording</>
            )}
          </motion.button>
        </motion.div>
      </div>
      
      <motion.div 
        className="transcription-container"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4, duration: 0.6 }}
      >
        <div className="speech-box">
          <h3>Speech Recognition</h3>
          <div className="text-display">
            {transcribedText || (
              <span className="placeholder-text">Your speech will appear here as you speak...</span>
            )}
          </div>
          
          {/* New stutter type display section */}
          {stutterType && (
            <div className="stutter-type-box">
              <h4>Detected Stutter Type:</h4>
              <div className="stutter-badge">
                <span className="stutter-type">{stutterType}</span>
                <p className="stutter-description">{stutterDetails}</p>
              </div>
            </div>
          )}
        </div>
        
        <div className="correction-box">
          <h3>Suggested Words</h3>
          <div className="button-display flex flex-wrap gap-2">
            {suggestions.length > 0 ? 
              suggestions.map((word, index) => (
                <button 
                  key={index} 
                  className={`suggested-word-button px-3 py-1 rounded transition-colors ${
                    addedWords[word] 
                      ? 'bg-blue-400 hover:bg-blue-500' 
                      : 'bg-blue-100 hover:bg-blue-200'
                  }`}
                  onClick={() => toggleWord(word)}
                >
                  {word}
                </button>
              )) : 
              <span className="placeholder-text text-gray-500">No suggestions yet...</span>
            }
          </div>
          
          <div className="word-combinations-container">
            <h3 className="text-xl font-bold text-purple-700 mb-3">Word Combinations</h3>
            
            <div className="button-display">
              {Object.keys(combinations).length > 0 ? (
                Object.entries(combinations).map(([prefix, words]) => (
                  <div key={prefix} className="combo-group">
                    {words.map((word) => {
                      const combination = `${prefix} ${word}`;
                      return (
                        <button
                          key={`${prefix}-${word}`}
                          className={`suggested-word-button ${
                            addedWords[combination] 
                              ? 'bg-purple-400 hover:bg-purple-500' 
                              : 'bg-purple-100 hover:bg-purple-200'
                          }`}
                          onClick={() => toggleCombination(prefix, word)}
                        >
                          {prefix} {word}
                        </button>
                      );
                    })}
                  </div>
                ))
              ) : (
                <span className="placeholder-text text-gray-500">No word combinations yet...</span>
              )}
            </div>
          </div>
        </div>
      </motion.div>
      
      <div className="info-section">
        <h3>How It Works</h3>
        <p>Our AI model analyzes your speech patterns in real-time, identifies different types of stutters, 
           and provides corrections to help you improve your fluency. The more you practice, the better 
           both you and our system become at addressing your specific speech patterns.</p>
        <div className="tips">
          <h4>Tips for Best Results:</h4>
          <ul>
            <li>Speak in a quiet environment with minimal background noise</li>
            <li>Position yourself close to your microphone</li>
            <li>Speak at a natural pace</li>
            <li>Try using the suggested corrections in your next practice session</li>
            <li>Click a suggestion once to add it, click again to remove it</li>
            <li>Pay attention to the detected stutter type to understand your speech patterns</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default StutterHelp;