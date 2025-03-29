import React, { useState, useRef, useEffect, useCallback } from 'react';
import './Therapy.css';

const Therapy = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [highlightPosition, setHighlightPosition] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [totalTime, setTotalTime] = useState(0);
  const [playbackSpeed, setPlaybackSpeed] = useState(1.0);
  const [autoHighlightSpeed, setAutoHighlightSpeed] = useState(150); // Characters per minute
  const [isAutoHighlighting, setIsAutoHighlighting] = useState(false);
  const [currentText, setCurrentText] = useState(
    "It was a bright, sunny morning when Jack and his family decided to visit the zoo. Jack had been looking forward to this day for weeks. He loved animals and was excited to see them up close. His mom packed a picnic lunch, and they all got into the car to drive to the zoo. When they arrived, Jack could see many people already walking around. The zoo was full of families, children, and even some school groups. The first stop was the lion exhibit. Jack could hear the lions roaring from a distance. As they got closer, he saw the large, powerful animals resting in the shade. He was amazed at how big they were. Next, they went to see the monkeys. The monkeys were jumping from tree to tree and making funny noises. Jack laughed as he watched them swing around so easily. His little sister, Emma, pointed at the baby monkey and said it was the cutest thing she had ever seen. After the monkeys, Jack and his family walked to the elephant enclosure. The elephants were eating leaves and using their trunks to grab food. Jack was fascinated by how long their trunks were and how gently they ate. He thought the elephants looked very wise. Later, they visited the penguin exhibit. The penguins were swimming in the water and waddling on the ground. Jack liked how fast they could swim, and he watched them slide on their bellies with excitement. At the end of the day, Jack and his family sat on a bench and ate their lunch. They talked about their favorite animals. Jack couldn’t wait to tell his friends about his visit to the zoo."
  );

  // Refs for DOM elements and objects
  const recognitionRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const timerRef = useRef(null);
  const canvasRef = useRef(null);
  const wavesCanvasRef = useRef(null);
  const textContainerRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const animationFrameRef = useRef(null);
  const autoHighlightTimerRef = useRef(null);
  
  // Performance optimization refs
  const lastRecognizedIndexRef = useRef(0);
  const currentTranscriptRef = useRef('');
  const processingTimeoutRef = useRef(null);
  const textLowerCaseRef = useRef(currentText.toLowerCase());
  const wordsArrayRef = useRef(currentText.toLowerCase().split(/\s+/));

  // Precompute text data on init or when text changes
  useEffect(() => {
    textLowerCaseRef.current = currentText.toLowerCase();
    wordsArrayRef.current = currentText.toLowerCase().split(/\s+/);
  }, [currentText]);

  // Initialize canvas for wave visualization
  const initializeCanvas = useCallback(() => {
    if (!canvasRef.current || !wavesCanvasRef.current) return;
    
    const canvas = canvasRef.current;
    const wavesCanvas = wavesCanvasRef.current;
    
    // Main circular visualizer
    canvas.width = 80;
    canvas.height = 80;
    
    // Background waves
    wavesCanvas.width = window.innerWidth;
    wavesCanvas.height = 150;
  }, []);

  // Initialize speech recognition and audio visualization
  useEffect(() => {
    initializeCanvas();
    
    const handleResize = () => {
      if (wavesCanvasRef.current) {
        wavesCanvasRef.current.width = window.innerWidth;
      }
    };
    
    window.addEventListener('resize', handleResize);

    // Setup speech recognition
    if ('webkitSpeechRecognition' in window) {
      const SpeechRecognition = window.webkitSpeechRecognition;
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = true;
      recognitionRef.current.interimResults = true;
      
      // Use higher settings for more real-time response
      recognitionRef.current.interimResults = true;
      recognitionRef.current.maxAlternatives = 1;
      
      recognitionRef.current.onresult = handleSpeechResult;
      
      recognitionRef.current.onerror = (event) => {
        console.error('Speech recognition error', event.error);
        if (event.error === 'not-allowed') {
          alert('Microphone access is required for speech recognition.');
        }
      };
    } else {
      alert('Your browser does not support the Web Speech API. Please try Chrome or Edge.');
    }

    return () => {
      window.removeEventListener('resize', handleResize);
      
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach(track => track.stop());
      }
      
      if (recognitionRef.current && isRecording) {
        recognitionRef.current.stop();
      }
      
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
      
      if (processingTimeoutRef.current) {
        clearTimeout(processingTimeoutRef.current);
      }
      
      if (autoHighlightTimerRef.current) {
        clearInterval(autoHighlightTimerRef.current);
      }
    };
  }, [initializeCanvas]);

  // Auto-highlight effect
  useEffect(() => {
    if (isAutoHighlighting) {
      // Clear any existing timer
      if (autoHighlightTimerRef.current) {
        clearInterval(autoHighlightTimerRef.current);
      }
      
      // Calculate interval based on speed (characters per minute)
      // Convert to milliseconds per character
      const msPerChar = 60000 / autoHighlightSpeed;
      
      autoHighlightTimerRef.current = setInterval(() => {
        setHighlightPosition(prev => {
          const newPosition = prev + 1;
          
          // If we've reached the end, reset or stop
          if (newPosition >= currentText.length) {
            // Optional: reset to beginning
            // return 0;
            
            // Or stop auto-highlighting
            setIsAutoHighlighting(false);
            return prev;
          }
          
          scrollToHighlight(newPosition);
          return newPosition;
        });
      }, msPerChar);
      
      return () => {
        if (autoHighlightTimerRef.current) {
          clearInterval(autoHighlightTimerRef.current);
        }
      };
    }
  }, [isAutoHighlighting, autoHighlightSpeed, currentText]);

  // Handle speech recognition results with optimized performance
  const handleSpeechResult = (event) => {
    // Skip processing if we have a pending timeout (debounce)
    if (processingTimeoutRef.current) {
      clearTimeout(processingTimeoutRef.current);
    }
    
    let interimTranscript = '';
    let finalTranscript = currentTranscriptRef.current;
    
    for (let i = event.resultIndex; i < event.results.length; i++) {
      const transcript = event.results[i][0].transcript;
      if (event.results[i].isFinal) {
        finalTranscript += ' ' + transcript;
        // Process final results immediately for better responsiveness
        processTranscript(finalTranscript);
      } else {
        interimTranscript += transcript;
        // Process interim results with a very short delay to reduce processing load
        processingTimeoutRef.current = setTimeout(() => {
          processTranscript(finalTranscript + ' ' + interimTranscript);
        }, 50);
      }
    }
    
    currentTranscriptRef.current = finalTranscript;
  };

  // Process the transcript with optimized matching algorithm
  const processTranscript = (transcript) => {
    if (!transcript || transcript.trim() === '') return;
    
    const cleanTranscript = transcript.toLowerCase().trim();
    const words = cleanTranscript.split(/\s+/);
    
    // Fast processing - use the last few words
    const lastFewWordsCount = Math.min(5, words.length);
    const startIndex = Math.max(0, words.length - lastFewWordsCount);
    const lastFewWords = words.slice(startIndex);
    
    // Use a faster match method for short phrases
    findAndHighlightMatch(lastFewWords);
  };

  // Optimized matching algorithm
  const findAndHighlightMatch = (searchWords) => {
    if (!searchWords || searchWords.length === 0) return;
    
    const textLower = textLowerCaseRef.current;
    const allWords = wordsArrayRef.current;
    
    // Start searching from the last recognized position
    const startPosition = Math.max(0, lastRecognizedIndexRef.current);
    
    // Try to find exact matches for the last spoken word
    const lastWord = searchWords[searchWords.length - 1];
    
    // Skip very short words (less likely to be unique)
    if (lastWord && lastWord.length >= 3) {
      // Look for the word after our current position
      let wordIndex = -1;
      let bestMatchPos = -1;
      
      // Find all occurrences of the word
      let searchPos = startPosition;
      while ((searchPos = textLower.indexOf(lastWord, searchPos)) !== -1) {
        // Check if this is a full word match (not part of another word)
        const isWordBoundaryBefore = searchPos === 0 || !textLower[searchPos - 1].match(/[a-z0-9]/i);
        const isWordBoundaryAfter = searchPos + lastWord.length >= textLower.length || 
                                     !textLower[searchPos + lastWord.length].match(/[a-z0-9]/i);
                                     
        if (isWordBoundaryBefore && isWordBoundaryAfter) {
          // This is a legitimate word boundary match
          if (searchPos > startPosition) {
            bestMatchPos = searchPos + lastWord.length;
            break; // Take the first match after our current position
          }
        }
        searchPos += lastWord.length;
      }
      
      // If we found a match
      if (bestMatchPos !== -1) {
        lastRecognizedIndexRef.current = bestMatchPos;
        setHighlightPosition(bestMatchPos);
        scrollToHighlight(bestMatchPos);
      }
    }
  };
  
  // Faster scroll implementation
  const scrollToHighlight = (position) => {
    if (!textContainerRef.current) return;
    
    // Skip animation for faster response
    const container = textContainerRef.current;
    
    // Calculate approximation of where this text position would be
    const textLength = currentText.length;
    const containerHeight = container.scrollHeight;
    const approximatePosition = (position / textLength) * containerHeight;
    
    // Center the position in the viewport
    const scrollPosition = Math.max(0, approximatePosition - (container.clientHeight / 2));
    
    // Use scrollTo without smooth behavior for immediate response
    container.scrollTop = scrollPosition;
  };

  const toggleRecording = async () => {
    // If auto-highlighting is on, turn it off when recording starts
    if (isAutoHighlighting) {
      setIsAutoHighlighting(false);
    }
    
    if (!isRecording) {
      try {
        // Reset position when starting a new recording
        setHighlightPosition(0);
        lastRecognizedIndexRef.current = 0;
        currentTranscriptRef.current = '';
        setCurrentTime(0);
        setTotalTime(0);
        
        // Start timer
        let seconds = 0;
        timerRef.current = setInterval(() => {
          seconds++;
          setCurrentTime(seconds);
          setTotalTime(seconds);
        }, 1000);
        
        // Start audio stream
        const stream = await navigator.mediaDevices.getUserMedia({ 
          audio: { 
            echoCancellation: true,
            noiseSuppression: true,
            sampleRate: 16000
          } 
        });
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
        }
        
        setIsRecording(true);
      } catch (error) {
        console.error('Error accessing microphone:', error);
        alert('Please allow microphone access to use this feature.');
      }
    } else {
      // Stop recording
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach(track => track.stop());
      }
      
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
      
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
      
      if (processingTimeoutRef.current) {
        clearTimeout(processingTimeoutRef.current);
      }
      
      setIsRecording(false);
    }
  };

  const toggleAutoHighlight = () => {
    // If recording is on, turn it off when auto-highlighting starts
    if (isRecording) {
      toggleRecording();
    }
    
    setIsAutoHighlighting(prev => !prev);
    
    // Reset position when starting auto-highlighting
    if (!isAutoHighlighting) {
      setHighlightPosition(0);
    }
  };

  const changeAutoHighlightSpeed = (change) => {
    setAutoHighlightSpeed(prev => {
      const newSpeed = Math.max(50, Math.min(500, prev + change));
      return newSpeed;
    });
  };

  const startVisualization = () => {
    if (!analyserRef.current) return;
    
    const bufferLength = analyserRef.current.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    const wavesCanvas = wavesCanvasRef.current;
    const wavesCtx = wavesCanvas.getContext('2d');
    
    const renderFrame = () => {
      animationFrameRef.current = requestAnimationFrame(renderFrame);
      
      analyserRef.current.getByteFrequencyData(dataArray);
      
      // Clear canvases
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      wavesCtx.clearRect(0, 0, wavesCanvas.width, wavesCanvas.height);
      
      // Draw waves with purple theme
      drawWaves(wavesCtx, wavesCanvas.width, wavesCanvas.height, dataArray);
      
      // Draw circle visualization
      drawCircleVisualizer(ctx, canvas.width, canvas.height, dataArray);
    };
    
    renderFrame();
  };

  const drawCircleVisualizer = (ctx, width, height, dataArray) => {
    const centerX = width / 2;
    const centerY = height / 2;
    const radius = Math.min(centerX, centerY) - 5;
    
    // Draw background circle
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
    ctx.strokeStyle = 'rgba(124, 58, 237, 0.2)';
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // Calculate average amplitude for size
    let sum = 0;
    for (let i = 0; i < dataArray.length; i++) {
      sum += dataArray[i];
    }
    const avgAmplitude = sum / dataArray.length;
    const dynamicRadius = radius * (0.8 + avgAmplitude / 512);
    
    // Draw animated fill
    ctx.beginPath();
    ctx.arc(centerX, centerY, dynamicRadius, 0, 2 * Math.PI);
    ctx.fillStyle = 'rgba(124, 58, 237, 0.6)';
    ctx.fill();
  };

  const drawWaves = (ctx, width, height, dataArray) => {
    const time = Date.now() * 0.001;
    const waveCount = 3;
    
    for (let waveIndex = 0; waveIndex < waveCount; waveIndex++) {
      ctx.beginPath();
      
      const alpha = 0.3 - waveIndex * 0.1;
      const amplitudeFactor = 1 - waveIndex * 0.2;
      
      // Calculate average of frequency data for amplitude (optimized)
      let sum = 0;
      const sampleSize = Math.min(dataArray.length, 32); // Sample fewer data points for performance
      for (let i = 0; i < sampleSize; i++) {
        sum += dataArray[i * Math.floor(dataArray.length / sampleSize)];
      }
      const averageAmplitude = sum / sampleSize;
      const dynamicAmplitude = Math.max(5, averageAmplitude * 0.5 * amplitudeFactor);
      
      ctx.moveTo(0, height / 2);
      
      // Optimize wave drawing - use fewer points for better performance
      const step = Math.max(5, Math.floor(width / 120)); // Adaptive step size based on screen width
      
      for (let x = 0; x < width; x += step) {
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

  // Format time for display (MM:SS)
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  // Change playback speed
  const adjustPlaybackSpeed = () => {
    const speeds = [0.5, 0.75, 1.0, 1.25, 1.5];
    const currentIndex = speeds.indexOf(playbackSpeed);
    const nextIndex = (currentIndex + 1) % speeds.length;
    setPlaybackSpeed(speeds[nextIndex]);
  };

  // Memoize text rendering to avoid unnecessary re-renders
  const renderTextWithHighlight = useCallback(() => {
    if (!currentText) return null;
    
    const beforeHighlight = currentText.substring(0, highlightPosition);
    const afterHighlight = currentText.substring(highlightPosition);
    
    return (
      <div className="text-container" ref={textContainerRef}>
        <p>
          <span className="spoken-text">{beforeHighlight}</span>
          <span className="unspoken-text">{afterHighlight}</span>
        </p>
      </div>
    );
  }, [currentText, highlightPosition]);

  return (
    <div className="Therapy-container">
      {/* Wave visualization background */}
      <canvas ref={wavesCanvasRef} className="waves-canvas"></canvas>
      
      {/* Header */}
      <div className="header">
        <h1 className="app-title">Therapy</h1>
        <p className="app-subtitle">Personalized Stutter Detection and Helper Model</p>
      </div>
      
      {/* Main content */}
      <div className="main-content">
        {/* Text display */}
        <div className="text-display">
          {renderTextWithHighlight()}
        </div>
      </div>
      
      {/* Auto-highlight speed controls */}
      <div className="auto-highlight-controls">
        <div className="speed-display">
          <span>Auto-Highlight Speed: {autoHighlightSpeed} CPM</span>
        </div>
        <div className="speed-buttons">
          <button 
            className="speed-btn" 
            onClick={() => changeAutoHighlightSpeed(-25)}
            disabled={autoHighlightSpeed <= 50}
          >
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M19 12H5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>
          <button 
            className={`auto-highlight-btn ${isAutoHighlighting ? 'active' : ''}`} 
            onClick={toggleAutoHighlight}
          >
            {isAutoHighlighting ? 'Pause' : 'Auto-Highlight'}
          </button>
          <button 
            className="speed-btn" 
            onClick={() => changeAutoHighlightSpeed(25)}
            disabled={autoHighlightSpeed >= 500}
          >
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M12 5V19" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M5 12H19" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>
        </div>
      </div>
      
      {/* Control panel */}
      <div className="controls">
        <div className="time-display">
          {formatTime(currentTime)} / {formatTime(totalTime)}
        </div>
        
        <div className="control-buttons">
          <button className="control-btn volume-btn">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M11 5L6 9H2V15H6L11 19V5Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M15.54 8.46C16.4774 9.39764 17.004 10.6692 17.004 11.995C17.004 13.3208 16.4774 14.5924 15.54 15.53" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>
          
          <button className="control-btn record-btn" onClick={toggleRecording}>
            <canvas ref={canvasRef} className="visualizer"></canvas>
            {isRecording ? (
              <svg className="control-icon" width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <rect x="6" y="6" width="12" height="12" rx="1" fill="white"/>
              </svg>
            ) : (
              <svg className="control-icon" width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M8 5V19L19 12L8 5Z" fill="white"/>
              </svg>
            )}
          </button>
          
          <button className="control-btn speed-btn" onClick={adjustPlaybackSpeed}>
            <span>{playbackSpeed}x</span>
          </button>
        </div>
      </div>
      
      {/* Add CSS for new controls */}
      <style>{`
        .auto-highlight-controls {
          display: flex;
          flex-direction: column;
          align-items: center;
          margin-bottom: 20px;
          background: rgba(124, 58, 237, 0.1);
          padding: 12px;
          border-radius: 12px;
          width: 90%;
          max-width: 500px;
          margin: 0 auto 20px;
        }
        
        .speed-display {
          font-size: 14px;
          margin-bottom: 8px;
          color: #7c3aed;
        }
        
        .speed-buttons {
          display: flex;
          align-items: center;
          gap: 12px;
        }
        
        .speed-btn {
          background: rgba(124, 58, 237, 0.2);
          border: none;
          border-radius: 50%;
          width: 36px;
          height: 36px;
          display: flex;
          align-items: center;
          justify-content: center;
          cursor: pointer;
          color: #7c3aed;
          transition: all 0.2s;
        }
        
        .speed-btn:hover {
          background: rgba(124, 58, 237, 0.3);
        }
        
        .speed-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
        
        .auto-highlight-btn {
          background: #7c3aed;
          color: white;
          border: none;
          padding: 8px 16px;
          border-radius: 8px;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s;
        }
        
        .auto-highlight-btn:hover {
          background: #6d28d9;
        }
        
        .auto-highlight-btn.active {
          background: #ef4444;
        }
        
        .auto-highlight-btn.active:hover {
          background: #dc2626;
        }
      `}</style>
    </div>
  );
};

export default Therapy;