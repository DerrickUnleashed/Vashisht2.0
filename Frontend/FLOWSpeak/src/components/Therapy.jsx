import React, { useState, useRef, useEffect } from 'react';
import './Therapy.css';

const Therapy = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [highlightPosition, setHighlightPosition] = useState(0);
  const [lastRecognizedIndex, setLastRecognizedIndex] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [totalTime, setTotalTime] = useState(0);
  const [playbackSpeed, setPlaybackSpeed] = useState(1.0);
  const [currentText, setCurrentText] = useState(
    "Hello, fellow stutterers, my name is Max Wolens, I'm 29 years old, and I've stuttered since I was 2 years old. After I graduated from college, I started a career in technology, where I was often required to make presentations for my job. I had made presentations in college, but never professionally. I realized quickly that I wasn't able to maintain presentations on Zoom, as I experienced a ton of blocks, and I was truly experiencing cognitive overload — I wasn't able to simultaneously produce fluent speech while coming up with a dialogue that sounded intelligent and coherent."
  );

  // Refs
  const recognitionRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const timerRef = useRef(null);
  const canvasRef = useRef(null);
  const wavesCanvasRef = useRef(null);
  const textContainerRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const animationFrameRef = useRef(null);

  // Initialize speech recognition and audio visualization
  useEffect(() => {
    // Initialize canvas for wave visualization
    if (canvasRef.current && wavesCanvasRef.current) {
      initializeCanvas();
    }

    // Setup speech recognition
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
            compareWithText(finalTranscript);
          } else {
            interimTranscript += transcript;
          }
        }
        
        const displayText = finalTranscript || interimTranscript;
        setTranscript(displayText);
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

    // Cleanup function
    return () => {
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
    };
  }, []);

  const initializeCanvas = () => {
    const canvas = canvasRef.current;
    const wavesCanvas = wavesCanvasRef.current;
    
    // Main circular visualizer
    canvas.width = 80;
    canvas.height = 80;
    
    // Background waves
    wavesCanvas.width = window.innerWidth;
    wavesCanvas.height = 150;
    
    window.addEventListener('resize', () => {
      wavesCanvas.width = window.innerWidth;
    });
  };

  // Compare spoken text with the displayed text to update highlight position
  const compareWithText = (spokenText) => {
    if (!spokenText) return;
    
    const cleanSpokenText = spokenText.toLowerCase().trim();
    const textLower = currentText.toLowerCase();
    
    // Extract the last few words for more accurate matching
    const words = cleanSpokenText.split(/\s+/);
    
    // Get a phrase of the last few words (more reliable for matching)
    const lastFewWords = words.slice(-Math.min(5, words.length)).join(' ');
    
    if (lastFewWords.length < 3) return; // Skip very short phrases
    
    // Find this phrase in the text, starting from the last position
    // We'll use a sliding window approach if exact match fails
    let matchPosition = -1;
    let searchStartPos = Math.max(0, lastRecognizedIndex - 20); // Start a bit before last match
    
    // Try exact match first
    matchPosition = textLower.indexOf(lastFewWords, searchStartPos);
    
    // If exact match fails, try matching individual words
    if (matchPosition === -1) {
      // Try each word in the last few words
      for (let i = words.length - 1; i >= Math.max(0, words.length - 5); i--) {
        const word = words[i];
        if (word.length < 3) continue; // Skip very short words
        
        const wordPos = textLower.indexOf(word, searchStartPos);
        if (wordPos !== -1) {
          // Found a match for this word
          matchPosition = wordPos + word.length;
          break;
        }
      }
    } else {
      // Found exact match for the phrase
      matchPosition = matchPosition + lastFewWords.length;
    }
    
    // Update highlight position if we found a match
    if (matchPosition !== -1 && matchPosition > lastRecognizedIndex) {
      setHighlightPosition(matchPosition);
      setLastRecognizedIndex(matchPosition);
      
      // Scroll the text container to show the current position
      scrollToHighlight(matchPosition);
    }
  };
  
  // Auto-scroll to keep the highlight visible
  const scrollToHighlight = (position) => {
    if (!textContainerRef.current) return;
    
    // Calculate the position as a percentage of total text length
    const scrollPercentage = position / currentText.length;
    
    // Apply scroll based on percentage
    const container = textContainerRef.current;
    const scrollTarget = Math.max(
      0, 
      (scrollPercentage * container.scrollHeight) - (container.clientHeight / 2)
    );
    
    // Smooth scroll to the position
    container.scrollTo({
      top: scrollTarget,
      behavior: 'smooth'
    });
  };

  const toggleRecording = async () => {
    if (!isRecording) {
      try {
        // Reset position when starting a new recording
        setHighlightPosition(0);
        setLastRecognizedIndex(0);
        setTranscript('');
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

  // Render the text with highlighted portion
  const renderTextWithHighlight = () => {
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
  };

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
    </div>
  );
};

export default Therapy;