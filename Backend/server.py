from flask import Flask, request, jsonify, render_template, Response, send_from_directory
from flask_cors import CORS
import os
import tempfile
import numpy as np
import time
import logging
import threading
import json
import uuid
import wave
import soundfile as sf
from collections import deque
import websocket
from flask_socketio import SocketIO
import torch

# Initialize Flask app
app = Flask(__name__, static_folder="../Frontend/FLOWSpeak/dist/assets", template_folder="../Frontend/FLOWSpeak/dist")
CORS(app)  # Enable CORS for all routes
socketio = SocketIO(app, cors_allowed_origins="*")  # Initialize SocketIO for real-time communication

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dictionary to store active audio streams
active_streams = {}
# Dictionary to store user session data
user_sessions = {}

# Audio processing parameters
SAMPLE_RATE = 16000  # Common sample rate for speech recognition
CHUNK_SIZE = 4096    # Audio chunk size for processing
BUFFER_SIZE = 5      # Number of chunks to buffer before processing

# ------------------------
# STUTTER DETECTION MODEL
# ------------------------
# Note: In a real implementation, you would load your ML model here
# This is a placeholder for the actual ML model integration

class StutterDetector:
    """
    Placeholder for the actual ML model that would detect and classify stutters
    In a real implementation, this would load a trained model (e.g., using TensorFlow or PyTorch)
    """
    def __init__(self):
        # Initialize your model here
        # For example: self.model = tf.keras.models.load_model('path/to/model')
        logger.info("StutterDetector initialized (placeholder)")
        
        # TODO: Load your pretrained model here
        # Example with PyTorch:
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.model = YourStutterModel()
        # self.model.load_state_dict(torch.load('path/to/model.pth'))
        # self.model.to(self.device)
        # self.model.eval()
        
    def process_audio(self, audio_data):
        """
        Process audio data to detect stutters
        
        Parameters:
        audio_data: Audio data (numpy array or similar)
        
        Returns:
        Dictionary with detection results and classifications
        """
        # Placeholder implementation
        # In a real implementation, you would:
        # 1. Preprocess the audio
        # 2. Extract features
        # 3. Run inference with your model
        # 4. Post-process results
        
        # TODO: Replace with actual model inference
        # Example implementation:
        if audio_data is None:
            # Simulate processing time
            time.sleep(0.1)
            
            # Return placeholder results (random for demonstration)
            return {
                "has_stutter": np.random.choice([True, False], p=[0.3, 0.7]),
                "stutter_type": np.random.choice(["block", "repetition", "prolongation", "none"], 
                                                p=[0.2, 0.2, 0.1, 0.5]),
                "confidence": np.random.uniform(0.7, 0.99)
            }
        
        try:
            # Process actual audio data
            # 1. Preprocess audio (normalize, filter noise, etc.)
            # processed_audio = self.preprocess_audio(audio_data)
            
            # 2. Extract features (MFCC, spectrogram, etc.)
            # features = self.extract_features(processed_audio)
            
            # 3. Run model inference
            # Example with PyTorch:
            # with torch.no_grad():
            #     features_tensor = torch.tensor(features, device=self.device)
            #     predictions = self.model(features_tensor)
            
            # 4. Post-process results
            # stutter_type = self.decode_predictions(predictions)
            
            # For now, return placeholder results
            return {
                "has_stutter": np.random.choice([True, False], p=[0.3, 0.7]),
                "stutter_type": np.random.choice(["block", "repetition", "prolongation", "none"], 
                                                p=[0.2, 0.2, 0.1, 0.5]),
                "confidence": np.random.uniform(0.7, 0.99)
            }
            
        except Exception as e:
            logger.error(f"Error in stutter detection: {str(e)}")
            return {
                "has_stutter": False,
                "stutter_type": "none",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def preprocess_audio(self, audio_data):
        """
        Preprocess audio data for model input
        
        Parameters:
        audio_data: Raw audio data
        
        Returns:
        Preprocessed audio data
        """
        # TODO: Implement audio preprocessing
        # Example steps:
        # 1. Normalize audio
        # 2. Apply filters (bandpass, noise reduction)
        # 3. Segment audio into frames
        # return preprocessed_audio
        pass
        
    def extract_features(self, processed_audio):
        """
        Extract features from preprocessed audio
        
        Parameters:
        processed_audio: Preprocessed audio data
        
        Returns:
        Extracted features
        """
        # TODO: Implement feature extraction
        # Example features:
        # 1. MFCC (Mel-frequency cepstral coefficients)
        # 2. Spectrograms
        # 3. Prosodic features (pitch, energy, etc.)
        # return features
        pass
        
    def decode_predictions(self, predictions):
        """
        Decode model predictions into stutter classifications
        
        Parameters:
        predictions: Raw model output
        
        Returns:
        Dictionary with stutter classifications
        """
        # TODO: Implement prediction decoding
        # Example:
        # stutter_classes = ['block', 'repetition', 'prolongation', 'none']
        # predicted_class = stutter_classes[predictions.argmax()]
        # confidence = predictions.max()
        # return {
        #     "has_stutter": predicted_class != 'none',
        #     "stutter_type": predicted_class,
        #     "confidence": confidence
        # }
        pass

class TextCorrector:
    """
    Placeholder for the text correction model
    In a real implementation, this would use NLP models to correct stuttered speech
    """
    def __init__(self):
        logger.info("TextCorrector initialized (placeholder)")
        
        # TODO: Load your pretrained NLP model here
        # Example:
        # self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained("your-model-name")
        # self.tokenizer = transformers.AutoTokenizer.from_pretrained("your-model-name")
        
    def correct_text(self, text, stutter_info=None):
        """
        Correct text based on detected stutters
        
        Parameters:
        text: The transcribed text
        stutter_info: Information about detected stutters (optional)
        
        Returns:
        Corrected text
        """
        # TODO: Replace with actual NLP model for correction
        # Example implementation with transformers:
        # inputs = self.tokenizer(text, return_tensors="pt")
        # outputs = self.model.generate(**inputs)
        # corrected_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # return corrected_text
        
        # Simple simulation of stutter correction
        # Just for demonstration purposes - in a real app you would use ML models
        words = text.split()
        corrected_words = []
        
        for i, word in enumerate(words):
            # Simulate correction of repetition stutters (e.g., "I-I-I am" -> "I am")
            if i > 0 and word == words[i-1]:
                continue
                
            # Simulate correction of letter repetitions (e.g., "wwwhat" -> "what")
            if len(word) > 3:
                chars = list(word)
                i = 1
                while i < len(chars):
                    if chars[i] == chars[i-1]:
                        chars.pop(i)
                    else:
                        i += 1
                word = ''.join(chars)
                
            corrected_words.append(word)
            
        return ' '.join(corrected_words)
    
    def autocomplete_text(self, partial_text, stutter_info=None):
        """
        Provide autocomplete suggestions based on partial text
        
        Parameters:
        partial_text: The partial text to complete
        stutter_info: Information about detected stutters (optional)
        
        Returns:
        List of autocomplete suggestions
        """
        # TODO: Implement autocomplete functionality
        # Example with a language model:
        # 1. Tokenize the partial text
        # 2. Get predictions for next tokens
        # 3. Return top k suggestions
        
        # Placeholder implementation
        if not partial_text:
            return []
            
        # Simple demo suggestions
        last_word = partial_text.split()[-1] if partial_text.split() else ""
        
        # Provide suggestions based on common words
        common_words = ["the", "and", "to", "of", "in", "that", "have", "with", "you", "for"]
        suggestions = [word for word in common_words if word.startswith(last_word.lower())]
        
        # If no matches, provide some default suggestions
        if not suggestions:
            suggestions = ["the", "and", "to"]
            
        return suggestions[:3]  # Return top 3 suggestions

class SpeechToTextService:
    """
    Service for converting speech to text
    In a real implementation, this would use a speech recognition API or model
    """
    def __init__(self):
        logger.info("SpeechToTextService initialized")
        
        # TODO: Initialize your speech recognition model or API client
        # Example with Google Cloud Speech-to-Text:
        # from google.cloud import speech
        # self.client = speech.SpeechClient()
        
    def transcribe(self, audio_data, sample_rate=16000):
        """
        Transcribe audio data to text
        
        Parameters:
        audio_data: Audio data as numpy array
        sample_rate: Sample rate of the audio
        
        Returns:
        Transcribed text
        """
        # TODO: Implement actual transcription with a model or API
        # Example with Google Cloud Speech:
        # audio = speech.RecognitionAudio(content=audio_data.tobytes())
        # config = speech.RecognitionConfig(
        #     encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        #     sample_rate_hertz=sample_rate,
        #     language_code="en-US",
        # )
        # response = self.client.recognize(config=config, audio=audio)
        # return response.results[0].alternatives[0].transcript
        
        # Placeholder implementation - in a real app, you would use a speech recognition service
        # Just returning placeholder text for demonstration
        return "This is a placeholder for transcribed speech"

# Initialize the models and services
stutter_detector = StutterDetector()
text_corrector = TextCorrector()
speech_to_text = SpeechToTextService()

# ------------------------
# SESSION MANAGEMENT
# ------------------------

def create_session():
    """Create a new session for a user"""
    session_id = str(uuid.uuid4())
    user_sessions[session_id] = {
        "created_at": time.time(),
        "audio_buffer": deque(maxlen=BUFFER_SIZE),
        "last_activity": time.time(),
        "stutter_stats": {
            "total_stutters": 0,
            "stutter_types": {"block": 0, "repetition": 0, "prolongation": 0}
        }
    }
    return session_id

def update_session_activity(session_id):
    """Update the last activity timestamp for a session"""
    if session_id in user_sessions:
        user_sessions[session_id]["last_activity"] = time.time()

def cleanup_sessions(max_idle_time=1800):  # 30 minutes
    """Clean up inactive sessions"""
    current_time = time.time()
    to_remove = []
    
    for session_id, session_data in user_sessions.items():
        if current_time - session_data["last_activity"] > max_idle_time:
            to_remove.append(session_id)
            
    for session_id in to_remove:
        del user_sessions[session_id]
        
    return len(to_remove)

# Schedule regular session cleanup
def schedule_cleanup():
    """Schedule regular cleanup of inactive sessions"""
    threading.Timer(300, schedule_cleanup).start()  # Run every 5 minutes
    removed = cleanup_sessions()
    if removed > 0:
        logger.info(f"Cleaned up {removed} inactive sessions")

# Start the session cleanup scheduler
schedule_cleanup()

# ------------------------
# API ROUTES
# ------------------------

@app.route('/')
def index():
    """Serve the main application page"""
    return render_template('index.html')  # Make sure this points to the correct template

@app.route('/api/process-speech', methods=['POST'])
def process_speech():
    """
    Process speech text and return corrected version
    
    Expects JSON input with 'speech' field containing the transcribed text
    Returns JSON with 'corrected_text' and 'analysis' fields
    """
    try:
        data = request.json
        speech_text = data.get('speech', '')
        
        if not speech_text:
            return jsonify({"error": "No speech text provided"}), 400
            
        # Analyze speech for stutters (in a real implementation, you might use audio instead)
        stutter_analysis = stutter_detector.process_audio(None)  # Using None as placeholder
        
        # Correct the text
        corrected_text = text_corrector.correct_text(speech_text, stutter_analysis)
        
        return jsonify({
            "original_text": speech_text,
            "corrected_text": corrected_text,
            "analysis": stutter_analysis
        })
        
    except Exception as e:
        logger.error(f"Error processing speech: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/stream-audio', methods=['POST'])
def stream_audio():
    """
    Receive streaming audio data
    
    This endpoint would receive chunks of audio data from the client
    and process them in real-time for stutter detection
    """
    try:
        # Get session ID or create a new one
        session_id = request.headers.get('X-Session-ID')
        if not session_id or session_id not in user_sessions:
            session_id = create_session()
            
        # Get audio data from request
        audio_data = request.data
        
        # Process the audio chunk in a background thread to avoid blocking
        threading.Thread(target=process_audio_chunk, args=(session_id, audio_data)).start()
        
        return jsonify({"status": "processing", "session_id": session_id})
        
    except Exception as e:
        logger.error(f"Error streaming audio: {str(e)}")
        return jsonify({"error": str(e)}), 500

def process_audio_chunk(session_id, audio_data):
    """
    Process an audio chunk in the background
    
    This would be called by the streaming endpoint to process audio chunks
    without blocking the response
    """
    try:
        # Update session activity
        update_session_activity(session_id)
        
        # Convert audio data to numpy array
        # In a real implementation, you would handle different audio formats
        # This assumes the audio is in the correct format already
        # TODO: Implement proper audio conversion based on your frontend's format
        # Example with soundfile:
        # with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_file:
        #     temp_file.write(audio_data)
        #     temp_file.flush()
        #     audio_np, sample_rate = sf.read(temp_file.name)
        
        # For this demo, we'll create a simple placeholder
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        
        # Add to session buffer
        user_sessions[session_id]["audio_buffer"].append(audio_np)
        
        # Check if we have enough data to process
        if len(user_sessions[session_id]["audio_buffer"]) >= BUFFER_SIZE:
            # Combine audio chunks
            combined_audio = np.concatenate(list(user_sessions[session_id]["audio_buffer"]))
            
            # Process the combined audio
            # 1. Detect stutters
            stutter_results = stutter_detector.process_audio(combined_audio)
            
            # 2. Transcribe speech (in a real implementation)
            transcribed_text = speech_to_text.transcribe(combined_audio)
            
            # 3. Correct text if stutters detected
            corrected_text = text_corrector.correct_text(transcribed_text, stutter_results)
            
            # 4. Update stutter statistics
            if stutter_results.get("has_stutter", False):
                user_sessions[session_id]["stutter_stats"]["total_stutters"] += 1
                stutter_type = stutter_results.get("stutter_type", "none")
                if stutter_type in user_sessions[session_id]["stutter_stats"]["stutter_types"]:
                    user_sessions[session_id]["stutter_stats"]["stutter_types"][stutter_type] += 1
            
            # 5. Send results to client via WebSocket
            result_data = {
                "session_id": session_id,
                "transcribed_text": transcribed_text,
                "corrected_text": corrected_text,
                "stutter_results": stutter_results,
                "stutter_stats": user_sessions[session_id]["stutter_stats"]
            }
            
            socketio.emit('stutter_results', result_data, room=session_id)
            
            logger.info(f"Processed audio chunk for session {session_id}: {len(audio_data)} bytes")
        
    except Exception as e:
        logger.error(f"Error processing audio chunk: {str(e)}")
        socketio.emit('error', {"error": str(e)}, room=session_id)

@app.route('/api/autocomplete', methods=['POST'])
def autocomplete():
    """
    Provide autocomplete suggestions for partial text
    
    Expects JSON input with 'partial_text' field
    Returns JSON with 'suggestions' field containing a list of possible completions
    """
    try:
        data = request.json
        partial_text = data.get('partial_text', '')
        session_id = request.headers.get('X-Session-ID')
        
        # Update session activity if valid session
        if session_id in user_sessions:
            update_session_activity(session_id)
        
        # Get autocomplete suggestions
        suggestions = text_corrector.autocomplete_text(partial_text)
        
        return jsonify({
            "partial_text": partial_text,
            "suggestions": suggestions
        })
        
    except Exception as e:
        logger.error(f"Error getting autocomplete suggestions: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/session-stats', methods=['GET'])
def session_stats():
    """
    Get statistics for a user session
    
    Expects session_id as query parameter
    Returns JSON with session statistics
    """
    try:
        session_id = request.args.get('session_id')
        
        if not session_id or session_id not in user_sessions:
            return jsonify({"error": "Invalid or expired session"}), 404
            
        # Update session activity
        update_session_activity(session_id)
        
        # Return session statistics
        return jsonify({
            "session_id": session_id,
            "created_at": user_sessions[session_id]["created_at"],
            "stutter_stats": user_sessions[session_id]["stutter_stats"]
        })
        
    except Exception as e:
        logger.error(f"Error getting session stats: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/save-recording', methods=['POST'])
def save_recording():
    """
    Save a complete recording for later analysis
    
    Expects multipart form with 'audio_file' field containing the audio file
    Returns JSON with the saved file info
    """
    try:
        if 'audio_file' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
            
        audio_file = request.files['audio_file']
        session_id = request.form.get('session_id')
        
        if not session_id:
            session_id = create_session()
        elif session_id in user_sessions:
            update_session_activity(session_id)
            
        # Create a directory for recordings if it doesn't exist
        recordings_dir = os.path.join(tempfile.gettempdir(), 'stutter_recordings')
        os.makedirs(recordings_dir, exist_ok=True)
        
        # Save the file with a unique name
        filename = f"{session_id}_{int(time.time())}.wav"
        file_path = os.path.join(recordings_dir, filename)
        audio_file.save(file_path)
        
        logger.info(f"Saved recording to {file_path}")
        
        # Process the complete recording in the background
        threading.Thread(target=process_complete_recording, args=(session_id, file_path)).start()
        
        return jsonify({
            "session_id": session_id,
            "filename": filename,
            "status": "processing"
        })
        
    except Exception as e:
        logger.error(f"Error saving recording: {str(e)}")
        return jsonify({"error": str(e)}), 500

def process_complete_recording(session_id, file_path):
    """
    Process a complete recording for comprehensive analysis
    
    This would be used for detailed analysis of longer recordings
    """
    try:
        # Load the audio file
        audio_data, sample_rate = sf.read(file_path)
        
        # In a real implementation, you would:
        # 1. Run the complete audio through your stutter detection model
        # 2. Transcribe the complete audio
        # 3. Perform detailed analysis on stuttering patterns
        # 4. Save the results to the user's session or database
        
        # For this demo, we'll just update the session with placeholder results
        if session_id in user_sessions:
            update_session_activity(session_id)
            
            # Generate placeholder results
            full_analysis = {
                "recording_duration": len(audio_data) / sample_rate,
                "stutter_count": np.random.randint(5, 20),
                "stutter_types": {
                    "block": np.random.randint(1, 10),
                    "repetition": np.random.randint(1, 10),
                    "prolongation": np.random.randint(1, 5)
                },
                "fluency_score": np.random.uniform(60, 95)
            }
            
            # Send results to client
            socketio.emit('complete_analysis', {
                "session_id": session_id,
                "analysis": full_analysis,
                "file_path": file_path
            }, room=session_id)
            
            logger.info(f"Processed complete recording for session {session_id}")
        
    except Exception as e:
        logger.error(f"Error processing complete recording: {str(e)}")
        if session_id in user_sessions:
            socketio.emit('error', {"error": str(e)}, room=session_id)

# ------------------------
# WEBSOCKET ROUTES
# ------------------------

@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    logger.info(f"Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection"""
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('join')
def handle_join(data):
    """
    Handle client joining a session room
    
    This allows for real-time updates to be sent to specific clients
    """
    session_id = data.get('session_id')
    
    if not session_id:
        # Create a new session if none provided
        session_id = create_session()
        socketio.emit('session_created', {"session_id": session_id}, room=request.sid)
    elif session_id not in user_sessions:
        # Create a new session if the provided one doesn't exist
        session_id = create_session()
        socketio.emit('session_created', {"session_id": session_id}, room=request.sid)
    else:
        # Update activity for existing session
        update_session_activity(session_id)
    
    # Join the room for this session
    socketio.join_room(session_id)
    logger.info(f"Client {request.sid} joined session {session_id}")

@socketio.on('leave')
def handle_leave(data):
    """Handle client leaving a session room"""
    session_id = data.get('session_id')
    
    if session_id:
        socketio.leave_room(session_id)
        logger.info(f"Client {request.sid} left session {session_id}")

@socketio.on('stream_audio')
def handle_streaming_audio(data):
    """
    Handle streaming audio data via WebSockets
    
    This is an alternative to the HTTP endpoint for streaming audio
    """
    session_id = data.get('session_id')
    audio_data = data.get('audio_data')
    
    if not session_id or not audio_data:
        socketio.emit('error', {"error": "Missing session_id or audio_data"}, room=request.sid)
        return
    
    # Create session if it doesn't exist
    if session_id not in user_sessions:
        session_id = create_session()
        socketio.emit('session_created', {"session_id": session_id}, room=request.sid)
    
    # Process the audio chunk
    threading.Thread(target=process_audio_chunk, args=(session_id, audio_data)).start()
    
    # Acknowledge receipt
    socketio.emit('audio_received', {"session_id": session_id}, room=request.sid)

# ------------------------
# MAIN APPLICATION ENTRY
# ------------------------

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=True)