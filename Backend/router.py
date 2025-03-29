from flask import Blueprint, request, jsonify
from fetcher import fetch_corrected_text

router = Blueprint('router', __name__)

@router.route('/api/process-speech', methods=['POST'])
def process_speech():
    data = request.get_json()
    speech = data.get('speech', '')
    
    # Fetch corrected text
    corrected_text = fetch_corrected_text(speech)
    
    return jsonify({'corrected_text': corrected_text})
