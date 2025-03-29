from flask import Blueprint, request, jsonify
from transformers import pipeline

router = Blueprint('router', __name__)

@router.route('/api/process-speech', methods=['POST'])
def process_speech():
    data = request.get_json()
    speech = data.get('speech', '')
    pipe = pipeline("fill-mask", model="./local_bert", tokenizer="./local_bert")
    result = pipe(speech + " [MASK]")
    corrected_text=''
    for i in result:
        if i['token_str'].isalpha():
            corrected_text += '['
            corrected_text += i['token_str']
            corrected_text += ']    '
    print(corrected_text)  # Print to terminal
    return jsonify({'corrected_text': corrected_text})