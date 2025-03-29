'''from flask import Blueprint, request, jsonify
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
    return jsonify({'corrected_text': corrected_text})'''

from flask import Blueprint, request, jsonify
from transformers import pipeline

router = Blueprint('router', __name__)

@router.route('/api/process-speech', methods=['POST'])
def process_speech():
    data = request.get_json()
    speech = data.get('speech', '')
    pipe = pipeline("fill-mask", model="./local_bert", tokenizer="./local_bert")
    
    # Predict top 5 single words
    result1 = pipe(speech + " [MASK]")
    top1_words = [i['token_str'] for i in result1[:5] if i['token_str'].isalpha()]

    response_text = "Single Predictions: \n"
    response_text += " | ".join(f"[{word}]" for word in top1_words) + "\n\n"

    response_text += "Double Predictions: \n"
    combinations = ""

    # For each predicted first word, predict top 5 words for the second masked word
    for word1 in top1_words:
        result2 = pipe(speech + " " + word1 + " [MASK]")
        top2_words = [i['token_str'] for i in result2[:5] if i['token_str'].isalpha()]
        
        # Concatenate combinations of the first and second word
        for word2 in top2_words:
            combinations += f"[{word1}][{word2}]      /        \n"
    
    response_text += combinations
    print(response_text)  # Print to terminal
    return jsonify({'corrected_text': response_text})
