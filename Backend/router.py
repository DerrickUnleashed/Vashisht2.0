from flask import Blueprint, request, jsonify
from transformers import pipeline

router = Blueprint('router', __name__)

@router.route('/api/process-speech', methods=['POST'])
def process_speech():
    data = request.get_json()
    speech = data.get('speech', '')
    pipe = pipeline("fill-mask", model="./local_bert", tokenizer="./local_bert")

    result1 = pipe(speech + " [MASK]")
    top1_words = [i['token_str'] for i in result1[:5] if i['token_str'].isalpha()]

    response = {
        "top1_words": top1_words,
        "combinations": {}
    }

    for word1 in top1_words:
        result2 = pipe(speech + " " + word1 + " [MASK]")
        top2_words = [i['token_str'] for i in result2[:5] if i['token_str'].isalpha()]
        response["combinations"][word1] = top2_words

    return jsonify(response)
