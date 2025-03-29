from transformers import pipeline
pipe = pipeline("fill-mask", model="./local_bert", tokenizer="./local_bert")
sentence = "The weather is really nice"
result = pipe(sentence + " [MASK]")
print(f"{result[0]['token_str']}/{result[1]['token_str']}/{result[2]['token_str']}/{result[3]['token_str']}/{result[4]['token_str']}")