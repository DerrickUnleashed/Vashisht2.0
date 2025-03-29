from transformers import pipeline
pipe = pipeline("fill-mask", model="./local_bert", tokenizer="./local_bert")
sentence = "the weather is"
result = pipe(sentence + " [MASK]")
for i in result:
    if i['token_str'].isalpha():
        print(i['token_str'])