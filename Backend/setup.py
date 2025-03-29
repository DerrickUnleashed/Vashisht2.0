from transformers import AutoModelForMaskedLM, AutoTokenizer

model_name = "google/bert_uncased_L-12_H-768_A-12"
model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained("./local_bert")
tokenizer.save_pretrained("./local_bert")
