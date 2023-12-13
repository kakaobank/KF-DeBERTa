from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("kakaobank/kf-deberta-base")
tokenizer = AutoTokenizer.from_pretrained("kakaobank/kf-deberta-base")

text = "카카오뱅크와 에프엔가이드가 금융특화 언어모델을 공개합니다."
tokens = tokenizer.tokenize(text)
print(tokens)

inputs = tokenizer(text, return_tensors="pt")
model_output = model(**inputs)
print(model_output)
