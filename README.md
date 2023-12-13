 # KF-DeBERTa
카카오뱅크 & 에프엔가이드에서 학습한 금융 도메인 특화 언어모델을 공개합니다.  

- [[huggingface][kakaobank/kf-deberta-base]](https://huggingface.co/kakaobank/kf-deberta-base)

## Model description
* KF-DeBERTa는 범용 도메인 말뭉치와 금융 도메인 말뭉치를 함께 학습한 언어모델 입니다.
* 모델 아키텍쳐는 [DeBERTa-v2](https://github.com/microsoft/DeBERTa#whats-new-in-v2)를 기반으로 학습하였습니다.
  * ELECTRA의 RTD를 training objective로 사용한 DeBERTa-v3는 일부 task(KLUE-RE, WoS, Retrieval)에서 상당히 낮은 성능을 확인하여 최종 아키텍쳐는 DeBERTa-v2로 결정하였습니다.
* 범용 도메인 및 금융 도메인 downstream task에서 모두 우수한 성능을 확인하였습니다.
  * 금융 도메인 downstream task의 철저한 성능검증을 위해 다양한 데이터셋을 통해 검증을 수행하였습니다.
  * 범용 도메인 및 금융 도메인에서 기존 언어모델보다 더 나은 성능을 보여줬으며 특히 KLUE Benchmark에서는 RoBERTa-Large보다 더 나은 성능을 확인하였습니다.

## Usage
```python3
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("kakaobank/kf-deberta-base")
tokenizer = AutoTokenizer.from_pretrained("kakaobank/kf-deberta-base")

text = "카카오뱅크와 에프엔가이드가 금융특화 언어모델을 공개합니다."
tokens = tokenizer.tokenize(text)
print(tokens)

inputs = tokenizer(text, return_tensors="pt")
model_output = model(**inputs)
print(model_output)
```

## Benchmark
* 모든 task는 아래와 같은 기본적인 hyperparameter search만 수행하였습니다.
  * batch size: {16, 32}
  * learning_rate: {1e-5, 3e-5, 5e-5}
  * weight_decay: {0, 0.01}
  * warmup_proportion: {0, 0.1}

**KLUE Benchmark**

|        Model         |       YNAT       |        KLUE-ST         |   KLUE-NLI   |             KLUE-NER              |            KLUE-RE            |        KLUE-DP         |         KLUE-MRC          |          WoS           |       AVG        |
|:--------------------:|:----------------:|:----------------------:|:------------:|:---------------------------------:|:-----------------------------:|:----------------------:|:-------------------------:|:----------------------:|:----------------:|
|                      |        F1        |      Pearsonr/F1       |     ACC      |         F1-Entity/F1-Char         |         F1-micro/AUC          |        UAS/LAS         |         EM/ROUGE          |        JGA/F1-S        |                  | 
|     mBERT (Base)     |      82.64       |      82.97/75.93       |    72.90     |            75.56/88.81            |          58.39/56.41          |      88.53/86.04       |        49.96/55.57        |      35.27/88.60       |      71.26       |
|     XLM-R (Base)     |      84.52       |      88.88/81.20       |    78.23     |            80.48/92.14            |          57.62/57.05          |      93.12/87.23       |        26.76/53.36        |      41.54/89.81       |      72.28       |
|    XLM-R (Large)     |      87.30       |      93.08/87.17       |    86.40     |            82.18/93.20            |          58.75/63.53          |      92.87/87.82       |        35.23/66.55        |      42.44/89.88       |      76.17       |
|    KR-BERT (Base)    |      85.36       |      87.50/77.92       |    77.10     |            74.97/90.46            |          62.83/65.42          |      92.87/87.13       |        48.95/58.38        |      45.60/90.82       |      74.67       |
|   KoELECTRA (Base)   |      85.99       |      93.14/85.89       |    86.87     |            86.06/92.75            |          62.67/57.46          |      90.93/87.07       |        59.54/65.64        |      39.83/88.91       |      77.34       |
|   KLUE-BERT (Base)   |      86.95       |      91.01/83.44       |    79.87     |            83.71/91.17            |          65.58/68.11          |      93.07/87.25       |        62.42/68.15        |      46.72/91.59       |      78.50       |
| KLUE-RoBERTa (Small) |      85.95       |      91.70/85.42       |    81.00     |            83.55/91.20            |          61.26/60.89          |      93.47/87.50       |        58.25/63.56        |      46.65/91.50       |      77.28       |
| KLUE-RoBERTa (Base)  |      86.19       |      92.91/86.78       |    86.30     |            83.81/91.09            |          66.73/68.11          |      93.75/87.77       |        69.56/74.64        |      47.41/91.60       |      80.48       |
| KLUE-RoBERTa (Large) |      85.88       |      93.20/86.13       |  **89.50**   |            84.54/91.45            |        **71.06**/73.33        |      93.84/87.93       |    **75.26**/**80.30**    |      49.39/92.19       |      82.43       |
|  KF-DeBERTa (Base)   | **<u>87.51</u>** | **<u>93.24/87.73</u>** | <u>88.37</u> | **<u>89.17</u>**/**<u>93.30</u>** | <u>69.70</u>/**<u>75.07</u>** | **<u>94.05/87.97</u>** | <u>72.59</u>/<u>78.08</u> | **<u>50.21/92.59</u>** | **<u>82.83</u>** |

* 굵은글씨는 모든 모델중 가장높은 점수이며, 밑줄은 base 모델 중 가장 높은 점수입니다.

**금융도메인 벤치마크**
|        Model        | FN-Sentiment (v1) | FN-Sentiment (v2) | FN-Adnews |  FN-NER   |  KorFPB   | KorFiQA-SA | KorHeadline | Avg (FiQA-SA 제외)  |
|:-------------------:|:-----------------:|:-----------------:|:---------:|:---------:|:---------:|:----------:|:-----------:|:-----------------:|
|                     |        ACC        |        ACC        |    ACC    | F1-micro  |    ACC    |    MSE     |   Mean F1   |                   |
| KLUE-RoBERTa (Base) |       98.26       |       91.21       |   96.34   |   90.31   |   90.97   |   0.0589   |    81.11    |       94.03       |
|  KoELECTRA (Base)   |       98.26       |       90.56       |   96.98   |   89.81   |   92.36   |   0.0652   |    80.69    |       93.90       |
|  KF-DeBERTa (Base)  |     **99.36**     |     **92.29**     | **97.63** | **91.80** | **93.47** | **0.0553** |  **82.12**  |     **95.27**     |

* **FN-Sentiment**: 금융도메인 감성분석
* **FN-Adnews**: 금융도메인 광고성기사 분류
* **FN-NER**: 금융도메인 개체명인식
* **KorFPB**: FinancialPhraseBank 번역데이터
  * Cite: ```Malo, Pekka, et al. "Good debt or bad debt: Detecting semantic orientations in economic texts." Journal of the Association for Information Science and Technology 65.4 (2014): 782-796.```
* **KorFiQA-SA**: FiQA-SA 번역데이터
  * Cite: ```Maia, Macedo & Handschuh, Siegfried & Freitas, Andre & Davis, Brian & McDermott, Ross & Zarrouk, Manel & Balahur, Alexandra. (2018). WWW'18 Open Challenge: Financial Opinion Mining and Question Answering. WWW '18: Companion Proceedings of the The Web Conference 2018. 1941-1942. 10.1145/3184558.3192301.``` 
* **KorHeadline**: Gold Commodity News and Dimensions 번역데이터
  * Cite: ```Sinha, A., & Khandait, T. (2021, April). Impact of News on the Commodity Market: Dataset and Results. In 
    Future of Information and Communication Conference (pp. 589-601). Springer, Cham.```


**범용도메인 벤치마크**
|        Model        |   NSMC    |   PAWS    |  KorNLI   |  KorSTS   |     KorQuAD     | Avg (KorQuAD 제외) |
|:-------------------:|:---------:|:---------:|:---------:|:---------:|:---------------:|:----------------:|
|                     |    ACC    |    ACC    |    ACC    | spearman  |      EM/F1      |                  |
| KLUE-RoBERTa (Base) |   90.47   |   84.79   |   81.65   |   84.40   |   86.34/94.40   |      85.33       |
|  KoELECTRA (Base)   |   90.63   |   84.45   |   82.24   |   85.53   |   84.83/93.45   |      85.71       |
|  KF-DeBERTa (Base)  | **91.36** | **86.14** | **84.54** | **85.99** | **86.60/95.07** |    **87.01**     |



## License
KF-DeBERTa의 소스코드 및 모델은 MIT 라이선스 하에 공개되어 있습니다.  
라이선스 전문은 [MIT 파일](LICENSE)에서 확인할 수 있습니다.  
모델의 사용으로 인해 발생한 어떠한 손해에 대해서도 당사는 책임을 지지 않습니다.  

## Citation
```
@proceedings{jeon-etal-2023-kfdeberta,
  title         = {KF-DeBERTa: Financial Domain-specific Pre-trained Language Model},
  author        = {Eunkwang Jeon, Jungdae Kim, Minsang Song, and Joohyun Ryu},
  booktitle     = {Proceedings of the 35th Annual Conference on Human and Cognitive Language Technology},
  moth          = {oct},
  year          = {2023},
  publisher     = {Korean Institute of Information Scientists and Engineers},
  url           = {http://www.hclt.kr/symp/?lnb=conference},
  pages         = {143--148},
}
```
