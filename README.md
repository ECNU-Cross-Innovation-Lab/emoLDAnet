# emoLDAnet

The files are orgnized as follows
```
emoLDAnet
├── DL (Deep Learning)
│   ├── FER
│   ├── mtcnn
│   ├── main.py
│   ├── utils.py
│   └── video_process.py
├── ML (Machine Learning)
│   ├── ML.py
│   └── utils.py
├── R
├── Rawdata
│   ├── AI
│   └── Questionnaire
└── Result Data
    ├── AI
    └── Correlation

```


## Code

1. MTCNN face detection network is in [MTCNN](./DL/mtcnn/).
2. Facial Expression Recognition based on deep learning is in [FER](./DL/FER/). The emotion dataset for training can be obtained [here](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data).
3. OCC-PAD-LDA modeling based on Psychology is in [utils.py](./DL/utils.py).
4. Selection of classifier for machine learning is in [ML.py](./ML/ML.py).

## Experiment Results

1. The accuracy, recall, and F1 score of AI predictions are in [Result Data/AI](./Result%20Data/AI/).
2. The correlation of AI and psychological qustionnaire for SVM, Tree and Forest based on results of five folds is in [Result Data/Correlation](./Result%20Data/Correlation/).
3. The emotional OCC data predicted by deep learning module is in [Rawdata/AI](./Rawdata/AI/). The file is in the format of *"abnormal emotion type/fold.csv"*.
4. The Questionnaire data of subjects is in [Rawdata/Questionnaire](./Rawdata/Questionnaire/).

