# emoLDAnet

## Code

1. MTCNN face detection network is in [MTCNN](./DL/mtcnn/).
2. Facial Expression Recognition based on deep learning is in [FER](./DL/FER/). The emotion dataset for training can be obtained [here](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data).
3. OCC-PAD-LDA modeling based on Psychology is in [utils.py](./DL/utils.py).
4. Selection of classifier for machine learning is in [ML.py](./ML/ML.py).

## Experiment Results

1. The correlation of AI and psychological qustionnaire for SVM, Tree and Forest based on results of five folds is in 
[Statistical source data of Corelation analysis between AI - Psy. questionnaire](https://github.com/ECNU-Cross-Innovation-Lab/emoLDAnet/tree/main/Statistical%20source%20data%20of%20Corelation%20analysis%20between%20AI%20-%20Psy.%20questionnaire).  
The file names are in the format of *"classfier-fold_kcor.csv"*.

2. The accuracy, recall, and F1 score of AI predictions are in [Statistical source data of AI analysis/25 times of LDA](https://github.com/ECNU-Cross-Innovation-Lab/emoLDAnet/tree/main/Statistical%20source%20data%20of%20AI%20analysis/25%20times%20of%20LDA).  
The file directory is in the format of *"abnormal emotion type/seed/fold.csv"*.

