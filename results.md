# ==== Model Comparison Report ==== 
_Created on 2025-10-22 20:47:53_

| Date | Model | Embedding | Classifier | Dataset | Precision | Recall | F1 | Hamming | Notes |
|------|--------|------------|-------------|----------|-----------|--------|----|----------|--------|
| 2025-10-23 12:36 | BaselineTagPredictor | COUNT | LogisticRegression | val | 0.6022 | 0.3758 | 0.4524 | 0.0769 | baseline model : countVectorizer + problem descriptions using OVR logistic regression |
| 2025-10-23 12:40 | BaselineTagPredictor | TFIDF | LogisticRegression | val | 0.6579 | 0.1899 | 0.2865 | 0.0744 | baseline model : TfidfVectorizer + problem descriptions using OVR logistic regression |
| 2025-10-23 12:42 | BaselineTagPredictor | COUNT | RandomForestClassifier | val | 0.6283 | 0.1244 | 0.2005 | 0.0825 | baseline model : CountVectorizer + problem descriptions using OVR random forest |
| 2025-10-23 12:43 | BaselineTagPredictor | COUNT | XGBClassifier | val | 0.6493 | 0.4314 | 0.5136 | 0.0691 | baseline model : CountVectorizer + problem descriptions using OVR xgboost |
| 2025-10-23 12:45 | BaselineTagPredictor | COUNT+CODE | XGBClassifier | val | 0.6731 | 0.4778 | 0.5526 | 0.0663 | baseline model : CountVectorizer + problem descriptions + code using OVR xgboost |
| 2025-10-23 12:49 | BaselineTagPredictor | TFIDF | XGBClassifier | val | 0.6712 | 0.4501 | 0.5344 | 0.0696 | baseline model : TfidfVectorizer + problem descriptions using OVR xgboost |
| 2025-10-23 12:52 | BaselineTagPredictor | TFIDF+CODE | XGBClassifier | val | 0.6552 | 0.4455 | 0.5279 | 0.0671 | baseline model : TfidfVectorizer + problem descriptions + code using OVR xgboost |
| 2025-10-23 12:59 | BaselineTagPredictor | COUNT+CODE | XGBClassifier | val | 0.6425 | 0.4715 | 0.5397 | 0.0673 | baseline model : CountVectorizer + problem descriptions + code + stats features using OVR xgboost |
| 2025-10-23 15:16 | BaselineTagPredictor | COUNT | XGBClassifier | test | 0.7579 | 0.4822 | 0.5838 | 0.0664 | tuned ovr xgboost model using CountVectorizer on description + code data |
| 2025-10-23 15:17 | BaselineTagPredictor | COUNT | XGBClassifier | test | 0.7315 | 0.5182 | 0.6005 | 0.0669 | tuned ovr xgboost model using CountVectorizer on description + code data, thresh = 0.4 |
| 2025-10-23 15:17 | BaselineTagPredictor | COUNT | XGBClassifier | test | 0.6884 | 0.5696 | 0.6166 | 0.0693 | tuned ovr xgboost model using CountVectorizer on description + code data, thresh = 0.3 |
| 2025-10-23 15:18 | BaselineTagPredictor | COUNT | XGBClassifier | test | 0.7462 | 0.4973 | 0.5909 | 0.0665 | tuned ovr xgboost model using CountVectorizer on description + code data, thresh = 0.45 |
| 2025-10-23 15:18 | BaselineTagPredictor | COUNT | XGBClassifier | test | 0.7588 | 0.4835 | 0.5851 | 0.0660 | tuned ovr xgboost model using CountVectorizer on description + code data, thresh = 0.49 |
| 2025-10-23 15:19 | BaselineTagPredictor | COUNT | XGBClassifier | test | 0.7510 | 0.4933 | 0.5899 | 0.0662 | tuned ovr xgboost model using CountVectorizer on description + code data, thresh = 0.47 |
| 2025-10-23 15:19 | BaselineTagPredictor | COUNT | XGBClassifier | test | 0.7506 | 0.4835 | 0.5829 | 0.0663 | tuned ovr xgboost model using CountVectorizer on description + code data, thresh = 0.48 |
