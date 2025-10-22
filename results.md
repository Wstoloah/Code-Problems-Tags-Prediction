# ==== Model Comparison Report ====

_Created on 2025-10-22 20:47:53_

| Date | Model | Embedding | Classifier | Dataset | Precision | Recall | F1 | Hamming | Notes |
|------|--------|------------|-------------|----------|-----------|--------|----|----------|--------|
| 2025-10-22 20:47 | BaselineTagPredictor | TF-IDF | RandomForestClassifier | val | 0.7795 | 0.3916 | 0.4706 | 0.0905 | basline model with random forest |
| 2025-10-22 20:53 | BaselineTagPredictor | TF-IDF | LogisticRegression | val | 0.7078 | 0.4484 | 0.5076 | 0.0854 | basline model with logistic regression |
| 2025-10-22 20:55 | BaselineTagPredictor | TF-IDF | XGBClassifier | val | 0.7758 | 0.6812 | 0.7174 | 0.0826 | basline model with xgboost |
