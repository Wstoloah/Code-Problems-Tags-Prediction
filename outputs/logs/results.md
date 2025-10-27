# Model Comparison Report

_Created on 2025-10-27 23:59:55_

## Global Metrics

| Date | Model | ID | Embedding | Use code | Use stats features | Classifier | Dataset | Precision | Recall | F1 | Hamming | Notes |
|------|-------|-----|-----------|----------|--------------------|-----------|---------|-----------| -------|----|---------| ------|
| 2025-10-27 23:59 | BaselineTagPredictor | BaselineTagPredictor-001 | COUNT | False | False | LogisticRegression | val | 0.7075 | 0.3890 | 0.4724 | 0.0736 |  |
| 2025-10-28 00:00 | BaselineTagPredictor | BaselineTagPredictor-002 | COUNT | False | False | RandomForestClassifier | val | 0.0911 | 1.0000 | 0.1563 | 0.9089 |  |
| 2025-10-28 00:00 | BaselineTagPredictor | BaselineTagPredictor-003 | COUNT | False | False | XGBClassifier | val | 0.6694 | 0.4496 | 0.5286 | 0.0701 |  |
| 2025-10-28 00:00 | BaselineTagPredictor | BaselineTagPredictor-004 | COUNT | False | False | MultinomialNB | val | 0.4835 | 0.5497 | 0.5138 | 0.0812 |  |
| 2025-10-28 00:00 | BaselineTagPredictor | BaselineTagPredictor-005 | COUNT | False | False | MLPClassifier | val | 0.5346 | 0.3618 | 0.4257 | 0.0777 |  |
| 2025-10-28 00:01 | BaselineTagPredictor | BaselineTagPredictor-006 | TFIDF | False | False | LogisticRegression | val | 0.6142 | 0.2597 | 0.3434 | 0.0703 |  |
| 2025-10-28 00:01 | BaselineTagPredictor | BaselineTagPredictor-007 | TFIDF | False | False | RandomForestClassifier | val | 0.0911 | 1.0000 | 0.1563 | 0.9089 |  |
| 2025-10-28 00:01 | BaselineTagPredictor | BaselineTagPredictor-008 | TFIDF | False | False | XGBClassifier | val | 0.5617 | 0.4580 | 0.4921 | 0.0850 |  |
| 2025-10-28 00:01 | BaselineTagPredictor | BaselineTagPredictor-009 | TFIDF | False | False | MultinomialNB | val | 0.2250 | 0.0087 | 0.0167 | 0.0898 |  |
| 2025-10-28 00:02 | BaselineTagPredictor | BaselineTagPredictor-010 | TFIDF | False | False | MLPClassifier | val | 0.5541 | 0.3043 | 0.3836 | 0.0749 |  |
| 2025-10-28 00:03 | BaselineTagPredictor | BaselineTagPredictor-011 | COUNT | True | False | LogisticRegression | val | 0.5354 | 0.3543 | 0.4221 | 0.0721 |  |
| 2025-10-28 00:04 | BaselineTagPredictor | BaselineTagPredictor-012 | COUNT | True | False | RandomForestClassifier | val | 0.0911 | 1.0000 | 0.1563 | 0.9089 |  |
| 2025-10-28 00:04 | BaselineTagPredictor | BaselineTagPredictor-013 | COUNT | True | False | XGBClassifier | val | 0.6826 | 0.4774 | 0.5508 | 0.0668 |  |
| 2025-10-28 00:04 | BaselineTagPredictor | BaselineTagPredictor-014 | COUNT | True | False | MultinomialNB | val | 0.4149 | 0.5604 | 0.4652 | 0.0916 |  |
| 2025-10-28 00:04 | BaselineTagPredictor | BaselineTagPredictor-015 | COUNT | True | False | MLPClassifier | val | 0.5291 | 0.3266 | 0.3981 | 0.0719 |  |
| 2025-10-28 00:05 | BaselineTagPredictor | BaselineTagPredictor-016 | TFIDF | True | False | LogisticRegression | val | 0.6492 | 0.2430 | 0.3322 | 0.0665 |  |
| 2025-10-28 00:05 | BaselineTagPredictor | BaselineTagPredictor-017 | TFIDF | True | False | RandomForestClassifier | val | 0.0911 | 1.0000 | 0.1563 | 0.9089 |  |
| 2025-10-28 00:05 | BaselineTagPredictor | BaselineTagPredictor-018 | TFIDF | True | False | XGBClassifier | val | 0.6593 | 0.4830 | 0.5484 | 0.0703 |  |
| 2025-10-28 00:05 | BaselineTagPredictor | BaselineTagPredictor-019 | TFIDF | True | False | MultinomialNB | val | 0.0625 | 0.0008 | 0.0017 | 0.0924 |  |
| 2025-10-28 00:05 | BaselineTagPredictor | BaselineTagPredictor-020 | TFIDF | True | False | MLPClassifier | val | 0.5671 | 0.2789 | 0.3649 | 0.0731 |  |
| 2025-10-28 00:35 | TunedBaselineTagPredictor | TunedBaselineTagPredictor-001 | COUNT | True | False | XGBClassifier | test | 0.7433 | 0.4882 | 0.5831 | 0.0664 |  |
| 2025-10-28 00:36 | TunedBaselineTagPredictor | TunedBaselineTagPredictor-002 | TFIDF | True | False | XGBClassifier | test | 0.7361 | 0.4313 | 0.5318 | 0.0706 |  |

## Per-tag Metrics

| Model | ID | Notes | Tag | Precision | Recall | F1 | Support | Predicted |
|-------|-----|-------|-----|-----------|--------|----|---------|-----------|
| BaselineTagPredictor | BaselineTagPredictor-001 |  | math | 0.5630 | 0.4497 | 0.5000 | 149 | 119 |
| BaselineTagPredictor | BaselineTagPredictor-001 |  | graphs | 0.5714 | 0.4444 | 0.5000 | 54 | 42 |
| BaselineTagPredictor | BaselineTagPredictor-001 |  | strings | 0.7188 | 0.4894 | 0.5823 | 47 | 32 |
| BaselineTagPredictor | BaselineTagPredictor-001 |  | number theory | 0.9167 | 0.2500 | 0.3929 | 44 | 12 |
| BaselineTagPredictor | BaselineTagPredictor-001 |  | trees | 0.8571 | 0.5854 | 0.6957 | 41 | 28 |
| BaselineTagPredictor | BaselineTagPredictor-001 |  | geometry | 0.3333 | 0.1667 | 0.2222 | 6 | 3 |
| BaselineTagPredictor | BaselineTagPredictor-001 |  | games | 0.7000 | 0.5833 | 0.6364 | 12 | 10 |
| BaselineTagPredictor | BaselineTagPredictor-001 |  | probabilities | 1.0000 | 0.1429 | 0.2500 | 7 | 1 |
| BaselineTagPredictor | BaselineTagPredictor-002 |  | math | 0.3016 | 1.0000 | 0.4635 | 149 | 494 |
| BaselineTagPredictor | BaselineTagPredictor-002 |  | graphs | 0.1093 | 1.0000 | 0.1971 | 54 | 494 |
| BaselineTagPredictor | BaselineTagPredictor-002 |  | strings | 0.0951 | 1.0000 | 0.1738 | 47 | 494 |
| BaselineTagPredictor | BaselineTagPredictor-002 |  | number theory | 0.0891 | 1.0000 | 0.1636 | 44 | 494 |
| BaselineTagPredictor | BaselineTagPredictor-002 |  | trees | 0.0830 | 1.0000 | 0.1533 | 41 | 494 |
| BaselineTagPredictor | BaselineTagPredictor-002 |  | geometry | 0.0121 | 1.0000 | 0.0240 | 6 | 494 |
| BaselineTagPredictor | BaselineTagPredictor-002 |  | games | 0.0243 | 1.0000 | 0.0474 | 12 | 494 |
| BaselineTagPredictor | BaselineTagPredictor-002 |  | probabilities | 0.0142 | 1.0000 | 0.0279 | 7 | 494 |
| BaselineTagPredictor | BaselineTagPredictor-003 |  | math | 0.6528 | 0.3154 | 0.4253 | 149 | 72 |
| BaselineTagPredictor | BaselineTagPredictor-003 |  | graphs | 0.6452 | 0.3704 | 0.4706 | 54 | 31 |
| BaselineTagPredictor | BaselineTagPredictor-003 |  | strings | 0.6667 | 0.5532 | 0.6047 | 47 | 39 |
| BaselineTagPredictor | BaselineTagPredictor-003 |  | number theory | 0.7647 | 0.2955 | 0.4262 | 44 | 17 |
| BaselineTagPredictor | BaselineTagPredictor-003 |  | trees | 0.8667 | 0.6341 | 0.7324 | 41 | 30 |
| BaselineTagPredictor | BaselineTagPredictor-003 |  | geometry | 0.2500 | 0.1667 | 0.2000 | 6 | 4 |
| BaselineTagPredictor | BaselineTagPredictor-003 |  | games | 0.9091 | 0.8333 | 0.8696 | 12 | 11 |
| BaselineTagPredictor | BaselineTagPredictor-003 |  | probabilities | 0.6000 | 0.4286 | 0.5000 | 7 | 5 |
| BaselineTagPredictor | BaselineTagPredictor-004 |  | math | 0.5294 | 0.6040 | 0.5643 | 149 | 170 |
| BaselineTagPredictor | BaselineTagPredictor-004 |  | graphs | 0.5500 | 0.6111 | 0.5789 | 54 | 60 |
| BaselineTagPredictor | BaselineTagPredictor-004 |  | strings | 0.5932 | 0.7447 | 0.6604 | 47 | 59 |
| BaselineTagPredictor | BaselineTagPredictor-004 |  | number theory | 0.4419 | 0.4318 | 0.4368 | 44 | 43 |
| BaselineTagPredictor | BaselineTagPredictor-004 |  | trees | 0.6327 | 0.7561 | 0.6889 | 41 | 49 |
| BaselineTagPredictor | BaselineTagPredictor-004 |  | geometry | 0.4286 | 0.5000 | 0.4615 | 6 | 7 |
| BaselineTagPredictor | BaselineTagPredictor-004 |  | games | 0.6923 | 0.7500 | 0.7200 | 12 | 13 |
| BaselineTagPredictor | BaselineTagPredictor-004 |  | probabilities | 0.0000 | 0.0000 | 0.0000 | 7 | 0 |
| BaselineTagPredictor | BaselineTagPredictor-005 |  | math | 0.5641 | 0.4430 | 0.4962 | 149 | 117 |
| BaselineTagPredictor | BaselineTagPredictor-005 |  | graphs | 0.6053 | 0.4259 | 0.5000 | 54 | 38 |
| BaselineTagPredictor | BaselineTagPredictor-005 |  | strings | 0.6923 | 0.5745 | 0.6279 | 47 | 39 |
| BaselineTagPredictor | BaselineTagPredictor-005 |  | number theory | 0.5882 | 0.2273 | 0.3279 | 44 | 17 |
| BaselineTagPredictor | BaselineTagPredictor-005 |  | trees | 0.7273 | 0.3902 | 0.5079 | 41 | 22 |
| BaselineTagPredictor | BaselineTagPredictor-005 |  | geometry | 0.5000 | 0.3333 | 0.4000 | 6 | 4 |
| BaselineTagPredictor | BaselineTagPredictor-005 |  | games | 0.6000 | 0.5000 | 0.5455 | 12 | 10 |
| BaselineTagPredictor | BaselineTagPredictor-005 |  | probabilities | 0.0000 | 0.0000 | 0.0000 | 7 | 0 |
| BaselineTagPredictor | BaselineTagPredictor-006 |  | math | 0.6750 | 0.3624 | 0.4716 | 149 | 80 |
| BaselineTagPredictor | BaselineTagPredictor-006 |  | graphs | 0.7200 | 0.3333 | 0.4557 | 54 | 25 |
| BaselineTagPredictor | BaselineTagPredictor-006 |  | strings | 0.7436 | 0.6170 | 0.6744 | 47 | 39 |
| BaselineTagPredictor | BaselineTagPredictor-006 |  | number theory | 0.8750 | 0.1591 | 0.2692 | 44 | 8 |
| BaselineTagPredictor | BaselineTagPredictor-006 |  | trees | 0.9000 | 0.4390 | 0.5902 | 41 | 20 |
| BaselineTagPredictor | BaselineTagPredictor-006 |  | geometry | 0.0000 | 0.0000 | 0.0000 | 6 | 0 |
| BaselineTagPredictor | BaselineTagPredictor-006 |  | games | 1.0000 | 0.1667 | 0.2857 | 12 | 2 |
| BaselineTagPredictor | BaselineTagPredictor-006 |  | probabilities | 0.0000 | 0.0000 | 0.0000 | 7 | 0 |
| BaselineTagPredictor | BaselineTagPredictor-007 |  | math | 0.3016 | 1.0000 | 0.4635 | 149 | 494 |
| BaselineTagPredictor | BaselineTagPredictor-007 |  | graphs | 0.1093 | 1.0000 | 0.1971 | 54 | 494 |
| BaselineTagPredictor | BaselineTagPredictor-007 |  | strings | 0.0951 | 1.0000 | 0.1738 | 47 | 494 |
| BaselineTagPredictor | BaselineTagPredictor-007 |  | number theory | 0.0891 | 1.0000 | 0.1636 | 44 | 494 |
| BaselineTagPredictor | BaselineTagPredictor-007 |  | trees | 0.0830 | 1.0000 | 0.1533 | 41 | 494 |
| BaselineTagPredictor | BaselineTagPredictor-007 |  | geometry | 0.0121 | 1.0000 | 0.0240 | 6 | 494 |
| BaselineTagPredictor | BaselineTagPredictor-007 |  | games | 0.0243 | 1.0000 | 0.0474 | 12 | 494 |
| BaselineTagPredictor | BaselineTagPredictor-007 |  | probabilities | 0.0142 | 1.0000 | 0.0279 | 7 | 494 |
| BaselineTagPredictor | BaselineTagPredictor-008 |  | math | 0.4788 | 0.5302 | 0.5032 | 149 | 165 |
| BaselineTagPredictor | BaselineTagPredictor-008 |  | graphs | 0.6190 | 0.2407 | 0.3467 | 54 | 21 |
| BaselineTagPredictor | BaselineTagPredictor-008 |  | strings | 0.5357 | 0.6383 | 0.5825 | 47 | 56 |
| BaselineTagPredictor | BaselineTagPredictor-008 |  | number theory | 0.5938 | 0.4318 | 0.5000 | 44 | 32 |
| BaselineTagPredictor | BaselineTagPredictor-008 |  | trees | 0.7667 | 0.5610 | 0.6479 | 41 | 30 |
| BaselineTagPredictor | BaselineTagPredictor-008 |  | geometry | 0.1111 | 0.1667 | 0.1333 | 6 | 9 |
| BaselineTagPredictor | BaselineTagPredictor-008 |  | games | 0.8889 | 0.6667 | 0.7619 | 12 | 9 |
| BaselineTagPredictor | BaselineTagPredictor-008 |  | probabilities | 0.5000 | 0.4286 | 0.4615 | 7 | 6 |
| BaselineTagPredictor | BaselineTagPredictor-009 |  | math | 0.8000 | 0.0268 | 0.0519 | 149 | 5 |
| BaselineTagPredictor | BaselineTagPredictor-009 |  | graphs | 0.0000 | 0.0000 | 0.0000 | 54 | 0 |
| BaselineTagPredictor | BaselineTagPredictor-009 |  | strings | 1.0000 | 0.0426 | 0.0816 | 47 | 2 |
| BaselineTagPredictor | BaselineTagPredictor-009 |  | number theory | 0.0000 | 0.0000 | 0.0000 | 44 | 0 |
| BaselineTagPredictor | BaselineTagPredictor-009 |  | trees | 0.0000 | 0.0000 | 0.0000 | 41 | 0 |
| BaselineTagPredictor | BaselineTagPredictor-009 |  | geometry | 0.0000 | 0.0000 | 0.0000 | 6 | 0 |
| BaselineTagPredictor | BaselineTagPredictor-009 |  | games | 0.0000 | 0.0000 | 0.0000 | 12 | 0 |
| BaselineTagPredictor | BaselineTagPredictor-009 |  | probabilities | 0.0000 | 0.0000 | 0.0000 | 7 | 0 |
| BaselineTagPredictor | BaselineTagPredictor-010 |  | math | 0.5702 | 0.4631 | 0.5111 | 149 | 121 |
| BaselineTagPredictor | BaselineTagPredictor-010 |  | graphs | 0.5897 | 0.4259 | 0.4946 | 54 | 39 |
| BaselineTagPredictor | BaselineTagPredictor-010 |  | strings | 0.8333 | 0.5319 | 0.6494 | 47 | 30 |
| BaselineTagPredictor | BaselineTagPredictor-010 |  | number theory | 0.6667 | 0.1818 | 0.2857 | 44 | 12 |
| BaselineTagPredictor | BaselineTagPredictor-010 |  | trees | 0.7727 | 0.4146 | 0.5397 | 41 | 22 |
| BaselineTagPredictor | BaselineTagPredictor-010 |  | geometry | 0.0000 | 0.0000 | 0.0000 | 6 | 1 |
| BaselineTagPredictor | BaselineTagPredictor-010 |  | games | 1.0000 | 0.4167 | 0.5882 | 12 | 5 |
| BaselineTagPredictor | BaselineTagPredictor-010 |  | probabilities | 0.0000 | 0.0000 | 0.0000 | 7 | 0 |
| BaselineTagPredictor | BaselineTagPredictor-011 |  | math | 0.6071 | 0.4564 | 0.5211 | 149 | 112 |
| BaselineTagPredictor | BaselineTagPredictor-011 |  | graphs | 0.6316 | 0.4444 | 0.5217 | 54 | 38 |
| BaselineTagPredictor | BaselineTagPredictor-011 |  | strings | 0.6944 | 0.5319 | 0.6024 | 47 | 36 |
| BaselineTagPredictor | BaselineTagPredictor-011 |  | number theory | 0.8333 | 0.3409 | 0.4839 | 44 | 18 |
| BaselineTagPredictor | BaselineTagPredictor-011 |  | trees | 0.7667 | 0.5610 | 0.6479 | 41 | 30 |
| BaselineTagPredictor | BaselineTagPredictor-011 |  | geometry | 0.0000 | 0.0000 | 0.0000 | 6 | 5 |
| BaselineTagPredictor | BaselineTagPredictor-011 |  | games | 0.7500 | 0.5000 | 0.6000 | 12 | 8 |
| BaselineTagPredictor | BaselineTagPredictor-011 |  | probabilities | 0.0000 | 0.0000 | 0.0000 | 7 | 0 |
| BaselineTagPredictor | BaselineTagPredictor-012 |  | math | 0.3016 | 1.0000 | 0.4635 | 149 | 494 |
| BaselineTagPredictor | BaselineTagPredictor-012 |  | graphs | 0.1093 | 1.0000 | 0.1971 | 54 | 494 |
| BaselineTagPredictor | BaselineTagPredictor-012 |  | strings | 0.0951 | 1.0000 | 0.1738 | 47 | 494 |
| BaselineTagPredictor | BaselineTagPredictor-012 |  | number theory | 0.0891 | 1.0000 | 0.1636 | 44 | 494 |
| BaselineTagPredictor | BaselineTagPredictor-012 |  | trees | 0.0830 | 1.0000 | 0.1533 | 41 | 494 |
| BaselineTagPredictor | BaselineTagPredictor-012 |  | geometry | 0.0121 | 1.0000 | 0.0240 | 6 | 494 |
| BaselineTagPredictor | BaselineTagPredictor-012 |  | games | 0.0243 | 1.0000 | 0.0474 | 12 | 494 |
| BaselineTagPredictor | BaselineTagPredictor-012 |  | probabilities | 0.0142 | 1.0000 | 0.0279 | 7 | 494 |
| BaselineTagPredictor | BaselineTagPredictor-013 |  | math | 0.6827 | 0.4765 | 0.5613 | 149 | 104 |
| BaselineTagPredictor | BaselineTagPredictor-013 |  | graphs | 0.6667 | 0.3704 | 0.4762 | 54 | 30 |
| BaselineTagPredictor | BaselineTagPredictor-013 |  | strings | 0.6136 | 0.5745 | 0.5934 | 47 | 44 |
| BaselineTagPredictor | BaselineTagPredictor-013 |  | number theory | 0.7200 | 0.4091 | 0.5217 | 44 | 25 |
| BaselineTagPredictor | BaselineTagPredictor-013 |  | trees | 0.9167 | 0.5366 | 0.6769 | 41 | 24 |
| BaselineTagPredictor | BaselineTagPredictor-013 |  | geometry | 0.2857 | 0.3333 | 0.3077 | 6 | 7 |
| BaselineTagPredictor | BaselineTagPredictor-013 |  | games | 0.9091 | 0.8333 | 0.8696 | 12 | 11 |
| BaselineTagPredictor | BaselineTagPredictor-013 |  | probabilities | 0.6667 | 0.2857 | 0.4000 | 7 | 3 |
| BaselineTagPredictor | BaselineTagPredictor-014 |  | math | 0.5283 | 0.5638 | 0.5455 | 149 | 159 |
| BaselineTagPredictor | BaselineTagPredictor-014 |  | graphs | 0.4750 | 0.7037 | 0.5672 | 54 | 80 |
| BaselineTagPredictor | BaselineTagPredictor-014 |  | strings | 0.6333 | 0.8085 | 0.7103 | 47 | 60 |
| BaselineTagPredictor | BaselineTagPredictor-014 |  | number theory | 0.3485 | 0.5227 | 0.4182 | 44 | 66 |
| BaselineTagPredictor | BaselineTagPredictor-014 |  | trees | 0.6341 | 0.6341 | 0.6341 | 41 | 41 |
| BaselineTagPredictor | BaselineTagPredictor-014 |  | geometry | 0.2000 | 0.6667 | 0.3077 | 6 | 20 |
| BaselineTagPredictor | BaselineTagPredictor-014 |  | games | 0.5000 | 0.5833 | 0.5385 | 12 | 14 |
| BaselineTagPredictor | BaselineTagPredictor-014 |  | probabilities | 0.0000 | 0.0000 | 0.0000 | 7 | 2 |
| BaselineTagPredictor | BaselineTagPredictor-015 |  | math | 0.6373 | 0.4362 | 0.5179 | 149 | 102 |
| BaselineTagPredictor | BaselineTagPredictor-015 |  | graphs | 0.7059 | 0.4444 | 0.5455 | 54 | 34 |
| BaselineTagPredictor | BaselineTagPredictor-015 |  | strings | 0.7941 | 0.5745 | 0.6667 | 47 | 34 |
| BaselineTagPredictor | BaselineTagPredictor-015 |  | number theory | 0.5625 | 0.2045 | 0.3000 | 44 | 16 |
| BaselineTagPredictor | BaselineTagPredictor-015 |  | trees | 0.7333 | 0.5366 | 0.6197 | 41 | 30 |
| BaselineTagPredictor | BaselineTagPredictor-015 |  | geometry | 0.2000 | 0.1667 | 0.1818 | 6 | 5 |
| BaselineTagPredictor | BaselineTagPredictor-015 |  | games | 0.6000 | 0.2500 | 0.3529 | 12 | 5 |
| BaselineTagPredictor | BaselineTagPredictor-015 |  | probabilities | 0.0000 | 0.0000 | 0.0000 | 7 | 0 |
| BaselineTagPredictor | BaselineTagPredictor-016 |  | math | 0.7945 | 0.3893 | 0.5225 | 149 | 73 |
| BaselineTagPredictor | BaselineTagPredictor-016 |  | graphs | 0.7000 | 0.3889 | 0.5000 | 54 | 30 |
| BaselineTagPredictor | BaselineTagPredictor-016 |  | strings | 0.8571 | 0.5106 | 0.6400 | 47 | 28 |
| BaselineTagPredictor | BaselineTagPredictor-016 |  | number theory | 1.0000 | 0.1818 | 0.3077 | 44 | 8 |
| BaselineTagPredictor | BaselineTagPredictor-016 |  | trees | 0.8421 | 0.3902 | 0.5333 | 41 | 19 |
| BaselineTagPredictor | BaselineTagPredictor-016 |  | geometry | 0.0000 | 0.0000 | 0.0000 | 6 | 0 |
| BaselineTagPredictor | BaselineTagPredictor-016 |  | games | 1.0000 | 0.0833 | 0.1538 | 12 | 1 |
| BaselineTagPredictor | BaselineTagPredictor-016 |  | probabilities | 0.0000 | 0.0000 | 0.0000 | 7 | 0 |
| BaselineTagPredictor | BaselineTagPredictor-017 |  | math | 0.3016 | 1.0000 | 0.4635 | 149 | 494 |
| BaselineTagPredictor | BaselineTagPredictor-017 |  | graphs | 0.1093 | 1.0000 | 0.1971 | 54 | 494 |
| BaselineTagPredictor | BaselineTagPredictor-017 |  | strings | 0.0951 | 1.0000 | 0.1738 | 47 | 494 |
| BaselineTagPredictor | BaselineTagPredictor-017 |  | number theory | 0.0891 | 1.0000 | 0.1636 | 44 | 494 |
| BaselineTagPredictor | BaselineTagPredictor-017 |  | trees | 0.0830 | 1.0000 | 0.1533 | 41 | 494 |
| BaselineTagPredictor | BaselineTagPredictor-017 |  | geometry | 0.0121 | 1.0000 | 0.0240 | 6 | 494 |
| BaselineTagPredictor | BaselineTagPredictor-017 |  | games | 0.0243 | 1.0000 | 0.0474 | 12 | 494 |
| BaselineTagPredictor | BaselineTagPredictor-017 |  | probabilities | 0.0142 | 1.0000 | 0.0279 | 7 | 494 |
| BaselineTagPredictor | BaselineTagPredictor-018 |  | math | 0.6154 | 0.4832 | 0.5414 | 149 | 117 |
| BaselineTagPredictor | BaselineTagPredictor-018 |  | graphs | 0.6667 | 0.2963 | 0.4103 | 54 | 24 |
| BaselineTagPredictor | BaselineTagPredictor-018 |  | strings | 0.6136 | 0.5745 | 0.5934 | 47 | 44 |
| BaselineTagPredictor | BaselineTagPredictor-018 |  | number theory | 0.6154 | 0.5455 | 0.5783 | 44 | 39 |
| BaselineTagPredictor | BaselineTagPredictor-018 |  | trees | 0.9545 | 0.5122 | 0.6667 | 41 | 22 |
| BaselineTagPredictor | BaselineTagPredictor-018 |  | geometry | 0.4000 | 0.3333 | 0.3636 | 6 | 5 |
| BaselineTagPredictor | BaselineTagPredictor-018 |  | games | 0.9091 | 0.8333 | 0.8696 | 12 | 11 |
| BaselineTagPredictor | BaselineTagPredictor-018 |  | probabilities | 0.5000 | 0.2857 | 0.3636 | 7 | 4 |
| BaselineTagPredictor | BaselineTagPredictor-019 |  | math | 0.5000 | 0.0067 | 0.0132 | 149 | 2 |
| BaselineTagPredictor | BaselineTagPredictor-019 |  | graphs | 0.0000 | 0.0000 | 0.0000 | 54 | 1 |
| BaselineTagPredictor | BaselineTagPredictor-019 |  | strings | 0.0000 | 0.0000 | 0.0000 | 47 | 1 |
| BaselineTagPredictor | BaselineTagPredictor-019 |  | number theory | 0.0000 | 0.0000 | 0.0000 | 44 | 1 |
| BaselineTagPredictor | BaselineTagPredictor-019 |  | trees | 0.0000 | 0.0000 | 0.0000 | 41 | 1 |
| BaselineTagPredictor | BaselineTagPredictor-019 |  | geometry | 0.0000 | 0.0000 | 0.0000 | 6 | 1 |
| BaselineTagPredictor | BaselineTagPredictor-019 |  | games | 0.0000 | 0.0000 | 0.0000 | 12 | 0 |
| BaselineTagPredictor | BaselineTagPredictor-019 |  | probabilities | 0.0000 | 0.0000 | 0.0000 | 7 | 0 |
| BaselineTagPredictor | BaselineTagPredictor-020 |  | math | 0.6078 | 0.4161 | 0.4940 | 149 | 102 |
| BaselineTagPredictor | BaselineTagPredictor-020 |  | graphs | 0.6571 | 0.4259 | 0.5169 | 54 | 35 |
| BaselineTagPredictor | BaselineTagPredictor-020 |  | strings | 0.8400 | 0.4468 | 0.5833 | 47 | 25 |
| BaselineTagPredictor | BaselineTagPredictor-020 |  | number theory | 0.5625 | 0.2045 | 0.3000 | 44 | 16 |
| BaselineTagPredictor | BaselineTagPredictor-020 |  | trees | 0.8696 | 0.4878 | 0.6250 | 41 | 23 |
| BaselineTagPredictor | BaselineTagPredictor-020 |  | geometry | 0.0000 | 0.0000 | 0.0000 | 6 | 1 |
| BaselineTagPredictor | BaselineTagPredictor-020 |  | games | 1.0000 | 0.2500 | 0.4000 | 12 | 3 |
| BaselineTagPredictor | BaselineTagPredictor-020 |  | probabilities | 0.0000 | 0.0000 | 0.0000 | 7 | 0 |



| TunedBaselineTagPredictor | TunedBaselineTagPredictor-001 |  | math | 0.6243 | 0.3803 | 0.4726 | 284 | 173 |
| TunedBaselineTagPredictor | TunedBaselineTagPredictor-001 |  | graphs | 0.7538 | 0.4153 | 0.5355 | 118 | 65 |
| TunedBaselineTagPredictor | TunedBaselineTagPredictor-001 |  | strings | 0.6190 | 0.5493 | 0.5821 | 71 | 63 |
| TunedBaselineTagPredictor | TunedBaselineTagPredictor-001 |  | number theory | 0.6857 | 0.3038 | 0.4211 | 79 | 35 |
| TunedBaselineTagPredictor | TunedBaselineTagPredictor-001 |  | trees | 0.7917 | 0.6667 | 0.7238 | 57 | 48 |
| TunedBaselineTagPredictor | TunedBaselineTagPredictor-001 |  | geometry | 0.7222 | 0.3939 | 0.5098 | 33 | 18 |
| TunedBaselineTagPredictor | TunedBaselineTagPredictor-001 |  | games | 0.8750 | 0.5600 | 0.6829 | 25 | 16 |
| TunedBaselineTagPredictor | TunedBaselineTagPredictor-001 |  | probabilities | 0.8750 | 0.6364 | 0.7368 | 22 | 16 |
| TunedBaselineTagPredictor | TunedBaselineTagPredictor-002 |  | math | 0.5773 | 0.3944 | 0.4686 | 284 | 194 |
| TunedBaselineTagPredictor | TunedBaselineTagPredictor-002 |  | graphs | 0.7778 | 0.2373 | 0.3636 | 118 | 36 |
| TunedBaselineTagPredictor | TunedBaselineTagPredictor-002 |  | strings | 0.6301 | 0.6479 | 0.6389 | 71 | 73 |
| TunedBaselineTagPredictor | TunedBaselineTagPredictor-002 |  | number theory | 0.6364 | 0.3544 | 0.4553 | 79 | 44 |
| TunedBaselineTagPredictor | TunedBaselineTagPredictor-002 |  | trees | 0.9062 | 0.5088 | 0.6517 | 57 | 32 |
| TunedBaselineTagPredictor | TunedBaselineTagPredictor-002 |  | geometry | 0.5714 | 0.2424 | 0.3404 | 33 | 14 |
| TunedBaselineTagPredictor | TunedBaselineTagPredictor-002 |  | games | 0.8667 | 0.5200 | 0.6500 | 25 | 15 |
| TunedBaselineTagPredictor | TunedBaselineTagPredictor-002 |  | probabilities | 0.9231 | 0.5455 | 0.6857 | 22 | 13 |

