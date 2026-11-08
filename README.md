# Obesity Level Classification – ML Assignment 2

## a. Problem Statement

Obesity is a major risk factor for diabetes, heart disease and hypertension. The aim of this project is to **predict a person's obesity level (7 classes)** from their eating habits, physical condition and basic demographic details, using five classical machine learning classifiers. The five models are compared on a common held-out test set using six evaluation metrics. The trained models are served through an interactive Streamlit web application.

## b. Dataset Description

| Item | Details |
|---|---|
| Name | Estimation of Obesity Levels Based on Eating Habits and Physical Condition |
| Source | UCI Machine Learning Repository (ID 544), also on Kaggle as *Obesity Levels* |
| Instances | 2,111 (2,087 after removing 24 duplicate rows) |
| Features | 16 (8 numeric, 8 categorical) |
| Target | `NObeyesdad`, a multi-class label with 7 classes |
| Origin | Individuals from Mexico, Peru and Colombia. About 23% of the records were collected through a web survey; the rest were generated synthetically with SMOTE to balance the classes. |

**Target classes:** Insufficient_Weight, Normal_Weight, Overweight_Level_I, Overweight_Level_II, Obesity_Type_I, Obesity_Type_II, Obesity_Type_III. The classes are roughly balanced, at about 13–17% each.

**Features**

| Feature | Meaning | Type |
|---|---|---|
| Gender | Male / Female | Categorical |
| Age | Age in years | Numeric |
| Height | Height in metres | Numeric |
| Weight | Weight in kg | Numeric |
| family_history_with_overweight | Family member is or was overweight | Categorical (yes/no) |
| FAVC | Frequent consumption of high-calorie food | Categorical (yes/no) |
| FCVC | Frequency of eating vegetables (1–3) | Numeric |
| NCP | Number of main meals per day (1–4) | Numeric |
| CAEC | Eating between meals | Categorical (no / Sometimes / Frequently / Always) |
| SMOKE | Smoker | Categorical (yes/no) |
| CH2O | Daily water intake (1–3) | Numeric |
| SCC | Monitors calorie intake | Categorical (yes/no) |
| FAF | Physical activity frequency (0–3) | Numeric |
| TUE | Time spent on technology devices (0–2) | Numeric |
| CALC | Alcohol consumption | Categorical (no / Sometimes / Frequently / Always) |
| MTRANS | Usual mode of transport | Categorical (5 values) |

**Preprocessing:** duplicate rows are removed, and the data is split 75/25 into train and test sets, stratified by class (`random_state=74`). Inside each model's pipeline, numeric features are standardised (`StandardScaler`) and categorical features are one-hot encoded. The 522-row test split is saved as `test_data.csv`.

## c. GitHub Repository Link

<https://github.com/UdayBelde25/BitsMLAssignment>

**Live Streamlit app:** _to be added after deployment_

```
project-folder/
├── app.py                  # Streamlit web application
├── requirements.txt
├── README.md
├── test_data.csv           # 522-row held-out test set (upload this in the app)
├── data/
│   └── obesity_raw.csv     # original UCI dataset
└── model/
    ├── train_models.py     # training + evaluation script
    ├── *.pkl               # 5 saved sklearn pipelines
    ├── metrics.csv         # evaluation results
    └── meta.json           # feature list, class order, split info
```

To reproduce the results:
```bash
pip install -r requirements.txt
python model/train_models.py
streamlit run app.py
```

## d. Models Used

All five models were trained on the same training split and evaluated on the same test split. Precision, Recall and F1 are weighted averages over the 7 classes. AUC is the weighted one-vs-rest ROC-AUC.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9425 | 0.9944 | 0.9443 | 0.9425 | 0.9422 | 0.9332 |
| Decision Tree | 0.9215 | 0.9647 | 0.9250 | 0.9215 | 0.9204 | 0.9091 |
| kNN | 0.8410 | 0.9679 | 0.8469 | 0.8410 | 0.8215 | 0.8189 |
| Naive Bayes | 0.5402 | 0.8944 | 0.7182 | 0.5402 | 0.4704 | 0.5029 |
| Support Vector Machine | **0.9579** | **0.9978** | **0.9591** | **0.9579** | **0.9575** | **0.9511** |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong second place, with 94.3% accuracy and a 0.994 AUC. The obesity classes are essentially BMI bands (Weight / Height²), so they are close to separable using Height and Weight once the features are standardised. A linear model approximates these bands well but cannot follow the curved BMI boundary exactly, so its few errors fall between neighbouring classes: Overweight II ↔ I, and Normal → Insufficient Weight. |
| Decision Tree | 92.2% accuracy. The tree places axis-aligned thresholds on Weight and Height, which approximates the BMI cut-offs reasonably well, and it needs no scaling. Feature importance confirms this: Weight (0.47), Height (0.27) and Gender (0.16) account for over 90% of the splits. Step-shaped boundaries make it weaker on the borderline classes (Normal_Weight recall 0.76, often predicted as Overweight I), and its coarse leaf probabilities give the lowest AUC (0.965) apart from Naive Bayes. Depth was limited to 12 to reduce overfitting. |
| kNN | Moderate, at about 84%. Height and Weight carry most of the signal, but the distance is computed over all 16 features, including 20+ one-hot columns. The weak features dilute the distance, so the neighbours found are not always truly similar. It does worst on Normal_Weight (recall only 0.29), which sits between Insufficient Weight and the Overweight classes and is confused with all of them. |
| Naive Bayes | By far the weakest, at 54% accuracy and an MCC of 0.50. GaussianNB assumes the features are independent and normally distributed. Both assumptions fail here: the class depends on a combination of Height and Weight, which are correlated, and the one-hot or ordinal features (1–3 scales) are not Gaussian. It almost never predicts Obesity_Type_I (recall 0.01) or Overweight_Level_I (0.09), pushing them into neighbouring classes instead; for example, 50 Obesity_Type_I cases were predicted as Overweight_Level_II. Its AUC (0.89) shows that its rankings are still informative even though its hard predictions are poor. |
| Support Vector Machine | Best on every metric: 95.8% accuracy, 0.998 AUC and 0.951 MCC. The RBF kernel learns the non-linear (BMI-shaped) class boundaries smoothly, and standardised features suit it well. Its remaining errors are mostly Normal_Weight cases near the class boundaries. The trade-off is slower training and lower interpretability. |
| **Overall winner** | **Support Vector Machine (RBF kernel)**, with the highest scores on all six metrics. Logistic Regression is a close second (about 1.5 points lower), and is simpler and more interpretable. |

## Streamlit App Features

- **CSV upload** of test data, with a button to download the sample `test_data.csv`
- **Model selection dropdown** for the 5 classifiers
- **Evaluation metrics**: Accuracy, AUC, Precision, Recall, F1 and MCC
- **Confusion matrix** heatmap and a **classification report**
- **Model comparison** tab with all five models side by side and the best model highlighted
- **Predictions** tab with a filter for misclassified rows
