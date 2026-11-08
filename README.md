# Obesity Level Classification – ML Assignment 2

## a. Problem Statement

Obesity is a major risk factor for diabetes, heart disease and hypertension. The aim of this project is to **predict a person's obesity level (7 classes)** from their eating habits, physical condition and basic demographic details, using five classical machine learning classifiers. The models are compared using six evaluation metrics and served through an interactive Streamlit web application.

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

## c. GitHub Repository Link

<https://github.com/UdayBelde25/BitsMLAssignment>

## d. Models Used

Planned models: Logistic Regression, Decision Tree, k-Nearest Neighbours, Naive Bayes (Gaussian) and Support Vector Machine.

Evaluation metrics: Accuracy, AUC, Precision, Recall, F1 score and Matthews Correlation Coefficient (MCC).

_Results and observations to be added._
