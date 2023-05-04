# Heart Failure Prediction
**Authors : Esther Ki, Gyubeen Park, Satwik Behera**

## Table of Contents

1. Introduction
2. Data Description
  - 2.1 Data
  - 2.2 EDA
    - 2.2.1 Categorical variables
    - 2.2.2 Numerical Variables
3. Methods and Results
  - 3.1 Modelling
    - 3.1.1 Overview of Modeling
    - 3.1.2 Logistic Regression
      - 3.1.2.1 Full Additive Model
      - 3.1.2.2 Selective Interaction Model
      - 3.1.2.3 Backward Stepwise Selection for Final Model
    - 3.1.3 Random Forest
    - 3.1.4 Decision Tree
    - 3.1.5 Naive Bayes
    - 3.1.6 k-Nearest Neighbors
4. Conclusion
  - 4.1 Summary of Findings
  - 4.2 Significance of Results
  - 4.3 Limitations and Future Work
    - 4.3.1 Cholesterol
      - 4.3.1.1 Grouping
      - 4.3.1.2 Mean Replacement
      - 4.3.1.3 Deletion
## 1. Introduction
Cardiovascular diseases(CVDs)  are the leading cause of death globally, accounting for 31% of all deaths, with an estimated 17.7 million lives lost each year. Early detection and management are necessary for people with heart diseases or high cardiovascular risk. Four out of five CVD deaths are due to heart attacks and strokes, and a third of these occur prematurely in people under 70. Since heart failure is common in cardiovascular diseases, we aim to build a model that predicts heart failure which will help to predict a potential heart disease.

## Data Description
### 2.1 Data
Our data is from Kaggle and five heart datasets from UCI Machine Learning Repository are combined over 11 common features. The five heart datasets are originally from the following: 
 1. Hungarian Institute of Cardiology,  Budapest
 2. University Hospital, Zurich, Switzerland 
 3. University Hospital, Basel, Switzerland
 4. V.A. Medical Center, Long Beach and Cleveland Clinic Foundation

The combined dataset has 918 observations and 12 variables including the response variable named HeartDisease with a binary value of 0(Normal) or 1(Heart Disease). 


To explain some of the important variables, Oldpeak and ST_slope are the features of exercise electrocardiogram(ECG). Oldpeak is ST depression induced by exercise relative to rest, and low readings can be a sign of reduced blood flow to the heart muscle. ST slope refers to the direction and change in the ST segment of ECG. An upsloping ST segment may be a normal finding in younger individuals who are physically active while a downsloping ST can be indicative of coronary artery disease or other cardiac conditions.

### 2.2 EDA
