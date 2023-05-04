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

## 2. Data Description
### 2.1 Data
Our data is from Kaggle and five heart datasets from UCI Machine Learning Repository are combined over 11 common features. The five heart datasets are originally from the following: 
 1. Hungarian Institute of Cardiology,  Budapest
 2. University Hospital, Zurich, Switzerland 
 3. University Hospital, Basel, Switzerland
 4. V.A. Medical Center, Long Beach and Cleveland Clinic Foundation

The combined dataset has 918 observations and 12 variables including the response variable named HeartDisease with a binary value of 0(Normal) or 1(Heart Disease). 
| <img width="599" alt="image" src="https://user-images.githubusercontent.com/43529908/236326944-19603126-c715-4d66-93fd-590717bb5435.png"> |
| ------------- |
| **Table 1**. Descriptions of the Variables in Our Dataset | 

To explain some of the important variables, Oldpeak and ST_slope are the features of exercise electrocardiogram(ECG). Oldpeak is ST depression induced by exercise relative to rest, and low readings can be a sign of reduced blood flow to the heart muscle. ST slope refers to the direction and change in the ST segment of ECG. An upsloping ST segment may be a normal finding in younger individuals who are physically active while a downsloping ST can be indicative of coronary artery disease or other cardiac conditions.

### 2.2 EDA
#### 2.2.1 Categorical variables
We first explored the categorical variables. There are much more males than females in the data in general and also more males in the heart disease group (Figure 1). Looking at Figure 2, the plot of ST Slope, there is more upsloping in the normal group while there are much more flat slopes and increased downsloping in the disease group. In Figure 3 and 4, we can see that exercise angina, which is a chest pain that occurs during physical activity, and high fasting blood sugar are more common in the heart disease group. 
|![image](https://user-images.githubusercontent.com/43529908/236327712-8bfa7842-7cd4-47c8-bfdd-d7f9436524db.png) | ![image](https://user-images.githubusercontent.com/43529908/236327833-68bab97a-96aa-4f5c-8e57-932b77a43d14.png) |
| --------------- | ------------------ |
|**Figure 1**. Distribution of Heart Disease by Sex  | **Figure 2**. Distribution of Heart Disease by ST Slope |
|![image](https://user-images.githubusercontent.com/43529908/236328148-fe327463-1f28-4a00-ae19-f21ce9ef4d93.png) | ![image](https://user-images.githubusercontent.com/43529908/236328246-1c2105af-d18c-4efc-a603-d1bb1e722ea6.png) |
| **Figure 3**. Distribution of Heart Disease by Fasting BS | **Figure 4**. Distribution of Heart Disease by Exercise Angina |

#### 2.2.2 Numerical Variables
Correlations between numerical variables are checked and confirmed there are no strong pairwise correlations (Figure 5 and 6). The box plot of Resting blood pressure (Figure 7) shows that there is 1 observation with resting blood pressure equal to 0 and we removed this observation. The box plot of Cholesterol (Figure 8) also indicates there are some 0 values in the column. We found out 172 out of 918 observations, which is about 19% of the data, have cholesterol equal to 0. Serum cholesterol level cannot be 0 so these zeros are missing values or there is also possibility that the data collectors recorded very low cholesterol level as 0. Since It is widely known and a lot of scientific journals have proved that high cholesterol levels can increase the risk of heart diseases, we looked at how many people with 0 cholesterol level have heart disease. Then it turned out 152 out of 172 of them have heart diseases. Therefore, we concluded that these zeros are missing values. After multiple trials to deal with these missing values, we came up with the decision to remove the Cholesterol variable. More detailed description and results of our trials can be found below in 4.3.1 Cholesterol.

| ![image](https://user-images.githubusercontent.com/43529908/236329753-8492dda6-149b-4253-908f-3cd88e36f62f.png) | ![image](https://user-images.githubusercontent.com/43529908/236329858-fcd82073-521d-4ace-a33e-bb0d64531dd5.png) |
| -------- | --------- |
| **Figure 5**. Scatter plot matrix of numerical variables | **Figure 6**. Correlation Heat Map |
| ![image](https://user-images.githubusercontent.com/43529908/236330060-01df166f-d0b4-47a6-99bd-374b8417945f.png) | ![image](https://user-images.githubusercontent.com/43529908/236330206-9da18289-deab-41a6-acce-3ab99735b046.png) |
| **Figure 7**. Box Plot of Resting Blood Pressure | **Figure 8**. Box Plot of Cholesterol |

## 3. Methods and Results
### 3.1 Modeling
Model selection is a crucial step in Machine Learning that involves choosing and validating the best model for prediction, considering factors like complexity, interpretability, and available data, and our use case requires a classification model with high accuracy and interpretability.
#### 3.1.1 Overview of Modeling
We chose to use five different models for our project, including Logistic regression, Random forest, Decision trees, Naive Bayes, and kNN, each with their own set of advantages and disadvantages, which are outlined in a table for comparison.
|              | Logistic Regression | Random Forest | Decision Trees | Naive Bayes | kNN    |
| ------------ | ------------------- | --------------| -------------- | ----------- | ------ |
| Advantages   | Designed for Classification, Useful for understanding the variable influence| No overfitting, Better than DT | Simple to Understand and visualize | Small training data | Simple to implement, Robust to noise |
| Disadvantages | Limited to Binary Classification | Slow real-time predictions, complex | Can create complex trees that don't generalize, Sensistive to Variations | Bad Estimator | Manual Selection of k, high computation cost |
