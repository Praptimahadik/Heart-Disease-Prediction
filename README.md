# Heart-Disease-Prediction

# Objectives
The primary objective of this project is to develop a machine learning model capable of predicting whether a patient is likely to have heart disease based on various medical and clinical parameters. Heart disease is one of the leading causes of mortality worldwide, and early prediction can support timely diagnosis, preventive healthcare, and effective treatment planning.

This project aims to:

Analyze patient health records to identify patterns associated with heart disease.
Perform exploratory data analysis (EDA) to understand feature behavior and relationships.
Preprocess and prepare the dataset for machine learning.
Train multiple classification models and compare their performance.
Apply hyperparameter tuning to optimize model performance.
Select the best-performing model for heart disease prediction.
The project follows the complete machine learning pipeline, beginning from raw data ingestion and ending with model evaluation and comparison.

# Problem Statement
The problem addressed in this project is a binary classification problem, where the target variable heart_disease_present has two possible outcomes:

0 → No heart disease
1 → Presence of heart disease
Using patient medical information such as blood pressure, cholesterol levels, chest pain type, age, exercise-induced angina, and other diagnostic parameters, the objective is to classify whether a patient is at risk of heart disease.

# Data Reading and Data Collection
The dataset was provided in two separate CSV files:

values.csv — containing patient features
labels.csv — containing target labels
Both datasets were merged using the common identifier patient_id.

Code Logic Used:
import pandas as pd

df1 = pd.read_csv('values.csv')
df2 = pd.read_csv('labels.csv')

They were merged:
df = pd.merge(df1, df2, on='patient_id')

A merged file was also exported:
df.to_csv('merged.csv', index=False)

Why Merging Was Required ?

Since predictor variables and target labels were stored separately, merging was necessary to create one complete dataset for analysis and modeling.

Dataset Size After merging:

Rows: 180 observations
Columns: 15 variables
This means the dataset contains information from 180 patients with 14 predictors and 1 target variable.

# Data Inspection
Data inspection is a critical initial stage in any data science project, as it helps in understanding the structure, quality, and characteristics of the dataset before moving toward analysis and model building. In this project, after merging the two datasets (values.csv and labels.csv) using the common key patient_id, the resulting dataset was carefully inspected to ensure the data was properly loaded, complete, and suitable for further processing. This step helps identify potential issues such as missing values, incorrect data types, duplicates, irrelevant features, and unusual patterns in the data.

1) Viewing the Dataset
The first step in data inspection was viewing the dataset using df.head() and df.tail(). The head() function was used to display the first five rows of the dataset, while tail() displayed the last five rows. This provided an initial overview of the data and helped verify that the merge operation had been performed correctly.Viewing the first and last rows also gave an early sense of the types of values present in columns such as blood pressure, cholesterol, age, chest pain type, and thal values.

2. Shape of the Dataset
Next, the dimensions of the dataset were checked using the df.shape function. The dataset was found to contain 180 rows and 15 columns, indicating that information for 180 patients was available along with 15 variables. Out of these, 14 were predictor variables and 1 was the target variable.
Understanding the size of the dataset is important because it gives insight into the volume of available data for analysis and model training.

3. Data Types and Dataset Information
The structure and data types of all columns were examined using df.info(). This step provided detailed information about the dataset, including column names, number of non-null values, and the data type of each variable.

The dataset consisted of three major data types:

Integer (int64) features, mainly representing numerical and encoded categorical variables Float (float64) features, representing continuous variables such as ST depression Object features, representing categorical text variables

The numerical variables included features such as age, resting_blood_pressure, serum_cholesterol_mg_per_dl, max_heart_rate_achieved, and oldpeak_eq_st_depression, which contain continuous or measurable patient information.

Categorical variables included features such as thal, sex, chest_pain_type, and exercise_induced_angina, which represent categories or groups rather than continuous values.

The target variable heart_disease_present was identified as the dependent variable used for classification, where the model predicts whether heart disease is present or absent.

Checking data types is important because machine learning algorithms require data in appropriate formats. It also helps determine which columns may require encoding, scaling, or transformation during preprocessing.

4. Statistical Summary
Descriptive statistics were generated using df.describe() to summarize the numerical variables in the dataset. This provided important statistical measures such as mean, median, standard deviation, minimum and maximum values, and quartiles (25%, 50%, and 75%). Overall, the statistical summary provided a strong understanding of data distribution and helped guide subsequent exploratory analysis and preprocessing decisions.

5. Dropping Irrelevant Feature
After inspecting the dataset, the column patient_id was removed using:
df.drop('patient_id', axis=1, inplace=True)

This feature was dropped because it is only a unique identifier assigned to each patient and does not contain any medical or predictive information relevant to heart disease.

Including such identifier columns in machine learning models can be problematic because they do not contribute meaningful patterns and may introduce unnecessary noise into the model. Since each ID is unique, the model may attempt to treat it as a feature, which can negatively impact learning.

# Exploratory Data Analysis (EDA)
Exploratory Data Analysis (EDA) is one of the most important stages in a data science project, as it helps uncover patterns, trends, relationships, and anomalies present in the data before model building. In this project, EDA was performed to better understand how different medical attributes relate to heart disease and to identify important predictors for classification. It also helped in detecting issues such as skewness, outliers, feature distributions, and possible relationships among variables.

The EDA process was divided into three major parts: Univariate Analysis, Bivariate Analysis, and Multivariate Analysis. Each level of analysis served a different purpose in understanding the data from simple individual feature behavior to complex interactions among multiple variables.

### 1. Univariate Analysis
Univariate analysis involves studying one variable at a time to understand its individual behavior, distribution, and characteristics. This type of analysis is useful for detecting feature spread, skewness, outliers, frequency distributions, and general patterns within each variable.

####  1.1 Numerical Feature Distribution
For numerical variables, histograms along with density curves were plotted to analyze the distribution of each continuous feature. The numerical features examined included age, resting blood pressure, serum cholesterol, maximum heart rate achieved, and oldpeak (ST depression induced by exercise).

The main objective of plotting histograms was to understand the shape of distributions, whether variables followed normal distribution patterns, whether they were skewed, and whether extreme values or outliers existed.

####  1.2 Categorical Variable Analysis
For categorical variables, countplots were used to visualize category frequencies and understand how observations were distributed across classes. The categorical variables analyzed included sex, chest pain type, number of major vessels, fasting blood sugar, resting ECG results, exercise induced angina, slope of peak exercise ST segment, and thal.

The purpose of this analysis was to examine category frequencies, detect imbalance in categories, and understand how often certain medical conditions or attributes occurred in the patient population.

### Bivariate Analysis
####  Numerical Features vs Target
To analyze the relationship between numerical variables and the target variable heart_disease_present, boxplots were used to compare the distribution of each continuous feature across the two target classes: patients without heart disease (0) and patients with heart disease (1). Boxplots are particularly useful because they display median values, spread, interquartile ranges, and outliers, making it easier to compare how each feature behaves across both groups.

####  Categorical Features vs Target
To analyze relationships between categorical variables and heart disease, countplots were generated using the target variable as the hue. These plots helped compare how category frequencies differed between patients with and without heart disease, allowing visual identification of features associated with higher disease risk.

###  Multivariate Analysis
While univariate and bivariate analyses study individual features and pairwise relationships, multivariate analysis explores interactions among multiple variables simultaneously.

In this project, a pairplot was generated using:
sns.pairplot(df, hue=target)
The purpose of the pairplot was to visualize pairwise relationships among multiple features while distinguishing classes using color based on the target variable.

This helped observe:

Relationships among predictors
Potential clustering patterns
Possible separation between disease classes
Interaction effects among variables
Some variable combinations showed visible separation between patients with and without heart disease, indicating that combinations of features may provide stronger predictive power than individual features alone.

The pairplot also helped reveal whether clusters of patients belonging to different target classes existed and whether certain variable interactions may assist classification algorithms.

These insights are valuable because machine learning models often capture multivariate relationships that may not be obvious from individual feature analysis.

# Data Preprocessing and Feature Engineering
Data preprocessing is a crucial phase in the machine learning pipeline, as it converts raw and potentially inconsistent data into a clean and model-ready format. Even well-structured datasets require preprocessing to ensure data quality, reliability, and optimal model performance. In this project, preprocessing began with assessing the dataset for missing values and duplicate records, as both issues can negatively impact analysis and model training if not addressed properly.

Before applying advanced preprocessing techniques such as outlier treatment, encoding, and scaling, it was necessary to verify whether the dataset had completeness and uniqueness issues. These checks help ensure that the models learn from accurate and unbiased data.

#### Missing Value Check
The first preprocessing step was checking for missing or null values using:
df.isnull().sum()
This function calculates the number of missing values in each column and helps identify whether any features have incomplete observations.

After performing this check, it was found that all columns contained zero missing values, meaning the dataset was completely populated with no null entries.

#### Duplicate Check
After confirming data completeness, the next step was checking for duplicate records using:
df.duplicated().sum()
This function identifies rows that are exact repetitions of other rows and counts the total number of duplicates in the dataset.

The result showed:

0 duplicate records

Interpretation of Results
This means every patient observation in the dataset was unique, and there were no repeated entries.

# Outlier Detection and Treatment
After checking for missing values and duplicates, the next preprocessing step involved identifying and treating outliers. Outliers are extreme observations that lie significantly outside the normal range of a variable. In medical datasets, outliers may occur due to abnormal patient conditions, rare cases, measurement errors, or data entry issues. If left untreated, they can distort statistical analysis and negatively affect machine learning model performance, especially for algorithms sensitive to extreme values.

To detect outliers, the Interquartile Range (IQR) method was used. This method identifies unusual values based on the spread of the middle 50% of the data. First, the first quartile (Q1) representing the 25th percentile and the third quartile (Q3) representing the 75th percentile were calculated. The difference between them gives the Interquartile Range (IQR):

IQR=Q3−Q1

Using IQR, lower and upper bounds were computed to detect outliers:

Lower Bound: Q1−1.5(IQR)

Upper Bound: Q3+1.5(IQR)

Any value falling outside these limits was considered an outlier.

#### Outliers Identified
Outlier detection showed the presence of extreme values in several numerical variables:

Resting blood pressure had 6 outliers
Serum cholesterol had 2 outliers
Oldpeak had 4 outliers
These variables showed unusual values that could disproportionately influence model learning if not handled properly.

#### Outlier Treatment Using Capping (Winsorization)
Instead of removing records containing outliers, capping (winsorization) was used. In this approach, values below the lower bound were replaced with the lower limit, and values above the upper bound were replaced with the upper limit.

This method was chosen instead of deleting rows because the dataset contains only 180 observations, which is relatively small for machine learning. Removing records could reduce available information and weaken model training.

Overall, outlier treatment improved data quality while preserving dataset size and important patient information.

# Encoding Categorical Variables
After cleaning numerical features, categorical variables needed transformation because machine learning algorithms generally require numerical inputs and cannot directly process text-based categories.

To convert categorical features into machine-readable format, One-Hot Encoding was applied using:
df = pd.get_dummies(df, drop_first=True)
One-hot encoding converts each category into separate binary columns containing 0s and 1s.

For example, a variable such as thal with multiple categories gets transformed into multiple indicator columns representing each category.
The parameter drop_first=True was specifically used to avoid the dummy variable trap, which occurs when encoded variables become perfectly correlated with one another.
This also helps reduce multicollinearity, where highly correlated predictors can negatively impact some models such as Logistic Regression.

# Multicollinearity Check
After encoding, the next step was checking for multicollinearity, which occurs when predictor variables are highly correlated with one another.

High multicollinearity can create problems such as:

1) Unstable model coefficients
2) Redundant information
3) Reduced interpretability
4) Poor generalization in some models
   
A correlation threshold of: Correlation > 0.8
was used as an indicator of potentially problematic multicollinearity.
After analyzing correlations, no feature pairs showed severe correlation above the threshold. Some moderate relationships existed, but no predictors were highly collinear.

# Feature Scaling
Once features were cleaned and encoded, scaling was performed.
First, predictors and target variable were separated:

X = df.drop('heart_disease_present',axis=1)
y = df['heart_disease_present']
Then Standardization was applied using StandardScaler.

Why Scaling Was Necessary
Features such as: Age,Blood pressure,Cholesterol exist on very different scales.
For example:
Cholesterol may be in hundreds Oldpeak may be small decimals
Without scaling, large-scale variables may dominate model learning.
##### Scaling is especially important for:

Logistic Regression
Support Vector Machine (SVM)
K-Nearest Neighbors (KNN)
These algorithms are sensitive to feature magnitude.
Overall, scaling ensured balanced contribution from all numerical features.

# Train-Test Split
After preprocessing, the dataset was split into training and testing sets using:

train_test_split(test_size=0.2)
An 80-20 split was used:

Training set: 144 observations (80%)
Testing set: 36 observations (20%)
Purpose of Train-Test Split:

The training set was used for model learning. During this phase, algorithms identify relationships between predictors and target labels.
The testing set was kept separate and used only after training to evaluate model performance on unseen data.

# Class Distribution Check
Before model training, the target variable distribution was examined.

Class counts were:
No heart disease (0): 100
Heart disease present (1): 80

Interpretation: 
This indicates the dataset is fairly balanced, with both classes reasonably represented.

# Model Building
After completing data preprocessing, the next stage was model building, where multiple machine learning classification algorithms were trained and evaluated to identify the most effective model for predicting heart disease. Rather than relying on a single algorithm, several models were implemented to compare performance, understand how different algorithms capture patterns in the data, and select the most suitable model based on both predictive accuracy and generalization ability.

Since this is a binary classification problem, the models were evaluated primarily using accuracy, confusion matrix analysis, cross-validation, and classification metrics such as precision, recall, and F1-score.

# Evaluation Metrics Used
After training the machine learning models, their performance was evaluated using multiple evaluation metrics rather than relying only on accuracy. In classification problems, especially in healthcare applications such as heart disease prediction, using multiple metrics is important because a single metric may not fully capture model performance.

For example, a model may show high accuracy but still fail to correctly identify patients with disease, which would be risky in a medical setting. Therefore, several evaluation metrics were used to measure performance from different perspectives, including Accuracy, Precision, Recall, F1-Score, and Confusion Matrix analysis.

# Model Comparison Report
Model Comparison Table (With and Without Tuning)
    Model	                         Accuracy (Without Tuning)	Accuracy (With Tuning)	Cross-Validation Accuracy
Logistic Regression                	         86.11%	         — (No tuning applied)	             —
Random Forest	                               86.11%	                  83.33%	               82.22%
Decision Tree	                               77.78%	                  77.78%	               67.78%
Gradient Boosting	                           75.00%	                  77.78%	               77.78%
XGBoost	                                     77.78%	                  83.33%	               81.11%
Support Vector Machine	                     86.11%	                  83.33%	               81.11%
K-Nearest Neighbors	                         83.33%	                  86.11%	               85.00%

Based on model comparison, four models emerged as top performers:

Logistic Regression
Random Forest
Support Vector Machine
Tuned K-Nearest Neighbors
Each achieved:

86.11% accuracy

##### Why These Were Considered Best Models ?

These models were considered strongest because they combined:

Highest test accuracy
Strong predictive capability
Good generalization performance
Consistent classification results

##### Recommended Final Model
Although multiple models tied in test accuracy, Tuned KNN can be selected as the final recommended model.

# Suggestions to the Hospital to Reduce Heart Disease Risk and Prevent Life-Threatening Cases
1. Strengthen Early Screening Programs
One of the most important suggestions for hospitals is to strengthen early screening and cardiovascular risk assessment programs. Since the analysis showed that factors such as blood pressure, cholesterol, exercise-induced angina, and ECG-related features play an important role in predicting heart disease, hospitals should ensure routine screening for these indicators. Regular cardiac screening can help detect abnormalities at an early stage before they progress into severe disease. Special attention should be given to middle-aged and older patients as they may have higher risk. Early screening allows timely medical intervention, reduces complications, and can significantly lower the chances of fatal cardiac events.

2. Use Predictive Analytics for Early Risk Identification
Hospitals should consider integrating machine learning–based predictive models into their clinical workflow to support early identification of high-risk patients. Using patient medical attributes such as age, cholesterol levels, blood pressure, chest pain symptoms, and exercise-related indicators, predictive systems can help flag patients who may be at greater risk of heart disease. This can assist doctors in identifying silent or early-stage cases that may otherwise go unnoticed. Using predictive analytics as a decision-support tool can improve diagnosis, prioritize critical patients, and support preventive treatment planning, ultimately helping prevent life-threatening situations.

3. Increase Preventive Cardiac Checkup Programs
Hospitals should expand preventive cardiac health checkup programs and encourage routine cardiovascular evaluations, especially for patients with known risk factors such as hypertension, diabetes, obesity, smoking history, or family history of heart disease. Preventive checkups involving ECG testing, cholesterol screening, stress testing, and cardiac consultation can help identify risk early and reduce the probability of severe cardiac events. Instead of focusing only on treatment after symptoms worsen, hospitals should promote preventive monitoring as part of regular healthcare.

4. Improve Patient Awareness and Education
Another important suggestion is increasing awareness among patients about heart disease risks and prevention. Many serious cardiac cases can be reduced if patients recognize symptoms early and seek timely treatment. Hospitals should conduct awareness programs on maintaining heart health through healthy diet, regular exercise, smoking cessation, stress management, and regular medical checkups. Educating patients about warning signs such as chest pain, breathlessness, or exercise discomfort can lead to earlier consultation and reduce emergency situations. Prevention often begins with awareness.

5. Focus on High-Risk Patient Monitoring
Hospitals should establish stronger monitoring systems for patients identified as high risk. Patients with elevated blood pressure, high cholesterol, abnormal oldpeak values, exercise-induced angina, or major vessel abnormalities should receive closer follow-up and more proactive care. Instead of waiting for disease progression, hospitals can implement structured monitoring, regular follow-ups, and early interventions for these patients. Focused monitoring of vulnerable patients can help prevent complications such as heart attacks and improve long-term outcomes.

6. Promote Lifestyle Intervention Programs
Hospitals should invest more in lifestyle intervention programs aimed at reducing heart disease risk factors. Since many cardiovascular conditions are influenced by lifestyle habits, preventive programs focusing on nutrition, exercise, weight management, and stress reduction can play a major role in reducing disease burden. Hospitals can offer cardiac wellness counseling, diet guidance, and supervised preventive programs for at-risk patients. Encouraging healthier lifestyles can reduce future hospitalizations and life-threatening cardiac complications.
