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

The age variable showed that a large number of patients were concentrated in middle-aged groups, indicating that heart disease cases in the dataset were more common among middle-aged and older individuals. This aligns with medical understanding that heart disease risk often increases with age.
*The resting blood pressure distribution showed reasonable spread, but a few observations appeared on the higher side, suggesting possible extreme blood pressure values. This hinted at potential outliers that would later require treatment.

The serum cholesterol variable also showed variability, with some patients having noticeably high cholesterol levels. Since high cholesterol is often linked to cardiovascular risk, this variable appeared potentially important for prediction. Some extreme values also suggested possible outliers.

The distribution of maximum heart rate achieved appeared relatively moderate, with observations spread around central values. This feature seemed fairly well distributed without major abnormalities.

The oldpeak variable, which measures ST depression, showed some skewness in its distribution, indicating non-symmetrical spread and possible presence of unusual values.

Overall, univariate analysis of numerical features helped identify distribution patterns, feature variability, and potential outliers. These findings later supported decisions related to outlier detection and preprocessing.

####  1.2 Categorical Variable Analysis
For categorical variables, countplots were used to visualize category frequencies and understand how observations were distributed across classes. The categorical variables analyzed included sex, chest pain type, number of major vessels, fasting blood sugar, resting ECG results, exercise induced angina, slope of peak exercise ST segment, and thal.

The purpose of this analysis was to examine category frequencies, detect imbalance in categories, and understand how often certain medical conditions or attributes occurred in the patient population.

The sex feature showed the distribution of male and female patients, helping assess whether one gender was more represented in the dataset.

The chest pain type variable showed variation across categories, with some chest pain types occurring much more frequently than others. This is important because different chest pain types may carry different levels of heart disease risk.

The number of major vessels feature revealed how vessel blockages were distributed across patients and suggested possible relevance in disease prediction.

The fasting blood sugar feature helped observe the proportion of patients with elevated glucose levels.

The resting ECG categories showed variation in heart electrical activity conditions.

The exercise induced angina feature was particularly useful because the presence or absence of angina during exercise may indicate cardiac issues.

Similarly, the slope of peak exercise ST segment and thal showed different category distributions, and these differences may contribute significantly to classification.

From these plots, it was observed that certain chest pain categories appeared more frequently, thal categories showed noticeable variation, and some categorical features appeared likely to be strong disease indicators.

Thus, categorical univariate analysis provided valuable insights into feature frequencies, possible imbalance, and medically important category patterns.

### Bivariate Analysis
####  Numerical Features vs Target
To analyze the relationship between numerical variables and the target variable heart_disease_present, boxplots were used to compare the distribution of each continuous feature across the two target classes: patients without heart disease (0) and patients with heart disease (1). Boxplots are particularly useful because they display median values, spread, interquartile ranges, and outliers, making it easier to compare how each feature behaves across both groups.

Age vs Heart Disease
The boxplot for age showed that patients with heart disease generally tended to belong to slightly higher age groups compared to patients without heart disease. The median age for the disease group appeared higher, suggesting age may have a positive relationship with heart disease risk. Although there was overlap between the two classes, the shift in central tendency indicates age may contribute to prediction.

Resting Blood Pressure vs Heart Disease
The resting blood pressure boxplot showed that patients diagnosed with heart disease tended to have slightly higher blood pressure values compared to those without disease. The disease group also showed a wider spread and some extreme observations, suggesting greater variability in blood pressure among affected patients. This supports the medical relevance of blood pressure as a cardiovascular risk factor.

Serum Cholesterol vs Heart Disease
For serum cholesterol, the boxplot showed noticeable differences between the two groups. Patients with heart disease appeared to have relatively higher cholesterol levels, with a broader spread and presence of outliers. The larger variation and elevated values in the disease group suggest cholesterol may be an important predictor of heart disease.

Maximum Heart Rate Achieved vs Heart Disease
The boxplot for maximum heart rate achieved showed a contrasting pattern, where patients without heart disease generally reached higher heart rates compared to those with disease. The heart disease group showed relatively lower median values, which may indicate reduced cardiac performance among affected patients. This difference suggests maximum heart rate has strong predictive relevance.

Oldpeak (ST Depression) vs Heart Disease
Among all numerical variables, oldpeak showed one of the clearest distinctions between classes. Patients with heart disease generally exhibited higher ST depression values than patients without disease. The disease group showed higher medians and greater spread, indicating abnormal ST depression is strongly associated with heart disease presence. This suggests oldpeak may be one of the most influential predictors.

####  Categorical Features vs Target
To analyze relationships between categorical variables and heart disease, countplots were generated using the target variable as the hue. These plots helped compare how category frequencies differed between patients with and without heart disease, allowing visual identification of features associated with higher disease risk.

Sex vs Heart Disease
The graph for sex showed variation in disease prevalence across gender categories. One gender appeared more represented among heart disease cases, suggesting sex may influence risk and contribute to prediction.

Chest Pain Type vs Heart Disease
The chest pain type countplot showed some of the strongest class differences. Certain chest pain categories had noticeably higher counts among patients with heart disease, while others appeared more common in non-disease cases. This strong separation suggests chest pain type is an important indicator and potentially a high-impact predictive feature.

Number of Major Vessels vs Heart Disease
The graph for number of major vessels showed a clear trend where certain vessel categories had higher association with heart disease. Patients with greater vessel involvement appeared more likely to belong to the disease class. This aligns with medical understanding and makes this feature highly significant.

Exercise Induced Angina vs Heart Disease
The countplot for exercise induced angina showed a strong contrast between the two classes. Patients experiencing exercise-induced angina appeared more frequently in the heart disease group, indicating a strong relationship with disease presence. This was one of the clearest categorical indicators.

Thal vs Heart Disease
The thal feature also showed strong variation between classes. Some thal categories were much more associated with heart disease than others, suggesting this variable carries substantial predictive power.

Resting ECG and ST Segment Slope vs Heart Disease
These variables showed moderate differences across target classes. While separation was not as strong as chest pain or thal, the graphs still suggested potential association with disease outcomes.

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

Capping was preferred because:

It retains all patient observations.
It avoids losing potentially useful medical information.
It reduces the influence of extreme values.
It is safer for small datasets than deleting rows.
It improves robustness without heavily altering distributions.
Post-treatment boxplots showed reduced spread from extreme observations, confirming that outlier influence was successfully controlled.

Overall, outlier treatment improved data quality while preserving dataset size and important patient information.
