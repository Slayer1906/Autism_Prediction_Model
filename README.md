Here is a professional and comprehensive `README.md` file tailored for your resume project. It highlights the technical depth of your work (EDA, SMOTE, Hyperparameter Tuning) to stand out to recruiters.

---

# Autism Prediction Using Machine Learning

## 📌 Project Overview

This project focuses on the early detection of **Autism Spectrum Disorder (ASD)** using machine learning techniques. By analyzing behavioral features and demographic data, the model predicts the likelihood of ASD in individuals. This tool aims to assist healthcare professionals and individuals in making informed decisions about seeking formal clinical diagnosis.

The project implements a complete Data Science pipeline, including extensive **Exploratory Data Analysis (EDA)**, advanced **preprocessing** (handling outliers and class imbalance), **model optimization**, and **evaluation**.

## 📊 Dataset

The dataset contains **800 entries** with **21 features**, comprising:

* **Behavioral Scores:** 10 binary questions (A1_Score to A10_Score) based on ASD screening methods.
* **Demographics:** Age, Gender, Ethnicity, Country of Residence.
* **Medical/Family History:** Jaundice, Family member with ASD.
* **Target Variable:** Class/ASD (0 = No ASD, 1 = ASD).

## 🛠️ Tech Stack

* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn, XGBoost, Imbalanced-learn (SMOTE)
* **Model persistence:** Pickle

## ⚙️ Key Workflow Steps

### 1. Exploratory Data Analysis (EDA)

* Conducted univariate analysis to understand feature distributions.
* Visualized demographic spreads and correlation heatmaps.
* Identified and visualized outliers in numerical features (Age, Result).

### 2. Data Preprocessing

* **Data Cleaning:** Standardized country names and handled missing values in 'ethnicity' and 'relation' columns.
* **Outlier Treatment:** Implemented a custom function to cap outliers in 'Age' and 'Result' using the IQR method (replacing with median).
* **Encoding:** Applied `LabelEncoder` to convert categorical variables into numeric format.
* **Class Imbalance Handling:** Utilized **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the dataset, increasing the training size from 640 to 1030 samples.

### 3. Model Training & Selection

Implemented and compared three powerful classifiers:

1. **Decision Tree**
2. **Random Forest**
3. **XGBoost**

### 4. Hyperparameter Tuning

* Used `RandomizedSearchCV` with 5-fold cross-validation to optimize parameters for all three models.
* Tuned parameters included `n_estimators`, `max_depth`, `learning_rate`, `criterion`, and `min_samples_split`.

## 🏆 Results

After rigorous tuning, **Random Forest** was selected as the best-performing model.

* **Best Model Parameters:** `RandomForestClassifier(bootstrap=False, max_depth=10)`
* **Cross-Validation Accuracy:** ~93%
* **Test Set Accuracy:** 85%

**Classification Report (Test Data):**
| Class | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **0 (Non-ASD)** | 0.92 | 0.88 | 0.90 |
| **1 (ASD)** | 0.64 | 0.75 | 0.69 |

## 📁 File Structure

* `train.csv`: Raw dataset used for training and testing.
* `encoders.pkl`: Saved label encoder objects for transforming new data.
* `best_model.pkl`: Serialized final Random Forest model ready for deployment.

## 🚀 How to Run

1. **Install Dependencies:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost

```


2. **Run the script/notebook:**
Load the dataset and execute the preprocessing and training cells.
3. **Prediction:**
Use the saved model to make predictions on new data:
```python
import pickle
model = pickle.load(open('best_model.pkl', 'rb'))
prediction = model.predict(new_data)

```



## 🔮 Future Improvements

* Deploy the model as a web application using **Streamlit** or **Flask**.
* Experiment with Neural Networks for potentially higher accuracy.
* Collect more diverse data to improve recall on the positive class.

---

### 💡 Tips for your Resume:

* **Highlight SMOTE:** Mentioning you handled "Class Imbalance" shows you understand real-world data problems.
* **Mention Tuning:** Specifically state that you used `RandomizedSearchCV` to improve model performance, as it shows you go beyond just `model.fit()`.
* **Business Impact:** In your bullet points, frame it as "Developed a screening tool with 85% accuracy to assist in early ASD diagnosis."
