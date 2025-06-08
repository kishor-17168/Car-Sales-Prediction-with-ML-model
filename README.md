# 🚗 Used Car Price Prediction using Machine Learning Models

## 📑 Table of Contents
1. [📘 Introduction](#1--introduction)  
2. [📌 Problem Statement & Objectives](#2--problem-statement--objectives)  
3. [🛠️ Tools and Technologies](#3--tools-and-technologies)  
4. [🚀 How to Use This Project](#4--how-to-use-this-project)  
5. [📂 Dataset Description](#5--dataset-description)  
6. [📊 Data Analysis](#6--data-analysis)  
   - [6.1 Data Cleaning & Preprocessing](#61--data-cleaning--preprocessing)  
   - [6.2 Exploratory Data Analysis (EDA)](#62--exploratory-data-analysis-eda)  
   - [6.3 Feature Engineering](#63--feature-engineering)  
   - [6.4 Model Training & Evaluation](#64--model-training--evaluation)  
7. [💡 Result & Discussion](#7--result--discussion)  
8. [📃 Conclusion & Recommendation](#8--conclusion--recommendation)  

---

## 1- Introduction


This project uses machine learning to predict the price of a used car. The model is trained on a dataset containing historical car sales data, enabling it to estimate the price of a car based on its features such as make, model, year, mileage, fuel type, and transmission.

Accurately predicting used car prices is valuable for buyers, sellers, and dealerships to make informed decisions in a competitive market. By leveraging data-driven techniques, this project demonstrates how machine learning can help automate and improve the pricing process, bringing transparency and efficiency to the used car market.


## 2- Problem Statement & Objectives

The problem statement for this project is to predict the price of a used car based on a set of features such as years used, mileage, engine size, power, kilometers driven, and number of seats. Accurately estimating car prices is challenging due to the variety of factors affecting market value and the variability across different car models and conditions.

### Objectives:
- To analyze and preprocess the used car dataset for machine learning suitability.
- To explore the relationships between car features and their selling price through exploratory data analysis.
- To build and compare multiple machine learning models for price prediction.
- To select the best performing model based on evaluation metrics and provide actionable insights.



## 3- Tools and Technologies
- **Programming Language:** Python  
- **Data Manipulation & Analysis:** Pandas, NumPy  
- **Data Visualization:** Matplotlib, Seaborn  
- **Machine Learning:** Scikit-learn  
- **Development Environment:** Jupyter Notebook / VS Code  
- **Version Control:** GitHub  
- **Other Tools:** Google Colab, Anaconda

## 4- How to Use This Project

Follow the steps below to run the project on your local machine:

### 1. Download the files from here 
```bash
https://github.com/kishor-17168/Car-Sales-Prediction-with-ML-model.git

```

### 2. Install Dependencies
Make sure you have Python installed. Then install the required libraries using:

```bash
pip install -r requirements.txt
```

> If `requirements.txt` is not available, install manually:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 3. Run the Project
Open the Jupyter Notebook or run the script:

```bash
jupyter UsedCarPricePrediction.ipynb
```
### 4. Add the Dataset

Make sure the dataset file (e.g., used_cars.csv) is placed in the correct project folder.
If needed, update the dataset path in the code.



---

You can now explore the data, visualize insights, and test machine learning models for predicting used car prices.

## 5- Dataset Description

The dataset used in this project contains historical data of used cars, including various features that influence the selling price. It provides the necessary information to train machine learning models to predict car prices based on their characteristics.

### 📌 Source:
The dataset was downloaded from an open-source internet resource. While it is not from a verified official source, the project focuses on demonstrating a real-world machine learning workflow that can be applied to any similar used car dataset.

### 📁 File Name:
`used_cars.csv`

### 📊 Key Features (after cleaning):
| Feature            | Description                                                 |
|--------------------|-------------------------------------------------------------|
| `Location`         | City or region where the car is listed                      |
| `Year`             | Year of manufacture of the car                              |
| `Kilometers_Driven`| Total distance driven (affects wear and tear)              |
| `Fuel_Type`        | Type of fuel used (e.g., Petrol, Diesel, CNG)              |
| `Transmission`     | Gear type — Manual or Automatic                             |
| `Owner_Type`       | Indicates number of previous owners                         |
| `Seats`            | Number of seats in the car                                  |
| `Company`          | Brand of the car (e.g., Maruti, Hyundai)                    |
| `Mileage(km/kg)`   | Fuel efficiency of the car                                  |
| `Engine(CC)`       | Engine size in cubic centimeters                            |
| `Power(bhp)`       | Engine power in brake horsepower                            |
| `Price`            | ✅ Target variable — final selling price of the car         |


## 6- Data Analysis

### 6.1- Data Cleaning & Preprocessing
#### ✅ Check for missing values
```python
car_dataset.isnull().sum()
```
#### 🧹 Drop unnecessary columns

```python
car_dataset.drop(["Unnamed: 0", "New_Price"], axis=1, inplace=True)

# 🧼 Remove rows with null values
car_dataset = car_dataset.dropna()
```

#### 🏷️ Extract numeric info and brand name from string columns

```python
for i in range(car_dataset.shape[0]):
    car_dataset.at[i, 'Company'] = car_dataset['Name'][i].split()[0]
    car_dataset.at[i, 'Mileage(km/kg)'] = car_dataset['Mileage'][i].split()[0]
    car_dataset.at[i, 'Engine(CC)'] = car_dataset['Engine'][i].split()[0]
    car_dataset.at[i, 'Power(bhp)'] = car_dataset['Power'][i].split()[0]

```
#### 🧾 Drop original string columns after transformation

```python
car_dataset.drop(["Name", "Mileage", "Engine", "Power"], axis=1, inplace=True)
```

#### 🔢 Convert the new columns to numeric (float)
``` python
car_dataset['Mileage(km/kg)'] = car_dataset['Mileage(km/kg)'].astype(float)
car_dataset['Engine(CC)'] = car_dataset['Engine(CC)'].astype(float)
car_dataset['Power(bhp)'] = car_dataset['Power(bhp)'].astype(float)

```



### 6.2- Exploratory Data Analysis (EDA)

In this section, we explore the dataset to understand the distribution and relationships of key features with the target variable, **Price**.

- **Categorical Features:**  
  We analyzed categories such as **Location**, **Fuel Type**, **Transmission**, **Owner Type**, and **Car Company** to see their frequency distributions and impact on car prices.

- **Visualizations:**  
  Using cat plots and heat map, we observed:
  - Cars in metropolitan locations tend to have higher prices.
  - Diesel and petrol cars dominate the dataset, with diesel cars generally priced higher.
  - Manual transmission cars are more common than automatic.
  - The majority of cars are first-owner vehicles.
  - Top companies by car count include Maruti, Hyundai, and Honda.

- **Encoding:**  
  To prepare the data for modeling:
  - One-hot encoding was applied to nominal categorical variables (Location, Fuel Type, Transmission).
  - Ordinal encoding was applied to Owner Type, reflecting the hierarchy of ownership.

This analysis helped identify important features and guided the feature engineering and modeling steps.

---

For the full code, detailed plots, and analysis, see the [Exploratory Data Analysis Notebook](UsedCarPricePrediction.ipynb).


### 6.3- Feature Engineering
Key transformations done to prepare data for modeling:

- Extracted car company from `Name`.
- Converted `Mileage`, `Engine`, and `Power` to numeric.
- Encoded categorical features (one-hot and ordinal).
- Dropped unused columns `Unnamed:	0`, `New_Price`.


These steps improved the model’s understanding and performance.

### 6.4- Model Training & Evaluation
Multiple regression models were trained to predict used car prices, with performance evaluated on both training and testing datasets:

- **Linear Regression:**  
  Achieved training accuracy of ~70% and testing accuracy of ~73%. It provided a good baseline model.

- **Lasso Regression:**  
  Showed lower accuracy (~57% training, ~60% testing), indicating it may have underfit the data with strong regularization.

- **Ridge Regression:**  
  Performed similarly to Linear Regression with ~70% training and ~72% testing accuracy.

- **Decision Tree Regressor:**  
  Nearly perfect accuracy on training data (~99.9%) but lower testing accuracy (~83.6%), indicating some overfitting. Cross-validation mean score was ~76%.

- **Random Forest Regressor:**  
  Delivered the best results with ~98% accuracy on training and ~91% on testing data. Cross-validation mean score was ~87%, showing good generalization.

**Conclusion:**  
Random Forest Regressor was selected as the final model due to its superior predictive accuracy and balanced performance between training and testing sets.

---

*Note:* Accuracy here refers to the coefficient of determination \( R^2 \) score, indicating how well the model explains variance in the target variable.

## 7- Result & Discussion
#### 🔍 Model Performance Summary

| Model                 | Training Accuracy (R²) | Testing Accuracy (R²) | Cross-Validation (Mean R²) |
|----------------------|------------------------|------------------------|-----------------------------|
| Linear Regression     | 0.7006                 | 0.7256                 | —                           |
| Lasso Regression      | 0.5660                 | 0.6015                 | —                           |
| Ridge Regression      | 0.7003                 | 0.7253                 | —                           |
| Decision Tree Regressor       | 0.9999                 | 0.8361                 | 0.7628                      |
| Random Forest Regressor        | 0.9820                 | 0.9123                 | 0.8683                      |

#### 💬 Discussion

- **Random Forest Regressor** delivered the **highest test accuracy** (91.2%) and strong cross-validation score (86.8%), making it the **most reliable model** for generalizing to unseen data.
- **Decision Tree Regressor** achieved nearly perfect training accuracy (~99.99%), but lower test accuracy and a cross-validation score of 76.3%, indicating **overfitting**.
- **Linear and Ridge Regression** provided consistent but moderate performance (~70–73%) with no cross-validation conducted. These models are simpler and serve as solid baselines.
- **Lasso Regression** showed the weakest performance due to strong regularization, unable to fully capture relationships in the dataset.


The high performance of **Random Forest Regressor** suggests that ensemble models can effectively learn from both **categorical** and **continuous** features in price prediction tasks.

> **Note:** Cross-validation was applied only to Decision Tree Regressor and Random Forest Regressor models, as they are more prone to overfitting. Simpler models like Linear, Ridge, and Lasso were evaluated using the train-test split, which provided sufficiently stable results for comparison.


## 8- Conclusion & Recommendation


#### ✅ Conclusion

This project successfully met its core objectives:

- The **used car dataset** was thoroughly cleaned and preprocessed — handling missing values, extracting relevant features (e.g., engine, power, mileage), and converting data types — ensuring it was suitable for machine learning models.
- Through **exploratory data analysis (EDA)**, we discovered meaningful patterns such as the positive correlation between engine size, power, and price, while factors like age and kilometers driven had a negative influence.
- Multiple machine learning models — including **Linear Regression, Ridge, Lasso, Decision Tree Regressor,** and **Random Forest Regressor** — were developed and evaluated.
- Among all models, **Random Forest Regressor** emerged as the **best performer**, achieving a testing R² score of **0.912** and strong generalization as shown by its **cross-validation score of 0.868**.
  
These outcomes demonstrate that machine learning can effectively predict used car prices when provided with relevant and well-preprocessed features.

#### 💡 Recommendations

- **Random Forest Regressor** is recommended for deployment due to its superior accuracy and robustness. It balances bias and variance well and can handle both categorical and numerical features efficiently.
- For further improvement, consider tuning hyperparameters or using **Gradient Boosting** or **XGBoost** for potential gains.
- Incorporate **real-time or larger-scale data** for production models to enhance prediction accuracy and scalability.
- Deploy the model with a user-friendly interface (e.g., a web app) to allow non-technical users to input car details and get instant price predictions.


---

### 👤 Author

**[Kishor Marandy]**  
📧 [christoferkishor@gmail.com]  
🔗 [LinkedIn Profile](https://www.linkedin.com/in/your-profile/)  
💻 [GitHub Profile](https://github.com/your-github-username)

