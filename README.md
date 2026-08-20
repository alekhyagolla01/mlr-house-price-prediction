# 🏠 House Price Prediction

### 🚀 Machine Learning Web Application using Multiple Linear Regression

<p align="center">

<b>Predict house prices using Machine Learning + Flask + HTML</b>

<br><br>

<a href="https://mlr-house-price-prediction-5.onrender.com">
  🌐 <b>Live Demo</b>
</a>

</p>

---

## 📌 Project Overview

**House Price Prediction** is a Machine Learning web application that predicts house prices based on multiple input features.

The project uses **Multiple Linear Regression** for prediction and **Flask** to connect the trained Machine Learning model with a user-friendly HTML interface.

The complete application is deployed online using **Render**.

### ✨ What this project demonstrates

* Data preprocessing
* Categorical data encoding
* Date feature extraction
* Multiple Linear Regression
* Train-Test Split
* Model evaluation
* RMSE calculation
* R² Score calculation
* Model serialization using Pickle
* Flask integration
* Web-based prediction
* Cloud deployment

---

# 🌐 Live Demo

<p align="center">

<a href="https://mlr-house-price-prediction-5.onrender.com">

## 🚀 Open House Price Prediction App

</a>

</p>

🔗 **Live Application:**
https://mlr-house-price-prediction-5.onrender.com

---

# 🧠 Machine Learning Pipeline

```text
                 ┌──────────────────┐
                 │      Dataset     │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ Data Preprocess  │
                 └────────┬─────────┘
                          ↓
             ┌──────────────────────────┐
             │ City & Country Mapping   │
             └────────────┬─────────────┘
                          ↓
                 ┌──────────────────┐
                 │ Date Processing  │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │   X and y Split  │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ Train-Test Split │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ LinearRegression │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ Model Evaluation │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │    Save Model    │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │   Flask Backend  │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │  HTML Frontend   │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ Render Deployment│
                 └──────────────────┘
```

---

# 🛠️ Tech Stack

| Technology      | Purpose               |
| --------------- | --------------------- |
| 🐍 Python       | Programming Language  |
| 🐼 Pandas       | Data Processing       |
| 🔢 NumPy        | Numerical Operations  |
| 🤖 Scikit-learn | Machine Learning      |
| 🌐 Flask        | Backend Web Framework |
| 🎨 HTML         | Frontend              |
| 🎨 CSS          | Styling               |
| 💾 Pickle       | Model Serialization   |
| 📊 Matplotlib   | Visualization         |
| 🔧 Git          | Version Control       |
| 🐙 GitHub       | Source Code Hosting   |
| ☁️ Render       | Deployment            |

---

# 📊 Machine Learning Model

The application uses **Multiple Linear Regression**.

### Mathematical Representation

```text
y = m₁x₁ + m₂x₂ + m₃x₃ + ... + mₙxₙ + c
```

Where:

```text
y  → Predicted House Price

x₁, x₂, ..., xₙ → Input Features

m₁, m₂, ..., mₙ → Regression Coefficients

c → Intercept
```

---

# 🔄 Data Preprocessing

Before training the model, the dataset is converted into a format suitable for Machine Learning.

## 🏙️ City Mapping

Categorical city values are converted into numerical values.

```text
City A → 0
City B → 1
City C → 2
```

Example:

```python
df["City"] = df["City"].map(city_mapping)
```

---

## 🌍 Country Mapping

Country values are also converted into numerical representations.

```text
Country A → 0
Country B → 1
Country C → 2
```

Example:

```python
df["Country"] = df["Country"].map(country_mapping)
```

---

# 📅 Date Feature Engineering

The date column is converted into useful numerical features.

For example:

```text
Original Date
     ↓
2026-08-20
     ↓
┌─────────────┐
│ Year  = 2026│
│ Month = 8   │
│ Day   = 20  │
└─────────────┘
```

The date information is converted into numerical values so that it can be used by the Machine Learning model.

---

# 🎯 Features and Target

The dataset is divided into independent and dependent variables.

## X — Independent Variables

```python
X = df.iloc[:, :-1]
```

`X` contains the input features used by the model.

## y — Dependent Variable

```python
y = df.iloc[:, -1]
```

`y` contains the target value — the house price.

```text
X → Input Features

y → House Price
```

---

# ✂️ Train-Test Split

The dataset is divided into training and testing data.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### Dataset Distribution

```text
             Complete Dataset
                    │
          ┌─────────┴─────────┐
          ↓                   ↓
      80% Training         20% Testing
          │                   │
          ↓                   ↓
      Train Model         Evaluate Model
```

The training data is used to train the model, while the testing data is used to evaluate its performance.

---

# 🤖 Model Training

Multiple Linear Regression is implemented using Scikit-learn.

```python
from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(X_train, y_train)
```

---

## 📐 Regression Coefficients

```python
reg.coef_
```

The coefficients represent the relationship between the input features and the predicted house price.

---

## 📍 Intercept

```python
reg.intercept_
```

The intercept represents the constant value in the regression equation.

---

# 📈 Model Prediction

## Training Prediction

```python
train_prediction = reg.predict(X_train)
```

## Testing Prediction

```python
test_prediction = reg.predict(X_test)
```

The predicted values are compared with the actual values to evaluate the model.

---

# 📉 RMSE — Root Mean Square Error

RMSE measures the difference between actual and predicted values.

### Formula

```text
RMSE = √[ Σ(yi - ŷi)² / n ]
```

Where:

```text
yi  → Actual value

ŷi  → Predicted value

n   → Number of observations
```

### Interpretation

```text
Lower RMSE
     ↓
Lower Prediction Error
     ↓
Better Prediction Performance
```

---

# 📊 R² Score

R² Score measures how well the model explains the variation in the target variable.

### Formula

```text
R² = 1 - [ Σ(yi - ŷi)² / Σ(yi - ȳ)² ]
```

### Python Implementation

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_test, test_prediction)
```

### Interpretation

```text
R² close to 1
      ↓
Better Model Fit
```

---

# 📋 Performance Metrics

| Metric        | Purpose                            |
| ------------- | ---------------------------------- |
| Training Loss | Measures training prediction error |
| Testing Loss  | Measures testing prediction error  |
| RMSE          | Measures prediction error          |
| R² Score      | Measures goodness of fit           |

> 💡 Since this is a regression problem, **R² Score and RMSE** are more appropriate than classification accuracy.

---

# 💾 Model Saving

The trained Machine Learning model is saved using Python's Pickle module.

```python
import pickle

with open("MLR.pkl", "wb") as file:
    pickle.dump(reg, file)
```

This allows the trained model to be reused without training it again.

---

# 📂 Model Loading

The saved model can be loaded inside the Flask application.

```python
with open("MLR.pkl", "rb") as file:
    model = pickle.load(file)
```

---

# 🧪 Custom Prediction

Users can enter their own house information through the web application.

The prediction process is:

```text
User Input
    ↓
HTML Form
    ↓
Flask Backend
    ↓
Loaded ML Model
    ↓
Prediction
    ↓
Predicted House Price
```

Example:

```python
prediction = model.predict(test_data)
```

---

# 🌐 Flask Web Application

Flask connects the HTML frontend with the Machine Learning model.

```text
┌──────────────────┐
│   User Browser   │
└────────┬─────────┘
         ↓
┌──────────────────┐
│    HTML + CSS    │
└────────┬─────────┘
         ↓
┌──────────────────┐
│   Flask Backend  │
└────────┬─────────┘
         ↓
┌──────────────────┐
│    MLR.pkl       │
│   ML Model       │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Price Prediction │
└────────┬─────────┘
         ↓
┌──────────────────┐
│  Display Result  │
└──────────────────┘
```

---

# 📁 Project Structure

```text
MLR-M-1/
│
├── 📄 app.py
├── 📄 model.py
├── 📊 house price.csv
├── 🤖 MLR.pkl
├── 📦 requirements.txt
├── 📖 README.md
│
└── templates/
    └── 📄 index.html
```

---

# 💻 Run the Project Locally

## 1️⃣ Clone Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

---

## 2️⃣ Open Project Directory

```bash
cd MLR-M-1
```

---

## 3️⃣ Create Virtual Environment

```bash
python -m venv .venv
```

---

## 4️⃣ Activate Virtual Environment

### Windows

```bash
.venv\Scripts\activate
```

---

## 5️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 6️⃣ Start Flask Application

```bash
python app.py
```

---

## 7️⃣ Open in Browser

```text
http://127.0.0.1:5000
```

---

# ☁️ Render Deployment

The application is deployed using **Render**.

### 🚀 Live Application

<a href="https://mlr-house-price-prediction-5.onrender.com">

**👉 Open House Price Prediction Application**

</a>

### Production Start Command

```bash
gunicorn app:app
```

### Deployment Flow

```text
GitHub Repository
       ↓
     Render
       ↓
Build Application
       ↓
Install Requirements
       ↓
Start Gunicorn
       ↓
Flask Application
       ↓
Live Website
```

---

# 📦 Requirements

```text
Flask
pandas
numpy
scikit-learn
matplotlib
gunicorn
```

---

# ⭐ Key Features

```text
✅ Multiple Linear Regression

✅ City Mapping

✅ Country Mapping

✅ Date Processing

✅ Train-Test Split

✅ Training Evaluation

✅ Testing Evaluation

✅ RMSE Calculation

✅ R² Score

✅ Pickle Model Saving

✅ Model Loading

✅ Custom Prediction

✅ Flask Backend

✅ HTML/CSS Frontend

✅ Render Deployment
```

---

# 🔮 Future Improvements

Possible improvements for this project include:

* 🌳 Add Random Forest Regression
* 🌲 Add Decision Tree Regression
* 📊 Add data visualization dashboards
* 🔢 Improve categorical encoding
* 🛡️ Add input validation
* 🎯 Improve prediction accuracy
* 📱 Create a responsive UI
* 📈 Add model performance graphs
* 🗄️ Add database integration
* 🔐 Add user authentication

---

# 👨‍💻 Author

### Your Name

💻 **GitHub:**
[Your GitHub Profile](YOUR_GITHUB_PROFILE_URL)

---

# ⭐ Support

If you found this project useful:

⭐ **Star the repository**

🍴 **Fork the project**

📢 **Share the project**

---

<p align="center">

## 🏠 House Price Prediction

**Machine Learning • Multiple Linear Regression • Flask • Render**

Made with ❤️ using Python

<br><br>

<a href="https://mlr-house-price-prediction-5.onrender.com">

🚀 <b>Try the Live Application</b>

</a>

</p>
