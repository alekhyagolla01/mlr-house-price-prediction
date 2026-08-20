# 🏠 House Price Prediction

### 🚀 Machine Learning Web Application using Multiple Linear Regression

<p align="center">

**Predict house prices using Machine Learning + Flask + HTML**

<br>

[🌐 Live Demo](YOUR_RENDER_LINK_HERE) •
[💻 GitHub Repository](YOUR_GITHUB_REPOSITORY_URL)

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
| 🎨 HTML / CSS   | Frontend              |
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

```python
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

```python
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

This allows the Machine Learning model to work with date-related information numerically.

---

# 🎯 Features and Target

The dataset is divided into:

### X — Independent Variables

```python
X = df.iloc[:, :-1]
```

`X` contains the input features used for prediction.

### y — Dependent Variable

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

The dataset is divided into training and testing sets.

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

---

# 🤖 Model Training

Multiple Linear Regression is implemented using Scikit-learn.

```python
from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(X_train, y_train)
```

### Regression Coefficients

```python
reg.coef_
```

The coefficients represent the effect of each feature on the predicted house price.

### Intercept

```python
reg.intercept_
```

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

The predictions are compared with the actual house prices to evaluate model performance.

---

# 📉 Model Evaluation

Two important regression metrics are used:

### 1️⃣ RMSE

**Root Mean Square Error** measures the average magnitude of prediction errors.

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
Smaller Prediction Error
    ↓
Better Model Performance
```

---

# 📊 R² Score

R² measures how well the model explains the variation in the target variable.

```text
R² = 1 - [ Σ(yi - ŷi)² / Σ(yi - ȳ)² ]
```

Python:

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_test, test_prediction)
```

### Interpretation

```text
R² close to 1
      ↓
Better model fit
```

---

# 📋 Performance Metrics

The project evaluates the model using:

| Metric        | Purpose                            |
| ------------- | ---------------------------------- |
| Training Loss | Measures training prediction error |
| Testing Loss  | Measures testing prediction error  |
| RMSE          | Measures prediction error          |
| R² Score      | Measures goodness of fit           |

> 💡 Since this is a regression problem, **R² Score and RMSE** are more appropriate than classification accuracy.

---

# 💾 Saving the Model

The trained model is saved using Python's Pickle module.

```python
import pickle

with open("MLR.pkl", "wb") as file:
    pickle.dump(reg, file)
```

This allows the trained model to be reused without training it again.

---

# 📂 Loading the Model

The saved model can be loaded inside the Flask application.

```python
with open("MLR.pkl", "rb") as file:
    model = pickle.load(file)
```

---

# 🧪 Custom Prediction

Users can enter their own house information through the web application.

The process is:

```text
User Input
    ↓
HTML Form
    ↓
Flask
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

Flask acts as the bridge between the frontend and Machine Learning model.

```text
┌──────────────────┐
│   User Browser   │
└────────┬─────────┘
         ↓
┌──────────────────┐
│   HTML + CSS     │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Flask Backend    │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ MLR.pkl Model    │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Price Prediction │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Display Result   │
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

## 1. Clone Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

## 2. Open Project Directory

```bash
cd MLR-M-1
```

## 3. Create Virtual Environment

```bash
python -m venv .venv
```

## 4. Activate Environment

### Windows

```bash
.venv\Scripts\activate
```

## 5. Install Dependencies

```bash
pip install -r requirements.txt
```

## 6. Start Flask Application

```bash
python app.py
```

## 7. Open in Browser

```text
http://127.0.0.1:5000
```

---

# ☁️ Deployment

The application is deployed using **Render**.

### Production Start Command

```bash
gunicorn app:app
```

### Live Application

🚀 **[Open Live Application](YOUR_RENDER_LINK_HERE)**

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

Some possible improvements for the project:

* Add more Machine Learning algorithms
* Compare Linear Regression with Random Forest and Decision Tree
* Add data visualization dashboards
* Improve categorical encoding using One-Hot Encoding
* Add input validation
* Improve prediction accuracy
* Add responsive UI
* Add model performance charts
* Add database integration

---

# 👨‍💻 Author

### **Your Name**

💻 GitHub: **[Your GitHub Profile](YOUR_GITHUB_PROFILE_URL)**

---

# ⭐ Support

If you found this project useful:

### ⭐ Star the repository on GitHub

### 🍴 Fork the project

### 📢 Share it with others

---

<p align="center">

### 🏠 House Price Prediction

**Machine Learning • Flask • Multiple Linear Regression • Render**

Made with ❤️ using Python

</p>
