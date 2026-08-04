# Flight Delay Predictor

A machine-learning web application that estimates the probability of a flight delay using flight schedules, route information, airport traffic patterns, historical delay behaviour, engineered temporal features, and optional weather information.

The final application uses an **Advanced XGBoost classifier** and provides an interactive prediction interface built with **Streamlit**.

---

## Project Overview

Flight delays are influenced by several factors, including departure time, airline, airport congestion, route characteristics, historical delay patterns, and weather conditions.

This project builds an end-to-end machine-learning pipeline to estimate the probability that a scheduled flight will be delayed.

The project includes:

- Data preprocessing and cleaning
- Exploratory data analysis
- Feature engineering
- Weather-data integration
- Handling of categorical and missing data
- Multiple machine-learning models
- Model tuning and comparison
- Probability-based flight delay prediction
- Interactive Streamlit web application

---

## Live Application

The application allows users to enter:

- Airline
- Origin airport
- Destination airport
- Flight date
- Scheduled departure time
- Flight distance
- Optional weather conditions

The trained model then estimates:

- **Delay Probability**
- **On-Time Probability**
- **Delay Risk Level**

> Deployment link will be added here after deployment.

---

## Machine Learning Models

Several models were evaluated during development:

1. Logistic Regression
2. Base XGBoost
3. Tuned XGBoost
4. Advanced XGBoost

The **Advanced XGBoost model** is used by the final prediction application.

### Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6088 | 0.2660 | 0.6239 | 0.3729 | 0.6549 | 0.2935 |
| Base XGBoost | 0.6541 | 0.2990 | 0.6362 | 0.4068 | 0.7017 | 0.3620 |
| Tuned XGBoost | 0.6942 | 0.3245 | 0.5914 | 0.4190 | 0.7124 | 0.3835 |
| **Advanced XGBoost** | **0.6967** | **0.3266** | **0.5905** | **0.4206** | **0.7118** | **0.3805** |

Because flight-delay prediction involves an imbalanced target, evaluation was not based on accuracy alone. Metrics such as **Recall, F1 Score, ROC-AUC, and PR-AUC** were also considered.

---

## Feature Engineering

The final model uses approximately **40 predictive features** derived from flight, route, airport, temporal, historical, and weather information.

Examples include:

### Temporal Features

- Departure hour
- Departure minute
- Day of week
- Month
- Weekend indicator
- Peak-hour indicator
- Season
- Quarter
- Time-of-day categories

### Route & Airport Features

- Airline
- Origin airport
- Destination airport
- Flight distance
- Distance category
- Origin traffic
- Destination traffic
- Route frequency

### Historical Delay Features

Historical patterns were incorporated to help the model capture differences in delay behaviour between airlines, airports, and routes.

### Weather Features

Optional weather information includes:

- Temperature
- Humidity
- Atmospheric pressure
- Wind speed
- Weather condition

The application can still generate predictions when weather information is unavailable.

---

## Weather Integration

Airport information was mapped to available weather cities.

The weather datasets contain historical:

- Temperature
- Humidity
- Pressure
- Wind speed
- Weather descriptions

Weather information was incorporated where matching airport-city data was available.

---

## Dataset

The project uses historical flight information together with supporting airline, airport, and weather datasets.

The flight dataset contains information such as:

- Airline
- Flight number
- Origin airport
- Destination airport
- Scheduled departure
- Departure delay
- Arrival information
- Distance
- Cancellation information
- Airline delay
- Weather delay
- Late aircraft delay

After preprocessing and cleaning, approximately **294K flight records** were used in the working dataset.

The target variable identifies whether a flight was delayed.

---

## Technology Stack

### Programming

- Python

### Data Processing

- Pandas
- NumPy

### Machine Learning

- Scikit-learn
- XGBoost
- Imbalanced-learn

### Model Persistence

- Joblib

### Web Application

- Streamlit

### Development

- Jupyter Notebook
- Visual Studio Code
- Git
- GitHub

---

## Project Structure

```text
Flight-Delay-Predictor/
│
├── app.py
├── train_model.py
├── utils.py
├── requirements.txt
├── README.md
│
├── assets/
│
├── data/
│   ├── airlines.csv
│   ├── airports.csv
│   ├── flights.csv
│   ├── temperature.csv
│   ├── humidity.csv
│   ├── pressure.csv
│   ├── wind_speed.csv
│   └── weather_description.csv
│
├── models/
│   ├── advanced_xgboost_model.pkl
│   ├── preprocessor.pkl
│   └── model_results.csv
│
└── notebooks/
    └── flight_delay_prediction.ipynb
```

> The exact filenames may vary depending on the final repository structure.

---

## Running the Project Locally

### 1. Clone the repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd Flight-Delay-Predictor
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Start the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## Prediction Workflow

The application follows this general pipeline:

```text
User Flight Information
        ↓
Input Validation
        ↓
Feature Engineering
        ↓
Preprocessing
        ↓
Advanced XGBoost Model
        ↓
Delay Probability
        ↓
Risk Classification
        ↓
Prediction Dashboard
```

---

## Application Output

For each flight, the application displays:

### Delay Probability

Estimated probability that the flight will experience a delay.

### On-Time Probability

Estimated probability that the flight will operate without a delay.

### Risk Level

Predictions are presented using an easy-to-understand risk category such as:

- 🟢 Low
- 🟡 Moderate
- 🔴 High

---

## Important Note

This project produces a **probabilistic machine-learning estimate** based on historical flight and weather patterns.

It is **not a real-time airline tracking system** and should not be interpreted as a guarantee that a particular flight will or will not be delayed.

Actual flight operations can be affected by real-time factors that are not available to the model.

---

## Possible Future Improvements

Potential future extensions include:

- Real-time weather API integration
- Live flight-status data
- Additional airport congestion information
- More recent flight datasets
- Delay-duration prediction
- Explainable AI / feature importance
- Automated model retraining
- Expanded geographic and weather coverage

---

## Project Goal

The goal of this project was to develop an end-to-end machine-learning application rather than only train a classification model.

The project demonstrates:

- Data preprocessing
- Feature engineering
- Handling imbalanced classification
- Model comparison
- Hyperparameter tuning
- Machine-learning evaluation
- Model persistence
- Interactive application development
- End-to-end ML deployment workflow

---

## Author

**Uttra Manhas**

B.Tech Computer Science & Engineering  
Interested in Data Science, Machine Learning and Data Analytics
