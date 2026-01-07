# ⚓ FYP Voyage Delay Prediction System

A machine learning-based system for predicting maritime voyage delays using AIS (Automatic Identification System) data, weather conditions, and global events. This project integrates multiple data sources to provide accurate delay predictions and risk assessments for maritime operations.

---

## 🌐 Live Dashboard

**🚀 Try the app now:** [Voyage Delay Prediction Dashboard on Streamlit Cloud](https://fyp-tp068218-voyage-delay-prediction-dashboard.streamlit.app/)

### 🧪 Test Cases

Want to try the app? Use these sample test cases:

#### **Test Case 1**
```
Route Option: Major Ports
Departure Port: Singapore
Destination Port: Hong Kong 
Departure Date: 2024-11-10
Departure Time: 08:00
```

---

#### **Test Case 2**
```
Route Option: Major Ports
Departure Port: Shanghai
Destination Port: Rotterdam
Departure Date: 2024-12-10
Departure Time: 14:00
```

---


**💡 Tips for Testing:**
- There is a daily limit to the AI recommendation requests (20/day) so if you faced an error, please try again the next day

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Data Sources](#-data-sources)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Technologies Used](#-technologies-used)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

This system analyzes maritime vessel movements and predicts potential voyage delays by combining:
- **AIS vessel tracking data** (2020-2024)
- **ERA5 weather data** (wind, waves, temperature, precipitation)
- **GDELT global events data** (geopolitical events affecting maritime trade)

The system provides both classification (Delayed/On-Time) and regression (delay duration in hours) predictions, along with AI-powered insights and recommendations powered by Google's Gemini AI.

### Key Capabilities
- 🔮 **Dual Prediction Models**: Classification (delayed/on-time) and regression (delay duration)
- 📊 **Risk Assessment**: Low, Moderate, or High risk categorization
- 🌤️ **Weather Impact Analysis**: Real-time weather condition visualization
- 🤖 **AI-Powered Insights**: Intelligent recommendations using Gemini AI
- 📈 **Interactive Visualizations**: Feature importance and impact charts

---

## ✨ Features

- **Real-time Delay Prediction**: Predict whether a voyage will be delayed and estimate delay duration
- **Risk Assessment**: Low, Moderate, or High risk classification with confidence scores
- **Weather Impact Analysis**: Visualize how weather conditions affect voyage delays
- **AI-Powered Insights**: Get intelligent recommendations using Google's Gemini AI
- **Interactive Dashboard**: User-friendly Streamlit interface with dynamic visualizations
- **Feature Importance Analysis**: Understand which factors most influence predictions
- **Historical Data Integration**: Leverages 4 years of maritime data (2020-2023)

---

## 📁 Project Structure

```
.
├── 📊 Data Files (via Git LFS)
│   ├── train_2020_2023.csv              # Training dataset (2020-2023)
│   ├── test_2024.csv                    # Test dataset (2024)
│   ├── era5_weather_2024_sampled.csv    # Weather data sample
│   ├── gdelt_events_clean_2024.csv      # GDELT events data
│
├── 📓 Jupyter Notebooks
│   ├── ais_data_processing.ipynb        # AIS data preprocessing
│   ├── weather_data.ipynb               # Weather data processing
│   ├── news_data.ipynb                  # GDELT events data processing
│   ├── data_integration_supervised.ipynb    # Data integration pipeline
│   ├── data_exploration_integrated.ipynb    # Exploratory data analysis
│   ├── model_training_supervised.ipynb      # Model training
│   └── model_tuning_supervised.ipynb        # Hyperparameter tuning
│
├── 🤖 Trained Models (via Git LFS)
│   ├── model_rf_classification_tuned.pkl    # Random Forest classifier
│   ├── model_rfr_regression_tuned.pkl       # Random Forest regressor
│   ├── scaler.pkl                           # Feature scaler
│   └── preprocessing_params.pkl             # Preprocessing parameters
│
├── 🚀 Deployment
│   ├── model_deployment.py              # Streamlit dashboard application
│   └── requirements.txt                 # Python dependencies
│
└── 📝 Documentation
    ├── README.md                        # This file
    ├── .gitignore                       # Git ignore rules
    └── .gitattributes                   # Git LFS tracking
```

---

## 🗄️ Data Sources

### 1. **AIS (Automatic Identification System) Data**

### 2. **ERA5 Weather Data**

### 3. **GDELT Events Data**

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Git LFS (for large files)
- Google Gemini API key (for AI insights)

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/voyage-delay-prediction.git
   cd voyage-delay-prediction
   ```

2. **Install Git LFS** (if not already installed)
   ```bash
   git lfs install
   git lfs pull
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   GEMINI_API_KEY=your_google_gemini_api_key_here
   ```

5. **Verify installation**
   ```bash
   python -c "import streamlit; import pandas; import sklearn; print('All dependencies installed!')"
   ```

---

## 🚀 Usage

### Running the Dashboard Locally

```bash
streamlit run model_deployment.py
```

The dashboard will open in your default browser at `http://localhost:8501`

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👤 Author

**[Ee Hann]**
- GitHub: [@walauceh](https://github.com/walauceh)

---

## 🙏 Acknowledgments

- **ERA5 Weather Data** - Copernicus Climate Change Service (C3S)
- **GDELT Project** - Global Database of Events, Language, and Tone
- **AIS Data Providers** - Maritime tracking data sources
- **Google Gemini AI** - AI-powered insights and recommendations
- **Streamlit Community** - Dashboard framework and deployment

---

## 📚 References

1. ECMWF ERA5 Reanalysis Documentation
2. GDELT Project Documentation
3. IMO Guidelines on AIS Data Standards
4. Maritime Delay Prediction Research Papers


⭐ Thanks for checking out my project!
