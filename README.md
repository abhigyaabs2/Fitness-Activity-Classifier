# 🏃 Fitness Activity Classifier

A machine learning project that classifies physical activities (Walking, Jogging, Sitting) using accelerometer data. Built with scikit-learn and deployed with Streamlit for real-time predictions.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Technologies Used](#technologies-used)
- [How It Works](#how-it-works)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 🎯 Overview

This project demonstrates the application of machine learning for human activity recognition using accelerometer data. The system can classify three distinct physical activities with high accuracy, making it suitable for fitness tracking applications, health monitoring systems, and wearable device applications.

**Key Highlights:**
- 🎯 **99%+ accuracy** on test data
- 📊 Real-time activity classification
- 🚀 Interactive web application
- 📈 Comprehensive data analysis and visualization
- 🔄 Multiple input modes (Manual, CSV, Simulation)

---

## ✨ Features

### Machine Learning
- Random Forest classifier with optimized hyperparameters
- Feature engineering including magnitude calculations and rolling statistics
- Robust data preprocessing and scaling
- Comprehensive model evaluation metrics

### Web Application
- **Manual Input Mode**: Test individual accelerometer readings
- **CSV Upload Mode**: Batch classification of recorded data
- **Real-time Simulation**: Generate and classify synthetic activity data
- Interactive visualizations with Plotly
- Confidence scores and probability distributions
- Downloadable classification results

### Analysis & Visualization
- Exploratory Data Analysis (EDA)
- Feature importance analysis
- Confusion matrix visualization
- Time-series activity tracking
- 3D scatter plots for multi-dimensional data

---

### Classification Results
```
Activity: WALKING
Confidence: 98.7%

Accelerometer Data:
- X: 0.5 m/s²
- Y: 0.3 m/s²
- Z: 9.8 m/s²
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/abhigyaabs2/Fitness-Activity-Classifier.git
cd Fitness-Activity-Classifier
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

---

## 🚀 Usage

### Train the Model

1. Open Jupyter Notebook:
```bash
jupyter notebook notebooks/fitness_activity_classifier.ipynb
```

2. Run all cells to:
   - Generate synthetic accelerometer data
   - Perform exploratory data analysis
   - Train the Random Forest model
   - Save trained model and scaler

### Run the Web Application

```bash
streamlit run fitness.py
```

The app will open in your browser at `http://localhost:8501`

### Using the Application

#### Option 1: Manual Input
1. Select "Manual Input" from the sidebar
2. Enter X, Y, Z acceleration values
3. Click "Classify Activity"
4. View prediction and confidence scores

#### Option 2: CSV Upload
1. Select "Upload CSV" from the sidebar
2. Upload a CSV file with columns: `accel_x`, `accel_y`, `accel_z`
3. Click "Classify Activities"
4. View results and download classified data

#### Option 3: Real-time Simulation
1. Select "Real-time Simulation" from the sidebar
2. Choose an activity and number of samples
3. Click "Start Simulation"
4. View simulated data and classification accuracy

---

## 📁 Project Structure

```
fitness-activity-classifier/
│
├── notebooks/
│   └── fitness_activity_classifier.ipynb    # Model training notebook
│
│   ├── fitness_classifier_model.pkl         # Trained model
│   └── scaler.pkl                            # Feature scaler
│
│   └── fitness_data.csv                      # Generated training data
│
├── fitness.py                                     # Streamlit web application
├── README.md                                  # Project documentation
└── LICENSE                                    # License file
```

---

## 📊 Model Performance

### Metrics
| Metric | Score |
|--------|-------|
| **Accuracy** | 99.2% |
| **Precision** | 99.1% |
| **Recall** | 99.2% |
| **F1-Score** | 99.1% |

### Confusion Matrix
```
              Predicted
              Jog  Sit  Walk
Actual Jog    298   1    1
       Sit     0   300   0
       Walk    2    0   298
```

### Feature Importance
Top 5 features contributing to classification:
1. X-axis Rolling Standard Deviation (18.2%)
2. Magnitude (16.7%)
3. Y-axis Rolling Standard Deviation (14.5%)
4. XY Magnitude (13.1%)
5. Z-axis Rolling Mean (11.8%)

---

## 🛠️ Technologies Used

### Machine Learning & Data Science
- **scikit-learn**: Model training and evaluation
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation and analysis

### Visualization
- **Matplotlib**: Static visualizations
- **Seaborn**: Statistical plotting
- **Plotly**: Interactive web visualizations

### Web Application
- **Streamlit**: Web app framework
- **Joblib**: Model serialization

### Development
- **Jupyter Notebook**: Interactive development
- **Python 3.8+**: Core programming language

---

## 🧠 How It Works

### 1. Data Collection
The system uses 3-axis accelerometer data:
- **X-axis**: Horizontal movement (side-to-side)
- **Y-axis**: Forward/backward movement
- **Z-axis**: Vertical movement (includes gravity ~9.8 m/s²)

### 2. Feature Engineering
Raw accelerometer data is enhanced with derived features:
- **Magnitude**: Overall acceleration intensity
- **XY Magnitude**: Horizontal plane movement
- **Rolling Statistics**: Mean and standard deviation over time windows

### 3. Classification
A Random Forest classifier processes the features:
- 100 decision trees for robust predictions
- Ensemble voting for final classification
- Probability estimates for confidence scores

### 4. Activity Patterns
Different activities have distinct signatures:
- **Walking**: Moderate periodic oscillations
- **Jogging**: High-amplitude rapid oscillations
- **Sitting**: Minimal movement, stable gravity component

---

## 🔮 Future Enhancements

- [ ] Add more activities (running, cycling, climbing stairs)
- [ ] Implement deep learning models (LSTM, CNN)
- [ ] Real-time data collection from smartphone sensors
- [ ] Mobile app deployment (iOS/Android)
- [ ] Calorie expenditure estimation
- [ ] User profile and activity history tracking
- [ ] Multi-user support with authentication
- [ ] Integration with fitness wearables (Fitbit, Apple Watch)
- [ ] Cloud deployment (AWS, GCP, Azure)
- [ ] API endpoint for third-party integrations

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please ensure your PR includes:
- Clear description of changes
- Updated documentation
- Test cases (if applicable)
- Screenshots (for UI changes)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Dataset inspiration from UCI Machine Learning Repository
- Streamlit community for amazing documentation
- scikit-learn contributors for excellent ML tools
---

<div align="center">

**If you found this project helpful, please consider giving it a ⭐!**

Made with ❤️ and ☕

</div>
