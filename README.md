# Power System Planning - Energy Deficit Prediction

A comprehensive machine learning project for predicting energy deficits in power distribution systems using time-series forecasting models. This project analyzes 33KV feeder data from power distribution networks to predict total energy deficits, helping power system planners optimize energy distribution and identify potential shortfalls.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Project Overview

This project focuses on analyzing power consumption patterns and availability data from 33KV feeders across different districts (APAPA and IJORA) to:

- Predict energy deficits in power distribution systems
- Analyze consumption patterns and availability trends
- Identify feeders with zero availability and recovery times
- Compare performance of various machine learning models
- Provide insights for power system planning and optimization

## ✨ Features

- **Comprehensive Data Analysis**: Exploratory data analysis of power consumption and availability patterns
- **Feature Engineering**: Creation of energy deficit metrics and time-series features
- **Multiple ML Models**: Implementation of LSTM, BiLSTM, GRU, CNN, and Linear Regression models
- **Hyperparameter Optimization**: Automated hyperparameter tuning using Keras Tuner
- **Model Comparison**: Performance comparison across different model architectures
- **Visualization**: Rich visualizations for data trends and model performance
- **Recovery Analysis**: Analysis of power recovery times after outages

## 📊 Dataset

The project uses power system data from 2019-2021 including:

- **Availability Data**: Daily availability hours for 33KV feeders
- **Consumption Data**: Daily power consumption in MWh
- **Geographic Coverage**: Multiple districts and circles
- **Temporal Coverage**: 3 years of daily measurements

### Data Structure
```
data/
├── 33KV Daily Availability 2019.xlsx
├── 33KV Daily Availability 2020.xlsx
├── 33KV Daily Availability 2021.xlsx
├── 33KV Daily Consumption 2019.xlsx
├── 33KV Daily Consumption 2020.xlsx
└── 33KV Daily Consumption 2021.xlsx
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd "Power System Planning"
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Required Packages
```txt
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.1.0
tensorflow>=2.10.0
keras-tuner>=1.1.0
xgboost>=1.6.0
openpyxl>=3.0.0
```

## 📁 Project Structure

```
Power System Planning/
├── data/                           # Raw data files
│   ├── 33KV Daily Availability *.xlsx
│   └── 33KV Daily Consumption *.xlsx
├── src/                            # Source code
│   ├── data_clean_and_analysis.ipynb
│   └── model_train.ipynb
├── cleaned_data/                   # Processed data
│   ├── merged_data.csv
│   ├── energy_deficit.csv
│   └── data_stat.csv
├── tuning_params/                  # Hyperparameter tuning results
│   ├── lstm_tuning/
│   ├── gru_tuning/
│   ├── bilstm_tuning/
│   └── cnn_tuning/
├── model_metrics/                  # Model performance metrics
│   ├── model_metrics_comparison.csv
│   └── best_model_metrics.csv
|   |__GRU_model.keras              # Saved best model
├── requirements.txt
├── README.md
└── Project Methodology.docx
```

## 📖 Usage

### 1. Data Processing and Analysis

Run the data cleaning and analysis notebook:

```bash
jupyter notebook src/data_clean_and_analysis.ipynb
```

This notebook performs:
- Data loading and merging from multiple Excel files
- Exploratory data analysis
- Feature engineering (energy deficit calculation)
- Visualization of consumption and availability patterns

### 2. Model Training and Evaluation

Run the model training notebook:

```bash
jupyter notebook src/model_train.ipynb
```

This notebook includes:
- Data preprocessing for time-series modeling
- Implementation of multiple ML models
- Hyperparameter optimization
- Model evaluation and comparison

### 3. Key Functions

**Data Loading Function:**
```python
def load_excel_and_clean(path, value_name):
    # Loads and cleans Excel data with proper date formatting
    # Returns cleaned DataFrame with date column
```

**Sequence Creation for Time-Series:**
```python
def create_sequences(X, y, time_steps):
    # Creates sequences for LSTM/GRU models
    # Returns X_seq, y_seq for time-series prediction
```

**Energy Deficit Calculation:**
```python
def energy_deficit_df(df):
    # Calculates energy deficit as max_capacity - current_consumption
    # Returns DataFrame with deficit values
```

## 🤖 Models Implemented

### 1. Linear Regression (Baseline)
- Simple baseline model for comparison
- Direct feature-to-target mapping

### 2. Long Short-Term Memory (LSTM)
- Handles sequential dependencies in time-series data
- Architecture: 64→32 units with dropout and regularization

### 3. Bidirectional LSTM (BiLSTM)
- Processes sequences in both forward and backward directions
- Enhanced pattern recognition capabilities

### 4. Gated Recurrent Unit (GRU)
- Simplified RNN architecture with fewer parameters
- Faster training compared to LSTM

### 5. 1D Convolutional Neural Network (CNN)
- Captures local patterns in time-series data
- Efficient for shorter sequence dependencies

### Hyperparameter Optimization

All models use Keras Tuner's Hyperband algorithm for efficient hyperparameter search:

- **Units**: 32-128 (step: 32)
- **Activation**: ['relu', 'tanh', 'sigmoid']
- **Dropout Rate**: 0.2-0.5 (step: 0.1)
- **Learning Rate**: 1e-5 to 1e-3 (log scale)
- **Batch Size**: [16, 32, 64]

## 📈 Results

### Model Performance Comparison

| Model | MAE | MSE | RMSE | MAPE | R² Score |
|-------|-----|-----|------|------|----------|
| **GRU** | 125.59 | 30,878.69 | 175.72 | 0.066 | **0.495** |
| **CNN** | 129.10 | 31,298.16 | 176.91 | 0.067 | 0.488 |
| **BiLSTM** | 126.85 | 31,543.66 | 177.61 | 0.067 | 0.484 |
| **LSTM** | 132.69 | 31,812.61 | 178.36 | 0.071 | 0.480 |

### Best Performing Models (After Hyperparameter Tuning)

| Model | MAE | MSE | RMSE | MAPE | R² Score |
|-------|-----|-----|------|------|----------|
| **GRU** | 128.78 | 32,725.14 | 180.90 | 0.066 | **0.465** |
| **CNN** | 129.10 | 31,298.16 | 176.91 | 0.067 | 0.488 |
| **BiLSTM** | 128.78 | 32,725.14 | 180.90 | 0.066 | 0.465 |
| **LSTM** | 137.36 | 35,465.06 | 188.32 | 0.071 | 0.420 |

### Key Insights

1. **GRU** achieved the best overall performance with the highest R² score (0.495)
2. **CNN** provided competitive results with faster training time
3. **BiLSTM** showed good performance but with higher computational cost
4. Hyperparameter tuning improved model stability but didn't always improve raw performance
5. Energy deficit patterns show strong seasonality and district-specific characteristics

### Visualization Examples

The project generates comprehensive visualizations including:
- Time-series plots of consumption and availability
- Energy deficit trends by feeder
- Model performance comparisons
- Training/validation loss curves
- Actual vs. predicted comparisons

## 🛠️ Technical Details

### Data Preprocessing
- **Normalization**: MinMaxScaler for feature scaling
- **Sequence Creation**: 3-timestep windows for time-series models
- **Missing Data**: Handled through dropna() and interpolation
- **Feature Engineering**: Energy deficit calculation and temporal features

### Model Architecture Details

**LSTM/BiLSTM/GRU:**
```python
model = Sequential([
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.2),
    Dense(1, activation='linear')
])
```

**CNN:**
```python
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    Flatten(),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1)
])
```

### Training Configuration
- **Optimizer**: Adam with gradient clipping
- **Loss Function**: Mean Squared Error (MSE)
- **Metrics**: MAE, RMSE, MAPE, R²
- **Validation Split**: 10%
- **Early Stopping**: Implemented for hyperparameter tuning

## 🔧 Advanced Features

### Recovery Time Analysis
- Calculates average recovery time after zero availability
- District-wise and feeder-wise recovery analysis
- Identifies patterns in power restoration

### Load Shedding Detection
- Analyzes feeders with zero availability but positive consumption
- Helps identify load management strategies

### District Comparison
- Focused analysis on APAPA and IJORA districts
- Comparative performance metrics across districts

## 📝 Future Enhancements

1. **Real-time Prediction**: Implement streaming data processing
2. **Weather Integration**: Include weather data for better predictions
3. **Ensemble Methods**: Combine multiple models for improved accuracy
4. **Deployment**: Create API endpoints for model serving
5. **Advanced Visualizations**: Interactive dashboards for stakeholders
6. **Seasonal Decomposition**: Enhanced time-series analysis
7. **Anomaly Detection**: Identify unusual consumption patterns

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation for API changes

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Power distribution data providers
- Open-source machine learning community
- TensorFlow and Keras development teams
- Scikit-learn contributors

---

**Note**: This project is for educational and research purposes. Ensure proper validation before using predictions for actual power system planning decisions.
