# Nvidia Stock Price Prediction (LSTM)
This project implements a deep learning model using PyTorch to forecast the daily closing prices of Nvidia (NVDA) stock. By leveraging a Long Short-Term Memory (LSTM) neural network, the model captures temporal dependencies in historical market data to predict future price movements. 

## Overview
Time-series forecasting in financial markets requires models capable of understanding long-term sequential context. This repository contains a Jupyter Notebook (`stock_price_prediction.ipynb`) that fetches historical NVDA data, preprocesses it for deep learning, and trains an LSTM network to predict the next day's closing price based on a 30-day lookback window.

## Technologies Used
- **Deep Learning Framework:** PyTorch (`torch`, `torch.nn`)
- **Data Manipulation:** Pandas, NumPy
- **Data Ingestion:** `yfinance` API
- **Preprocessing & Metrics:** scikit-learn (`StandardScaler`, `root_mean_squared_error`)
- **Visualization:** Matplotlib

## Model Architecture
The core forecasting engine is a custom PyTorch module:
- **Input Layer:** Accepts sequences of scaled daily closing prices (sequence length = 30).
- **LSTM Layers:** A 2-layer LSTM network with a hidden dimension of 32 to process the time-series data and extract temporal features.
- **Output Layer:** A fully connected linear layer that maps the final hidden state to a single continuous output (the predicted price).

## Project Workflow
1. **Data Acquisition:** Historical daily stock data (from Jan 1, 2020, onwards) is pulled directly via the Yahoo Finance API.
2. **Preprocessing:** Data is isolated to 'Close' prices and normalized using `StandardScaler` to ensure stable gradient descent. The data is then transformed into sliding window sequences.
3. **Training:** The model is trained on a GPU for 200 epochs using the Adam optimizer (learning rate: 0.01) and Mean Squared Error (MSE) loss.
4. **Inference:** Predictions are inverse-transformed back to the original price scale for interpretable evaluation.

## Results & Evaluation
The model's performance is quantified using Root Mean Squared Error (RMSE). 
- **Training RMSE:** ~2.00
- **Testing RMSE:** ~8.94

*Note: Results may vary slightly depending on the exact date range fetched at runtime.*

## Future Enhancements
To scale this project for production and expand its capabilities, the following features are planned:
- **Cloud Deployment:** Containerize the inference logic using Docker and expose it as a FastAPI service deployed on Google Cloud Platform (e.g., Cloud Run or Artifact Registry).
- **Multi-Agent Integration:** Package this predictive model as a specialized forecasting agent within a broader autonomous multi-agent financial analysis system.
- **Multivariate Inputs:** Incorporate additional financial indicators such as trading volume, moving averages, or macroeconomic indices.
- **Automated Workflows:** Implement CI/CD pipelines to automatically retrain the model on fresh daily data.

## Getting Started

### Prerequisites
Ensure you have Python 3.8+ installed along with the required libraries. 

```bash
pip install torch torchvision torchaudio numpy pandas yfinance scikit-learn matplotlib
