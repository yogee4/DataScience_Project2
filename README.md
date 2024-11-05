# Data Science Project: Time Series Forecasting

![](https://github.com/yogee4/DataScience_Project2/blob/main/logo.jpg)

## Introduction

This project focuses on time series forecasting to predict temperature trends using historical climate data. Accurate temperature forecasting is crucial for various sectors such as agriculture, energy, and disaster management. The goal is to utilize the Jena Climate dataset to train machine learning models that predict future temperatures based on past climate observations.

## Requirements

- Python Libraries: TensorFlow, Pandas, NumPy, Matplotlib (for data manipulation, visualization, and building deep learning models).

- Development Environment: Jupyter Notebook or any Python IDE for interactive development.

- Data: Jena Climate dataset containing detailed weather data from 2009 to 2016.

## About the Data

Source: Jena Climate dataset records climate variables every 10 minutes.

*Features:*

- Date Time: Timestamp for each record.
  
- T (degC): Temperature in degrees Celsius (target variable).

Other features include atmospheric pressure (p (mbar)), relative humidity (rh (%)), and various metrics related to vapor pressure and humidity (VPmax, VPact, VPdef, sh), among others.

- Dataset Size: Contains data points over several years, facilitating time series analysis and forecasting.
- Preprocessing: The data undergoes down-sampling to reduce frequency (from every 10 minutes to every hour) and is transformed to create feature windows for model training.
## Tools

- TensorFlow: Utilized to build and train deep learning models, specifically neural networks designed for sequential data.

- Pandas: Used for data manipulation and preprocessing.

- NumPy: Assists in numerical operations and reshaping data.

- Matplotlib: For data visualization, aiding in the exploratory data analysis phase.

## Conclusion

The project showcases the process of building a predictive model for temperature forecasting using time series analysis. By leveraging deep learning techniques, specifically with TensorFlow, the project highlights the capability of neural networks to capture temporal patterns in climate data. The results can serve as a foundation for further refinements and applications in real-world temperature prediction scenarios
