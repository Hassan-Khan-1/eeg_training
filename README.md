# XGBoost for The Classification of EEG Scalp Images

This repository contains the code for an EEG signal classification project using the **XGBoost** machine learning algorithm. The project focuses on classifying EEG data into normal and abnormal states, which can have significant implications for neurological diagnostics.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Data](#data)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Deployment](#deployment)

## Project Overview
This project aims to build a machine learning model that analyses EEG signals stored in **EDF (European Data Format)** files and classifies whether a patient is in a normal or abnormal state. The key steps include data pre-processing, feature extraction, model training using XGBoost, and deploying the model via **Streamlit**.

The ultimate goal of this project is to provide a proof-of-concept for real-time neurological diagnostics using machine learning algorithms on EEG data.

## Features
- EEG signal classification (normal vs abnormal)
- Feature extraction using the `mne` library
- XGBoost model training and evaluation
- Web-based model deployment with **Streamlit**
- Real-time classification of EEG data via user-uploaded EDF files

## Data
The project uses EEG data in **EDF format**. The data is pre-processed to filter out noise, segment it, and extract relevant features for classification. 

### Data Source:
The EEG dataset is not included in this repository. Here is a link to the dataset: https://github.com/dll-ncai/eeg_pre-diagnostic_screening

## Requirements
To run the project, you'll need the following dependencies:

- Python 3.8+
- mne
- XGBoost
- NumPy
- pandas
- Streamlit
- scikit-learn

### Additional Tools:
- Google Colab (for rapid prototyping and computation)
- GitHub (for version control and collaboration)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/eeg-classification-xgboost.git
   ```
2. Navigate into the project directory:
   ```bash
   cd eeg-classification-xgboost
   ```
3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Pre-process the EEG data:
   - Use the `mne` library to load the EDF files and extract features.
   - Filter and segment the EEG signals as required.

2. Train the XGBoost model:
   ```bash
   python train_model.py
   ```

3. Run the Streamlit app to deploy the model:
   ```bash
   streamlit run app.py
   ```

4. Upload an EDF file through the web app to classify the EEG signals as normal or abnormal.

## Model Performance

The model was evaluated using accuracy, precision, recall, and F1-score. These metrics helped assess its performance on both training and testing data.

- **Training Accuracy:** 79.84%
- **Testing Accuracy:** 69.69%
- **Precision (Testing):** 77.12%
- **Recall (Testing):** 68.89%
- **Training Time:** 41.5 seconds (using GPU-enabled XGBoost)

![image](https://github.com/user-attachments/assets/60d6e7ac-6043-4cce-8d73-b4bb3ddedca6)

![image](https://github.com/user-attachments/assets/8a87d3cd-13c8-4093-9834-23f5b2a4dead)

![image](https://github.com/user-attachments/assets/bfdfa37b-c9b2-4072-a87c-25531880297f)

![image](https://github.com/user-attachments/assets/64e442cc-5877-4dd8-b579-9cd1b4ec54d4)

![image](https://github.com/user-attachments/assets/f04b6007-d9f0-413b-91ef-230a17cd9059)

## Deployment
The model was deployed using **Streamlit** and hosted on **Google Colab**. Users can interact with the model by uploading their EEG files and receiving classification results in real time.

