

import streamlit as st
import os
import numpy as np
import mne
import xgboost as xgb
from scipy.signal import butter, sosfiltfilt
from io import BytesIO
import requests
import matplotlib.pyplot as plt
import tempfile

# Set page configuration
st.set_page_config(page_title="EDF File Classifier", layout="wide")

# Define utility functions for EDF processing and filtering
def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    """Bandpass filter for EEG data."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sosfiltfilt(sos, data)

@st.cache_data
def load_edf_data(file_path, channels):
    """Load EDF data and select specified channels."""
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    available_channels = [ch for ch in channels if ch in raw.ch_names]
    if not available_channels:
        raise ValueError("None of the selected channels are available in the EDF file.")
    raw.pick_channels(available_channels)
    data = raw.get_data()
    fs = raw.info['sfreq']
    return data, fs, raw

def process_edf(data, fs, model, channels, lowcut, highcut, epoch_s, epoch_e):
    """Process EEG data and return model predictions."""
    filtered_data = butter_bandpass_filter(data, lowcut, highcut, fs)
    
    epochs = mne.make_fixed_length_epochs(
        mne.io.RawArray(filtered_data, mne.create_info(channels, fs, ch_types='eeg')),
        duration=(epoch_e - epoch_s) / 1000, preload=True
    ).get_data()
    
    # Flatten the data for XGBoost
    X_flat = epochs.reshape(epochs.shape[0], -1)
    
    # Make predictions using the loaded model
    dmatrix = xgb.DMatrix(X_flat)
    predictions = model.predict(dmatrix)
    
    # Take the mean prediction over all epochs and classify
    avg_prediction = np.mean(predictions)
    if avg_prediction > 0.5:
        return "Abnormal", filtered_data
    else:
        return "Normal", filtered_data

@st.cache_resource
def load_model(model_file_path):
    """Load the pre-trained XGBoost model."""
    model = xgb.Booster()
    model.load_model(model_file_path)
    return model

def get_file_from_google_drive(google_drive_link):
    """Download EDF file from Google Drive link and save to a temporary file."""
    try:
        file_id = google_drive_link.split('/d/')[1].split('/')[0]
    except IndexError:
        st.error("Invalid Google Drive link format.")
        return None
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(download_url)
    if response.status_code == 200:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.edf')
        temp_file.write(response.content)
        temp_file.close()
        return temp_file.name
    else:
        st.error("Failed to download the file from Google Drive.")
        return None

# Main application
def main():
    st.title("🧠 EDF File Classifier")

    # Sidebar for user inputs
    st.sidebar.header("Select Model")
    model_options = {
        "Default XGBoost Model": "./xgboost_model.bin",
        "Custom Model Path": None,  # Placeholder for custom model path
    }
    selected_model_name = st.sidebar.selectbox("Choose a model", list(model_options.keys()))
    custom_model_path = ""
    if selected_model_name == "Custom Model Path":
        custom_model_path = st.sidebar.text_input("Enter custom model path", value="./xgboost_model.bin")

    if st.sidebar.button("Load Model"):
        if selected_model_name == "Custom Model Path":
            model_file_path = custom_model_path
        else:
            model_file_path = model_options[selected_model_name]

        try:
            model = load_model(model_file_path)
            st.session_state['model'] = model
            st.session_state['model_file_path'] = model_file_path
            st.sidebar.success(f"Model loaded successfully from `{model_file_path}`.")
        except Exception as e:
            st.sidebar.error(f"Error loading model from `{model_file_path}`: {e}")
            st.session_state['model'] = None

    if 'model' in st.session_state and st.session_state['model'] is not None:
        model = st.session_state['model']
        # Proceed with data input and processing
        st.header("Data Processing")
        
        st.subheader("Upload EDF File or Provide Link")
        uploaded_file = st.file_uploader("Choose an EDF file", type=["edf"])
        google_drive_link = st.text_input("Or enter Google Drive link of your EDF file")

        # Move Configuration to Sidebar
        st.sidebar.header("Configuration")
        channels = st.sidebar.multiselect(
            "Select EEG Channels",
            options=['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                     'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'PZ', 'CZ', 'A1', 'A2'],
            default=['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4']
        )
        lowcut = st.sidebar.slider("Lowcut Frequency", 0.1, 10.0, 1.0)
        highcut = st.sidebar.slider("Highcut Frequency", 30.0, 100.0, 40.0)
        epoch_s = st.sidebar.number_input("Epoch Start (ms)", value=0)
        epoch_e = st.sidebar.number_input("Epoch End (ms)", value=1300)
        
        if (uploaded_file or google_drive_link):
            if st.button("Enter"):
                # Proceed with data loading and processing
                if uploaded_file:
                    # Save uploaded file to a temporary file
                    temp_edf_file = tempfile.NamedTemporaryFile(delete=False, suffix='.edf')
                    temp_edf_file.write(uploaded_file.read())
                    temp_edf_file.close()
                    edf_file_path = temp_edf_file.name
                    st.write(f"Processing uploaded EDF file: `{uploaded_file.name}`")
                else:
                    edf_file_path = get_file_from_google_drive(google_drive_link)
                    if edf_file_path is None:
                        return
                    st.write("Processing EDF file from Google Drive link")
                
                # Load EDF data
                data_load_state = st.text("Loading data...")
                try:
                    data, fs, raw = load_edf_data(edf_file_path, channels)
                    data_load_state.text("Data loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading EDF data: {e}")
                    return

                # Visualize Raw EEG Signals
                st.subheader("EEG Signal Visualization")
                fig, ax = plt.subplots(figsize=(10, 4))
                times = np.linspace(0, len(data[0]) / fs, num=len(data[0]))
                for i in range(len(data)):
                    ax.plot(times, data[i] + i * 50, label=channels[i])  # Offset for visibility
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Amplitude (µV)')
                ax.set_title('Raw EEG Signals')
                ax.legend(loc='upper right')
                st.pyplot(fig)

                # Run Prediction and Get Filtered Data
                with st.spinner("Running Prediction..."):
                    try:
                        result, filtered_data = process_edf(data, fs, model, channels, lowcut, highcut, epoch_s, epoch_e)
                        st.subheader("Prediction Result")
                        st.write(f"**Prediction:** {result}")
                        if result == "Abnormal":
                            st.error("The EEG recording indicates an **Abnormal** condition.")
                        else:
                            st.success("The EEG recording indicates a **Normal** condition.")
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")
                        return

                # Visualize Filtered EEG Signals
                st.subheader("Filtered EEG Signal Visualization")
                fig_filtered, ax_filtered = plt.subplots(figsize=(10, 4))
                for i in range(len(filtered_data)):
                    ax_filtered.plot(times, filtered_data[i] + i * 50, label=channels[i])  # Offset for visibility
                ax_filtered.set_xlabel('Time (s)')
                ax_filtered.set_ylabel('Amplitude (µV)')
                ax_filtered.set_title('Filtered EEG Signals')
                ax_filtered.legend(loc='upper right')
                st.pyplot(fig_filtered)

                # Clean up temporary files
                if uploaded_file:
                    os.unlink(edf_file_path)
                elif google_drive_link:
                    os.unlink(edf_file_path)
            else:
                st.info("Click 'Enter' to load and process the EDF file.")
        else:
            st.warning("Please upload an EDF file or provide a Google Drive link to proceed.")
    else:
        st.warning("Please load a model to proceed.")

if __name__ == "__main__":
    main()
