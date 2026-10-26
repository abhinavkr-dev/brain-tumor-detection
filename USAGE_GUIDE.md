# Usage Guide — Brain Tumor Detection Using CNN-LSTM

## 1. Project Overview

This project focuses on detecting brain tumors from MRI images using deep learning. It uses a CNN-based model with MobileNetV2 for image feature extraction and an LSTM-based architecture as implemented in the project notebook.

The project also includes a Streamlit web application that allows users to upload an MRI image and view the model's prediction.

## 2. Requirements

Before running the project, make sure you have:

- Python 3.11 recommended
- The required Python libraries listed in `requirements.txt`
- The trained model file `mobilenetv2_best.keras`
- The project source code and supporting files

## 3. Installation

### Step 1: Clone the Repository

Open a terminal and run:

```bash
git clone https://github.com/chirayu08/Brain-Tumor-Detection-from-MRI-using-CNN-LSTM.git
```

### Step 2: Navigate to the Project Folder

```bash
cd Brain-Tumor-Detection-from-MRI-using-CNN-LSTM
```

### Step 3: Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate the environment.

**Windows:**

```bash
venv\Scripts\activate
```

**Linux/macOS:**

```bash
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Model File Setup

Ensure that the trained model file named `mobilenetv2_best.keras` is available in one of the locations supported by the application:

- `outputs/mobilenetv2_best.keras`
- `mobilenetv2_best.keras` in the project root

The application must be able to find the model file before predictions can be performed.

**Note:** The trained model file may need to be downloaded or added separately if it is not included in the repository.

## 5. Running the Application

From the project directory, run:

```bash
streamlit run app.py
```

Streamlit will provide a local URL in the terminal, usually:

```text
http://localhost:8501
```

Open this URL in your browser to access the application.

## 6. How to Use the Application

1. Open the Streamlit application in your browser.
2. Upload an MRI brain image using the image-upload option.
3. Wait for the model to process the image.
4. Review the predicted class displayed by the application.
5. If available, inspect the Grad-CAM visualization to understand which image regions influenced the model's prediction.
6. If available, generate or download the prediction report.

## 7. Prediction Classes

The application uses the following class mapping:

| Class Index | Label |
|---|---|
| 0 | Brain Tumor |
| 1 | Healthy |

Always refer to the label displayed by the application and verify that the class mapping matches the trained model.

## 8. Technologies Used

- **Python** — Main programming language
- **TensorFlow / Keras** — Deep learning model development and inference
- **MobileNetV2** — Image feature extraction
- **CNN-LSTM** — Model architecture used in the project
- **Streamlit** — Interactive web application
- **NumPy** — Numerical operations
- **Pillow (PIL)** — Image processing and report generation
- **OpenCV** — Image processing and computer vision
- **Matplotlib and Seaborn** — Visualization
- **Scikit-learn** — Model evaluation and machine learning utilities

## 9. Troubleshooting

### Missing Model File

If the application cannot find the trained model, check that `mobilenetv2_best.keras` exists in one of the supported locations.

### Missing Python Packages

Install the dependencies again:

```bash
pip install -r requirements.txt
```

### Streamlit Command Not Found

Try running:

```bash
python -m streamlit run app.py
```

### TensorFlow Installation Issues

Use a compatible Python version and install the TensorFlow version specified in `requirements.txt`.

## 10. Important Disclaimer

This project is intended for educational and research purposes only. Its predictions are not a substitute for professional medical advice, diagnosis, or treatment.

MRI images and predictions should be reviewed by qualified healthcare professionals. The model may produce incorrect predictions and should not be used as the sole basis for medical decisions.
