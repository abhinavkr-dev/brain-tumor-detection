# NeuroScan AI — Brain MRI Image Analysis

NeuroScan AI is an educational research prototype that classifies an uploaded brain MRI image using a MobileNetV2 transfer-learning model. It also provides a Grad-CAM visualization to help inspect regions that influenced the model output, a downloadable PDF summary, and a side-by-side workflow for analyzing two images independently.

> **Important:** This project is for educational/research use only. It is not a medical device and must not be used to diagnose, rule out, or treat a medical condition. Model scores are not necessarily calibrated medical probabilities. Consult a qualified medical professional for medical interpretation.

## Features

- Upload a JPG, JPEG, or PNG image for model inference.
- Display the predicted class and both class scores.
- Generate a Grad-CAM heatmap/overlay when supported by the model.
- Download a PDF report for an analyzed image.
- Analyze two images independently in the Compare MRIs tab.
- Validate image files and limit upload size.

## Model and class mapping

The application expects a trained Keras model named `mobilenetv2_best.keras`. The model output class mapping used by the app is:

- `0` — Brain Tumor
- `1` — Healthy / No Tumor

The input pipeline converts the image to grayscale, resizes it to 150 × 150, repeats the grayscale channel three times, and applies MobileNetV2 `preprocess_input`. Use the same preprocessing and class mapping as the notebook that trained the model.

## Project structure

```text
brain-tumor-detection/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── Brain_Tumor_Detection_CNN_LSTM.ipynb
└── outputs/
    └── mobilenetv2_best.keras
```

The notebook and trained model should be included only if you have them locally and have permission to share the dataset/model artifacts. Do not upload the full MRI dataset or your Python virtual environment to GitHub.

## Run locally

### 1. Create and activate a virtual environment (recommended)

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Add the trained model

Place `mobilenetv2_best.keras` in the `outputs` directory next to `app.py`:

```text
outputs/mobilenetv2_best.keras
```

The app also checks for the model file next to `app.py` as a fallback. The trained model file is not included in this starter bundle.

### 4. Start the application

```bash
streamlit run app.py
```

## Notes before publishing

- GitHub blocks individual files larger than 100 MB. Check the model file size before committing it. If it is too large, use Git LFS or publish instructions for obtaining the model instead.
- Confirm you have permission to redistribute any dataset, pretrained weights, and model artifacts.
- Do not include private data, credentials, virtual environments, caches, or generated patient reports in the repository.
- Validate the app and model locally before presenting or deploying it.

## Limitations

Performance depends on the training dataset, split, preprocessing, and model quality. Grad-CAM is an explanatory visualization, not a tumor segmentation mask or proof that a tumor exists. This prototype has not been established as clinically validated.
