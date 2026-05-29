# COS30019 Assignment 2B 

## Prerequisites
- Python 3.10+ (64-bit recommended)

## Python Packages
Install dependencies from `requirements.txt`:
- Flask (web UI)
- NumPy / Pandas (data handling)
- Matplotlib / NetworkX (graph visualization)
- scikit-learn / joblib (Random Forest + model persistence)
- TensorFlow (LSTM / GRU)
- xlrd (required to read the provided `.xls` dataset)
- colorama (colored terminal output on Windows; safe to install even if not used everywhere)

## Setup
1. Open a terminal in the project root folder (the folder that contains `app.py`).
2. Install dependencies:
```bash
   pip install -r requirements.txt
```

## Run the Web App
Start the Flask app:
```bash
python app.py
```
Then open the URL shown in the terminal (usually `http://127.0.0.1:5000/`).

## Notes
- As the Random Forest model takes up too much space, it has not been uploaded to GitHub and is only stored on the computer of the team member who trained the model.
