# 🍲 Vietnamese Food Classification (Pho, Com Tam, Bun)

This project is a deep learning image classification system built to recognize three traditional Vietnamese dishes:
- **Phở**
- **Cơm Tấm**
- **Bún**

The trained model is deployed as a **Flask web app**, allowing users to upload an image and receive a prediction of the dish name along with a confidence score.

This project uses the [Vietnamese Foods](https://www.kaggle.com/datasets/quandang/vietnamese-foods) dataset from Kaggle.
> **NOTE**: This project only used data for `pho`, `com_tam`, and `bun`. The `bun` class was created by merging three separate categories (`bun bo hue`, `bun cha`, `bun rieu`).

---

## 📁 Project Structure

```text
food_detect/
├── README.md
│
├── dataset_3_mon/
│   ├── bun/
│   ├── com_tam/
│   └── pho/
│
├── model/
│   └── food_classifier_model.keras
│
├── static/
│   ├── style.css
│   └── script.js
│
├── templates/
│   └── index.html
│
├── app.py             # Flask Server
├── train.py           # Model Training Script
└── requirements.txt   # Required libraries
```

---

## 🧠 Model Information
* **Architecture**: MobileNetV2 (using Transfer Learning)
* **Training Dataset**: Custom 3-class dataset (Pho, Com Tam, Bun)
* **Input Size**: 224 x 224 pixels
* **Framework**: TensorFlow / Keras
* **Output**: 3-class Softmax

The trained model file is stored at:
```bash
model/food_classifier_model.keras
```

---

## 🧾 File Descriptions
| File | Description |
| :--- | :--- |
| `train.py` | Python script to load data, train the model, and save it. |
| `app.py` | The main Flask server script that handles image uploads and prediction API. |
| `templates/index.html` | The web interface (frontend) for user image uploads. |
| `static/style.css` | CSS for styling the web interface. |
| `static/script.js` | JavaScript to handle form submission and fetch API results. |
| `model/` | Directory containing the trained `.keras` model file. |
| `dataset_3_mon/` | Directory containing the processed image data (3 classes). |
| `requirements.txt` | A list of all Python libraries needed to run the project. |

---

## 🚀 Setup and Running the Project

This project was developed and tested on a **macOS (Apple M1)** environment.

### 1. Clone the Repository

```bash
git clone https://github.com/Hoang271205/food_detector
cd food_detect
```

### 2. Create and Activate Virtual Environment

```bash
# Create a virtual environment (e.g., named 'ven')
python3 -m venv ven

# Activate the virtual environment
source ven/bin/activate
```

### 3. Install Dependencies

Use the `requirements.txt` file to install all necessary libraries.

```bash
pip install -r requirements.txt
```

### 4. (Very Important) Fix SSL Error on macOS

If you encounter an `[SSL: CERTIFICATE_VERIFY_FAILED]` error when running `train.py` (due to Python not finding security certificates), you must run the following command (only once):

1.  Open **Finder** -> **Applications** -> **Python 3.12** (or your Python version).
2.  Double-click the **`Install Certificates.command`** file.

---

## 🏃 How to Use

### 1. Run the Web App (Recommended)

The model is already trained and saved. You just need to run the Flask server:

```bash
python app.py
```

Once the server is running, open your browser and go to:
**[http://127.0.0.1:5000](http://127.0.0.1:5000)**

### 2. (Optional) Re-train the Model

If you want to train the model yourself using the data in `dataset_3_mon`, run:

```bash
python train.py
```

This process will take a few minutes and will overwrite the existing `model/food_classifier_model.keras` file.