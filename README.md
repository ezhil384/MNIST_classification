# MNIST Digit Classification

This project implements a machine learning pipeline to classify handwritten digits from the MNIST dataset using Python. It includes:
- A baseline logistic regression model
- A final tuned convolutional neural network (CNN)
- Visualization of training performance
- Inference pipeline for test-time predictions

---

![Dataset](sample_digits.png)

---

## 📁 Project Structure
```
.
├── training.csv              # Labeled MNIST training data
├── sample_test.csv          # Unlabeled MNIST test data
├── final_cnn_model.h5       # Saved final CNN model weights
├── final_script.py          # Main script: training + inference
├── MNIST-Notebook.ipynb     # Full Jupyter notebook with visualizations
├── submission.csv           # Output file of predictions for submission
├── README.md                # This file
```

---

## 🚀 Requirements
Install dependencies with:
```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```
You may also use a Conda environment:
```bash
conda create -n mnist-env python=3.9
conda activate mnist-env
pip install tensorflow numpy pandas matplotlib scikit-learn
```

---

## 🔧 How to Run the Project
### 1. Run Training + Save Model
```bash
python final_script.py
```
- Trains the CNN on `training.csv`
- Saves the model to `final_cnn_model.h5`
- Outputs predictions on `sample_test.csv` to `submission.csv`

### 2. Evaluate on Your Own Test File
Replace the file name inside `final_script.py`:
```python
sample_test = pd.read_csv("your_test_file.csv")
```
Make sure your test file:
- Has the same format as `sample_test.csv`
- Contains no label column (only pixel values)

---

## 📈 Performance
| Model                | Validation Accuracy |
|---------------------|---------------------|
| Logistic Regression | 92.5%               |
| Tuned CNN           | 99.2%               |

---

## 📊 Visualizations
Plots included in the notebook:
- Training vs Validation Accuracy over Epochs
- Digit image previews
- Architecture breakdowns

---

## 📬 Output
Final predictions are saved to:
```
submission.csv
```
Format:
```
Id,Label
1,7
2,2
...
```

---

## 🧠 Credits
Developed by Abhiraj Ezhil as part of Biostat 626 Midterm Project at University of Michigan, Winter 2025.
