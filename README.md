
# Obesity Level Prediction using Machine Learning

This project leverages machine learning techniques to predict an individual's obesity level based on lifestyle and physiological indicators such as dietary habits, physical activity, and health metrics. It aims to assist in early detection of obesity risk, which can be used to recommend targeted health interventions.

## 📌 Project Overview

- **Goal**: Classify individuals into obesity level categories using supervised machine learning.
- **Dataset**: Public dataset on obesity indicators (e.g., caloric intake, physical activity, age, BMI).
- **Output**: Multiclass classification (e.g., Underweight, Normal Weight, Overweight, Obese I/II/III).

## 🧠 Machine Learning Pipeline

1. **Data Preprocessing**
   - Data cleaning and handling missing values
   - Feature encoding and normalization

2. **Model Training**
   - Algorithms tested: Random Forest, Logistic Regression, SVM, KNN
   - Model evaluation with metrics like accuracy, precision, recall, F1-score

3. **Model Selection**
   - Comparison of models and hyperparameter tuning
   - Selection based on validation performance

4. **Results Visualization**
   - Confusion matrix, feature importance charts
   - Performance plots (e.g., ROC curves)

## 📂 Project Structure

```
obesity_level_prediction/
├── data/                  # Dataset and data dictionary
├── notebook/              # Jupyter notebooks for EDA, modeling, and evaluation
├── models/                # Trained model files (optional)
├── results/               # Visualizations and performance outputs
└── obesity_level_prediction.ipynb  # Main notebook
```

## 🛠️ Technologies Used

- Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
- Jupyter Notebook
- Optional: XGBoost, LightGBM, TensorFlow (if applied)

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/obesity-level-prediction.git
   cd obesity-level-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

4. Open `obesity_level_prediction.ipynb` and run all cells.

## 📈 Results Summary

| Model               | Accuracy | F1 Score |
|--------------------|----------|----------|
| Random Forest       | 92.3%    | 0.91     |
| Logistic Regression | 85.6%    | 0.84     |
| SVM                 | 88.1%    | 0.87     |

*(Example results, replace with actuals)*

## 📚 References

- [Original Dataset Source (UCI or Kaggle)](https://link-to-dataset)
- Research literature on obesity classification

## 👤 Author

**Robin Shah**  
BSc. in Computer Science | Data Analyst & ML Enthusiast  
[LinkedIn](https://linkedin.com/in/robinkshah) | [GitHub](https://github.com/robinkshah)
