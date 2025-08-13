# iris-flower-classification
## Objective
The objective of this project is to classify iris flowers into one of three species:
- Setosa
- Versicolor
- Virginica

based on four measurements:
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

---

##  Dataset
- Source: Built-in Iris dataset from [scikit-learn](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-dataset)  
  (originally from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris))
- Size: 150 samples (50 per species)
- Features: 4 numeric measurements
- Target: Species label (0, 1, 2 → Setosa, Versicolor, Virginica)

---

##  Steps I Followed

### 1. Data Loading
- Used `load_iris()` from scikit-learn to directly import the dataset.
- Converted it into a Pandas DataFrame for easier handling.

### 2. Exploratory Data Analysis (EDA)
- Checked data types, missing values, and basic statistics.
- Created visualizations using Seaborn and Matplotlib
  - Pair plots to see separability between species.
  - Histograms to understand feature distributions.
  - Boxplots to compare features by species.

### 3. Train-Test Split
- Used `train_test_split()` to split data into:
  - 80% training data
  - 20% testing data
- Stratified the split to maintain class balance.

### 4. Data Preprocessing
- Standardized features using `StandardScaler` to ensure all measurements are on the same scale.
- This step improves performance for algorithms like Logistic Regression and KNN.

### 5. Model Training
- Trained a Logistic Regression classifier as the baseline model.
- Also experimented with:
  - K-Nearest Neighbors (KNN)
  - Decision Tree Classifier

### 6. Model Evaluation
- Measured performance using:
  - Accuracy score
  - Classification report (precision, recall, F1-score)
  - Confusion matrix (visualized with Seaborn heatmap)
- Observed that most errors occurred between Versicolor and Virginica (common in Iris dataset).

---

## Results
| Model               | Accuracy |
|---------------------|----------|
| Logistic Regression | 97%      |
| KNN (k=5)           | 96%      |
| Decision Tree       | 100%     |

- Setosa was classified with perfect accuracy in all models.
- Slight confusion between Versicolor and Virginica for Logistic Regression and KNN.

---

##  Tools & Libraries Used
- Python 3
- Pandas → Data handling
- NumPy→ Numerical computations
- Matplotlib / Seaborn → Data visualization
- scikit-learn → Model building, training, and evaluation



 How to Run This Project
1. Clone the repository
   ```bash
   git clone https://github.com/saiharshitha055/iris-flower-classification.git
   cd iris-flower-classification
Install dependencies

bash
Copy
Edit
pip install numpy pandas scikit-learn matplotlib seaborn
Run the Jupyter Notebook

bash
Copy
Edit
jupyter notebook iris_classification.ipynb
Execute all cells to reproduce the results.

## Future Improvements
Deploy the model using Streamlit for interactive predictions.

Experiment with ensemble methods (Random Forest, Gradient Boosting).

Perform hyperparameter tuning for even better accuracy.
