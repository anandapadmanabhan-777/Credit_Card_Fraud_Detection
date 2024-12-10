
# Credit Card Fraud Detection

This project demonstrates how to build a machine learning model to detect fraudulent credit card transactions using a dataset of transaction records. The model is built using **Artificial Neural Networks (ANN)** and is trained to classify transactions as either fraudulent or non-fraudulent.

## Project Overview

The goal of this project is to develop a model that can detect fraudulent transactions in a given credit card dataset. The model is based on **Artificial Neural Networks (ANN)** and uses various techniques for data preprocessing, feature engineering, and model training to achieve high accuracy in fraud detection.

## Dataset

The dataset contains features related to customer transactions, such as:

- **TransactionAmount**: Amount of the transaction.
- **TransactionDate**: Date and time of the transaction.
- **Location**: Geographic location of the transaction.
- **TransactionType**: Type of transaction (purchase, refund, etc.).
- **Label**: Whether the transaction is fraudulent or not (1 for fraud, 0 for non-fraud).

### Dataset Source
You can upload your own dataset or use a standard dataset such as `Kaggle's Credit Card Fraud Detection dataset` or any other similar dataset.

## Steps Involved

### 1. **Data Preprocessing**
- **Handling Missing Data**: The dataset is checked for null values, and appropriate imputation or removal is done.
- **Removing Duplicates**: Duplicate records are removed.
- **Encoding Categorical Variables**: Non-numerical columns like `TransactionType`, `Location`, etc., are encoded using Label Encoding or One-Hot Encoding.
- **Outlier Handling**: Outliers in numerical features (e.g., `Amount`, `TransactionTime`) are treated to avoid bias in the model.
  
### 2. **Feature Engineering**
- **Temporal Features**: Extracting features like the **hour**, **day**, and **weekend** from the `TransactionDate` column.
- **Geographical Features**: If `Location` is included, geographical information is encoded or removed based on model performance.

### 3. **Model Building**
- **Artificial Neural Network (ANN)**: A simple ANN model is built with:
  - Input layer corresponding to the number of features.
  - Hidden layers with **ReLU activation** functions.
  - Output layer with a **sigmoid activation** function for binary classification (fraud or non-fraud).
  
- **Compilation**: The model is compiled using the **Adam optimizer** and **binary crossentropy loss** for classification tasks.

### 4. **Model Training and Evaluation**
- The dataset is split into training and testing sets.
- The model is trained for multiple epochs (e.g., 20 epochs) with a batch size of 32.
- The model's performance is evaluated using **accuracy**, **precision**, **recall**, and **F1-score**.

### 5. **Class Imbalance Handling**
- Techniques like **SMOTE** (Synthetic Minority Over-sampling Technique) are used to handle class imbalance between fraudulent and non-fraudulent transactions.

## Technologies Used
- **Python**
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations.
- **Scikit-learn**: For data preprocessing, model evaluation, and metrics.
- **TensorFlow/Keras**: For building and training the Artificial Neural Network (ANN).
- **Matplotlib/Seaborn**: For data visualization.

## Requirements

- Python 3.x
- TensorFlow 2.x
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

## How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
   ```

2. Navigate to the project directory:
   ```bash
   cd credit-card-fraud-detection
   ```

3. Place the dataset (`credit_card_fraud_dataset.csv`) in the project directory.

4. Run the notebook or script to preprocess the data, build the model, and train it:
   ```bash
   python fraud_detection_model.py
   ```

5. After training, the model will output the **accuracy**, **precision**, **recall**, and **F1-score** on the test dataset.

## Example Output

```plaintext
Model Accuracy: 98.2%
Confusion Matrix:
[[5000   10]
 [  15  100]]
Precision for fraud: 0.91
Recall for fraud: 0.87
F1 Score for fraud: 0.89
```

## Contributing

If you would like to contribute to this project, feel free to fork the repository, make changes, and submit a pull request. Contributions are welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### Optional Additions to README:
- **Link to Dataset**: If the dataset is available on an external source (e.g., Kaggle), you can provide a link to the dataset.
- **Future Enhancements**: You can list any possible future improvements or techniques you want to apply to the project (e.g., implementing more complex models like XGBoost, or improving the model’s performance using hyperparameter tuning).

---

This should serve as a solid foundation for your project. Let me know if you'd like me to help with any specific sections or further customizations!
