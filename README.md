
# Iris Species Classification using Neural Networks

## Objective

The objective of this project is to classify Iris species (Iris-setosa, Iris-versicolor, and Iris-virginica) based on the measurements of sepal and petal length and width using a neural network model implemented in Keras (TensorFlow). This project demonstrates the use of data preprocessing, neural network architecture design, and model evaluation for a multi-class classification task.

## Dataset Description

The dataset used in this project is the Iris dataset, which is stored in `iris.xlsx`. It contains 150 samples of Iris flowers and the following features:

- **SepalLengthCm**: Length of the sepal in centimeters.
- **SepalWidthCm**: Width of the sepal in centimeters.
- **PetalLengthCm**: Length of the petal in centimeters.
- **PetalWidthCm**: Width of the petal in centimeters.
- **species**: The species of the Iris flower, which can be one of the following:
  - Iris-setosa
  - Iris-versicolor
  - Iris-virginica

The target variable for classification is the species of the flower.

## Project Steps

1. **Data Loading and Exploration**: The dataset is loaded, and basic exploration is done to understand the structure and summary statistics.
2. **Data Preprocessing**: 
   - Scaling of features using `StandardScaler`.
   - One-hot encoding of the target variable (`species`) for multi-class classification.
3. **Train-Test Split**: The dataset is split into training and testing sets (80% train, 20% test).
4. **Model Architecture**:
   - Input Layer
   - Two hidden layers with 64 and 32 neurons, respectively, using ReLU activation.
   - Output layer with 3 neurons and softmax activation for classification.
5. **Model Compilation and Training**: The model is compiled using the Adam optimizer and categorical cross-entropy as the loss function. It is trained and evaluated on the dataset.
6. **Model Evaluation**: Performance is evaluated using:
   - Test Accuracy
   - Confusion Matrix
   - Classification Report (Precision, Recall, F1-score)
   - Multi-class ROC-AUC score
7. **Visualization**: Plots for accuracy and loss over epochs are generated.

## Steps to Run the Code in Google Colab

To run the code in Google Colab, follow these steps:

1. Upload the dataset (`iris.xlsx`) to your Google Drive.
2. Open Google Colab and upload the Python script or copy the code into a new notebook.
3. Mount your Google Drive by running the following command in a cell:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. Change the path to load the dataset from your Google Drive:
   ```python
   data = pd.read_excel('/content/drive/My Drive/path_to_your_dataset/iris.xlsx')
   ```
5. Install the required dependencies if they are not already installed:
   ```bash
   !pip install pandas numpy scikit-learn matplotlib seaborn tensorflow
   ```
6. Run the cells in sequence to execute the code.

## Dependencies and Installation Instructions

The following dependencies are required to run the project:

- **pandas**: For data manipulation and loading the dataset.
- **numpy**: For numerical computations.
- **scikit-learn**: For train-test split, scaling, and model evaluation.
- **matplotlib & seaborn**: For plotting graphs and visualizations.
- **tensorflow (Keras)**: For building and training the neural network.

To install the dependencies, run the following command:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn tensorflow
```

## Results

- **Test Accuracy**: The accuracy of the model on the test set is displayed after model evaluation.
- **Confusion Matrix and Classification Report**: Used to analyze the classification performance across the three species.
- **ROC-AUC Score**: Evaluates the model's ability to distinguish between the three classes using a one-vs-rest approach.

## Visualizations

During training, the script generates the following plots:
- **Accuracy vs. Epochs**: Shows how the accuracy improves over the training epochs.
- **Loss vs. Epochs**: Displays how the loss decreases during training.

## License

This project is licensed under the MIT License.

## Acknowledgments

The Iris dataset is a well-known dataset in the machine learning community. Special thanks to the contributors of TensorFlow, Keras, and Scikit-learn for their excellent libraries.

---

This README now includes all the required components:
- **Objective**: Added at the beginning to explain the project's purpose.
- **Dataset description**: Provided more detail about the dataset and its features.
- **Steps to run the code in Colab**: Added clear instructions for running the project in Google Colab, including Google Drive integration.
- **Dependencies and installation instructions**: Listed the required dependencies and how to install them.

Let me know if you'd like any further adjustments!
