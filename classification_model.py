"""
==========================================================
Project Title : Data Classification Using AI
Internship    : DecodeLabs AI Internship - Project 2
Author        : Janki Rathod

Description:
This project demonstrates supervised machine learning
using the Iris dataset. It performs data exploration,
visualization, model training, evaluation and prediction
of flower species.
==========================================================
"""

# =========================Import Required Libraries=================================

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay)

# ==========================Create Project Folders================================

def display_heading():

    print("=" * 60)
    print("        DATA CLASSIFICATION USING AI")
    print("=" * 60)
    print("DecodeLabs AI Internship - Project 2")
    print("=" * 60)

# ========================Create Project Folders==================================

def create_folders():

    os.makedirs("images", exist_ok=True)
    os.makedirs("model", exist_ok=True)
    print("\nProject folders created successfully.")

# =============================Load Iris Dataset=============================

def load_dataset():

    print("\nLoading Iris Dataset...")
    iris = load_iris()
    dataframe = pd.DataFrame(
        iris.data,
        columns=iris.feature_names)
    dataframe["Species"] = iris.target

    print("Dataset loaded successfully.")
    return iris, dataframe

# ==========================Explore Dataset================================

def explore_dataset(dataframe):

    print("\n" + "=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)

    print("\nFirst Five Records")
    print("-" * 60)

    print(dataframe.head())

    print("\nDataset Shape")
    print("-" * 60)

    print(dataframe.shape)

    print("\nColumn Names")
    print("-" * 60)

    print(dataframe.columns.tolist())

    print("\nMissing Values")
    print("-" * 60)

    print(dataframe.isnull().sum())

    print("\nStatistical Summary")
    print("-" * 60)

    print(dataframe.describe())

    print("\nSpecies Distribution")
    print("-" * 60)

    print(dataframe["Species"].value_counts())

# =======================Data Visualization===================================

def visualize_data(dataframe):

    print("\nGenerating Data Visualizations...")

    # ----------------------Graph 1 : Class Distribution--------------------------------
    plt.figure(figsize=(6, 4))
    species_count = dataframe["Species"].value_counts()
    species_names = ["Setosa","Versicolor","Virginica"]

    plt.bar(
        species_names,
        species_count.values,
        color=["skyblue", "lightgreen", "salmon"],
        edgecolor="black")

    plt.title("Class Distribution")
    plt.xlabel("Flower Species")
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    plt.savefig("images/class_distribution.png")
    plt.show()
    print("✓ Class Distribution Graph Saved")

    # --------------------------Graph 2 : Feature Histograms----------------------------

    dataframe.hist(
        figsize=(10, 8),
        edgecolor="black")

    plt.suptitle("Feature Distribution", fontsize=15)
    plt.tight_layout()
    plt.savefig("images/feature_histograms.png")
    plt.show()
    print("✓ Feature Histograms Saved")

    # ----------------------Graph 3 : Scatter Plot--------------------------------

    plt.figure(figsize=(7, 5))

    scatter = plt.scatter(
        dataframe.iloc[:, 0],
        dataframe.iloc[:, 2],
        c=dataframe["Species"],
        cmap="viridis",
        s=70,
        edgecolors="black")

    plt.title("Sepal Length vs Petal Length")
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Petal Length (cm)")

    legend_labels = ["Setosa","Versicolor","Virginica"]
    handles, _ = scatter.legend_elements()

    plt.legend(handles,legend_labels,title="Species")
    plt.tight_layout()
    plt.savefig("images/scatter_plot.png")
    plt.show()
    print("✓ Scatter Plot Saved")
    print("\nAll graphs have been saved inside the 'images' folder.")

# ============================Train Machine Learning Models==============================

def train_models(iris):

    print("\n" + "=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)

    # -------------------------Prepare Features and Labels-----------------------------
    X = iris.data
    y = iris.target

    # ------------------------Split Dataset------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,y,test_size=0.20,random_state=42,stratify=y)

    print("\nDataset split completed successfully.")

    # -----------------------Decision Tree Classifier-------------------------------
    print("\nTraining Decision Tree Classifier...")

    decision_tree = DecisionTreeClassifier(random_state=42)
    decision_tree.fit(X_train,y_train)
    dt_prediction = decision_tree.predict(X_test)
    dt_accuracy = accuracy_score(y_test,dt_prediction)

    print(f"Decision Tree Accuracy : {dt_accuracy * 100:.2f}%")

    # -----------------------K-Nearest Neighbors Classifier-------------------------------
    print("\nTraining K-Nearest Neighbors Classifier...")

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train,y_train)
    knn_prediction = knn.predict(X_test)
    knn_accuracy = accuracy_score(y_test,knn_prediction)

    print(f"KNN Accuracy           : {knn_accuracy * 100:.2f}%")

    # ----------------------Compare Models--------------------------------
    print("\n" + "-" * 60)
    print("MODEL COMPARISON")
    print("-" * 60)
    print(f"Decision Tree : {dt_accuracy * 100:.2f}%")
    print(f"KNN           : {knn_accuracy * 100:.2f}%")

    # ------------------------Select Best Model------------------------------

    if dt_accuracy >= knn_accuracy:
        best_model = decision_tree
        best_prediction = dt_prediction
        best_accuracy = dt_accuracy
        best_model_name = "Decision Tree"

    else:
        best_model = knn
        best_prediction = knn_prediction
        best_accuracy = knn_accuracy
        best_model_name = "K-Nearest Neighbors"

    print("\n" + "=" * 60)
    print("BEST MODEL")
    print("=" * 60)
    print(f"Model Name : {best_model_name}")
    print(f"Accuracy   : {best_accuracy * 100:.2f}%")

    return (
        best_model,best_model_name,best_prediction,best_accuracy,X_test,y_test
    )

# ============================Evaluate Machine Learning Model==============================

def evaluate_model(model_name, prediction, y_test):

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    # ------------------------Classification Report------------------------------
    print("\nClassification Report")
    print("-" * 60)
    report = classification_report(y_test,prediction,
        target_names=["Setosa","Versicolor","Virginica"]
    )
    print(report)
    # ---------------------Confusion Matrix---------------------------------
    print("\nGenerating Confusion Matrix...")
    matrix = confusion_matrix(y_test,prediction)
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Setosa","Versicolor","Virginica"]
    )
    fig, ax = plt.subplots(figsize=(6, 5))

    display.plot(cmap="Blues",ax=ax,colorbar=False)

    plt.title(f"Confusion Matrix ({model_name})")
    plt.tight_layout()
    plt.savefig("images/confusion_matrix.png",dpi=300)
    plt.show()
    print("✓ Confusion Matrix Saved")

    # ----------------------Accuracy Summary--------------------------------
    print("\n" + "=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    print(f"Selected Model : {model_name}")
    accuracy = accuracy_score(y_test,prediction)
    print(f"Final Accuracy : {accuracy * 100:.2f}%")
    print("\nModel evaluation completed successfully.")

# =========================Save Trained Machine Learning Model=================================
def save_model(model):

    print("\n" + "=" * 60)
    print("SAVING TRAINED MODEL")
    print("=" * 60)

    model_path = os.path.join("model","iris_classifier.pkl")

    try:
        joblib.dump(model,model_path)
        print("\n✓ Model saved successfully.")
        print(f"Location : {model_path}")

    except Exception as error:
        print("\nError while saving model.")
        print(error)

# ======================Load Saved Machine Learning Model====================================
def load_saved_model():

    print("\n" + "=" * 60)
    print("LOADING SAVED MODEL")
    print("=" * 60)

    model_path = os.path.join("model","iris_classifier.pkl")

    if not os.path.exists(model_path):
        print("\nSaved model not found.")
        return None
    try:
        loaded_model = joblib.load(model_path)
        print("\n✓ Model loaded successfully.")
        return loaded_model
    except Exception as error:
        print("\nUnable to load model.")
        print(error)
        return None

# =====================Display Available Flower Species=====================================

def display_species_information():

    print("\n" + "=" * 60)
    print("FLOWER SPECIES")
    print("=" * 60)
    print("0 --> Setosa")
    print("1 --> Versicolor")
    print("2 --> Virginica")
    print("\nThe trained model predicts one")
    print("of the above three flower species.")

# ====================Predict Flower Species=======================

def predict_species(model):

    print("\n" + "=" * 60)
    print("FLOWER SPECIES PREDICTION")
    print("=" * 60)

    species_names = ["Setosa","Versicolor","Virginica"]

    while True:
        try:
            print("\nEnter Flower Measurements")
            sepal_length = float(input("Sepal Length (cm) : "))
            sepal_width = float(input("Sepal Width (cm)  : "))
            petal_length = float(input("Petal Length (cm) : "))
            petal_width = float(input("Petal Width (cm)  : "))

            sample = [[sepal_length,sepal_width,petal_length,petal_width]]

            prediction = model.predict(sample)
            predicted_species = species_names[prediction[0]]

            print("\n" + "-" * 60)
            print("PREDICTION RESULT")
            print("-" * 60)
            print(f"Predicted Flower Species : {predicted_species}")
            print("-" * 60)

        except ValueError:
            print("\nInvalid Input!")
            print("Please enter only numeric values.")
            continue

        except Exception as error:
            print("\nPrediction Failed!")
            print(error)
            continue

        choice = input("\nDo you want to predict another flower? (Y/N): ").strip().lower()

        if choice != "y":
            print("\nThank you for using the application.")
            print("Program completed successfully.")
            break

# =============Display Project Completion Message=============================

def project_completed():

    print("\n" + "=" * 60)
    print("PROJECT COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("All tasks executed successfully.")
    print("✓ Dataset Loaded")
    print("✓ Dataset Explored")
    print("✓ Graphs Generated")
    print("✓ Models Trained")
    print("✓ Best Model Selected")
    print("✓ Model Evaluated")
    print("✓ Model Saved")
    print("✓ Prediction Completed")
    print("=" * 60)

# =======================Main Function===================================

def main():

    # Display project heading
    display_heading()

    # Create required folders
    create_folders()

    # Load dataset
    iris, dataframe = load_dataset()

    # Display dataset information
    explore_dataset(dataframe)

    # Generate graphs
    visualize_data(dataframe)

    # Train machine learning models
    (best_model,model_name,prediction,accuracy,X_test,y_test) = train_models(iris)

    # Evaluate the selected model
    evaluate_model(model_name,prediction,y_test)
    save_model(best_model)
    loaded_model = load_saved_model()
    display_species_information()

    # Predict flower species
    if loaded_model is not None:
        predict_species(loaded_model)
    else:
        print("\nPrediction cannot continue because the model was not loaded.")

    # Completion message
    project_completed()

# =======================Program Starts Here===================================
if __name__ == "__main__":
    main()