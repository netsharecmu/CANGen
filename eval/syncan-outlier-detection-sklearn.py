import json
import argparse
import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def train_test_anomaly_detection(train_csv_path, test_csv_path, results_json_file=None, model_type='ocsvm', model_params=None, sample_size=None):
    tqdm.write("Loading and preprocessing data...")
    # Load and preprocess data
    train_data = pd.read_csv(train_csv_path)
    test_data = pd.read_csv(test_csv_path)
    train_data.fillna(0, inplace=True)
    test_data.fillna(0, inplace=True)

    # Sampling if sample_size is specified
    if sample_size is not None:
        tqdm.write(f"Sampling data: Using {sample_size * 100}% of data")
        train_data = train_data.sample(frac=sample_size, random_state=42)
        test_data = test_data.sample(frac=sample_size, random_state=42)

    # Extract features and labels from test data
    X_train = train_data.drop(columns=['Label', 'Time'])[
        [f'Signal{i+1}' for i in range(4)]]
    X_test = test_data.drop(columns=['Label', 'Time'])[
        [f'Signal{i+1}' for i in range(4)]]
    y_test = test_data['Label']

    # Standardize the data
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    tqdm.write("Initializing and training the model...")
    # Initialize the model
    if model_type == 'ocsvm':
        model = OneClassSVM(**(model_params or {}))
    elif model_type == 'iforest':
        model = IsolationForest(**(model_params or {}))
    elif model_type == 'lof':
        model = LocalOutlierFactor(novelty=True, **(model_params or {}))
    elif model_type == 'kmeans':
        model = KMeans(**(model_params or {}))
    elif model_type == 'dbscan':
        model = DBSCAN(**(model_params or {}))
    else:
        raise ValueError("Unsupported model type")

    # Train and predict
    if model_type in ['ocsvm', 'iforest', 'lof']:
        model.fit(X_train)
        y_pred_test = model.predict(X_test)
        y_pred_test = [1 if pred == -1 else 0 for pred in y_pred_test]
    elif model_type == 'kmeans':
        model.fit(X_train)
        distances = model.transform(X_test)
        min_distances = np.min(distances, axis=1)
        mean_distance = np.mean(min_distances)
        std_distance = np.std(min_distances)
        threshold = mean_distance + 0.1 * std_distance  # Set your threshold factor here
        y_pred_test = [1 if dist > threshold else 0 for dist in min_distances]
    elif model_type == 'dbscan':
        # Fit DBSCAN on the training data
        model.fit(X_train)

        # Use NearestNeighbors to find the distance to the nearest cluster
        nn = NearestNeighbors(n_neighbors=1)
        # Fit only on cluster points, excluding noise
        nn.fit(X_train[model.labels_ != -1])

        # Find the distance and indices of the nearest neighbors in the train set for each point in the test set
        distances, indices = nn.kneighbors(X_test)

        # Define a threshold for distance to consider a point as an anomaly
        # This threshold could be set based on domain knowledge or some percentile of the training distances
        distance_threshold = np.percentile(
            distances[model.labels_[indices] != -1], 80)

        # Points in test set whose distance to the nearest train set neighbor is greater than the threshold are anomalies
        y_pred_test = [1 if dist >
                       distance_threshold else 0 for dist in distances.ravel()]

    tqdm.write("Evaluating the model...")
    # Evaluate the model
    print("Classification Report:\n", classification_report(y_test, y_pred_test))
    print("Accuracy Score:", accuracy_score(y_test, y_pred_test))
    print("F1 Score:", f1_score(y_test, y_pred_test))

    # Confusion matrix for additional metrics
    # tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
    # fpr = fp / (fp + tn)
    # tpr = tp / (tp + fn)

    # print("False Positive Rate:", fpr)
    # print("True Positive Rate:", tpr)

    # tqdm.write("Process completed.")
    # Output results to JSON file if specified
    if results_json_file is not None:
        tqdm.write(f"Outputting results to {results_json_file}")
        with open(results_json_file, 'w') as f:
            json.dump({
                'classification_report': classification_report(y_test, y_pred_test, output_dict=True),
                'accuracy_score': accuracy_score(y_test, y_pred_test),
                'f1_score': f1_score(y_test, y_pred_test),
                'confusion_matrix': confusion_matrix(y_test, y_pred_test).tolist()
            }, f)


# main function
# Example usage:
# python syncan-outlier-detection-sklearn.py --train_csv_path ../results/vehiclesec2024/small-scale/csv/realtabformer-tabular_syncan-flag_20231210131342271428670.csv --test_csv_path ../data_selected/syncan/test_flooding.csv --model_type ocsvm --sample_size 0.01
if __name__ == '__main__':
    # Construct command line argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv_path', type=str,
                        help='Path to the training CSV file')
    parser.add_argument('--test_csv_path', type=str,
                        help='Path to the test CSV file')
    parser.add_argument('--results_json_file', type=str,
                        default=None, help='Path to the results JSON file')
    parser.add_argument('--model_type', type=str,
                        default='ocsvm', help='Type of the model')
    parser.add_argument('--model_params', type=str,
                        default=None, help='Model parameters')
    parser.add_argument('--sample_size', type=float,
                        default=None, help='Sample size')
    args = parser.parse_args()

    train_test_anomaly_detection(
        train_csv_path=args.train_csv_path,
        test_csv_path=args.test_csv_path,
        results_json_file=args.results_json_file,
        model_type=args.model_type,
        model_params=args.model_params,
        sample_size=args.sample_size
    )


# train_test_anomaly_detection(
#     '../data_selected/syncan/train.csv',
#     '../data_selected/syncan/test_normal.csv',
#     model_type='ocsvm',
#     sample_size=0.01)

# train_test_anomaly_detection(
#     '../data_selected/syncan/train.csv',
#     '../data_selected/syncan/test_flooding.csv',
#     model_type='iforest',
#     sample_size=0.01)
# train_test_anomaly_detection(
#     '../data_selected/syncan/train.csv',
#     '../data_selected/syncan/test_flooding.csv',
#     model_type='lof',
#     sample_size=0.01)
# train_test_anomaly_detection(
#     '../data_selected/syncan/train.csv',
#     '../data_selected/syncan/test_flooding.csv',
#     model_type='kmeans',
#     sample_size=0.01)

# train_test_anomaly_detection(
#     '../data_selected/syncan/train.csv',
#     '../data_selected/syncan/test_flooding.csv',
#     model_type='dbscan',
#     sample_size=0.01)

# train_test_anomaly_detection(
#     '../results/vehiclesec2024/small-scale/csv/realtabformer-tabular_syncan-flag_20231210131342271428670.csv',
#     '../data_selected/syncan/test_flooding.csv',
#     model_type='ocsvm',
#     sample_size=0.01)

# train_test_anomaly_detection(
#     '../results/vehiclesec2024/small-scale/csv/tabddpm_syncan-flag_20231212164005631.csv',
#     '../data_selected/syncan/test_flooding.csv',
#     model_type='ocsvm',
#     sample_size=0.01)
