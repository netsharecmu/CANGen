import json
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier


def preprocess_data(df):
    # Fill NA with '00' and convert hex strings to integers
    df.fillna('00', inplace=True)
    df['CAN_ID'] = df['CAN_ID'].apply(lambda x: int(x, 16))
    for i in range(8):
        df[f'DATA_{i}'] = df[f'DATA_{i}'].apply(lambda x: int(x, 16))
    return df


def train_test_model(train_csv_path, test_csv_path, results_json_file=None, model_type='decision_tree', model_params=None, sample_size=None):
    # Load and preprocess datasets
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)
    train_df = preprocess_data(train_df)
    test_df = preprocess_data(test_df)

    # Sampling if specified
    if sample_size:
        train_df = train_df.sample(frac=sample_size)

    # Separate features and labels
    X_train = train_df.drop(columns=['Label'])
    y_train = train_df['Label']
    X_test = test_df.drop(columns=['Label'])
    y_test = test_df['Label']

    # Model selection
    if model_type == 'decision_tree':
        model = DecisionTreeClassifier(**(model_params or {}))
    elif model_type == 'logistic_regression':
        model = LogisticRegression(**(model_params or {}))
    elif model_type == 'naive_bayes':
        model = GaussianNB(**(model_params or {}))
    elif model_type == 'mlp':
        model = MLPClassifier(**(model_params or {}))
    elif model_type == 'random_forest':
        model = RandomForestClassifier(**(model_params or {}))
    else:
        raise ValueError(
            f"Unsupported model type '{model_type}'. Supported types are 'decision_tree', 'logistic_regression', 'naive_bayes', 'mlp', 'random_forest'.")

    # Train the model
    model.fit(X_train, y_train)

    # Predictions
    y_pred_test = model.predict(X_test)

    # Results
    results = {
        'classification_report': classification_report(y_test, y_pred_test, output_dict=True),
        'accuracy_score': accuracy_score(y_test, y_pred_test),
        'f1_score': f1_score(y_test, y_pred_test, average='weighted'),
        'confusion_matrix': confusion_matrix(y_test, y_pred_test).tolist()
    }

    # Export results
    if results_json_file:
        with open(results_json_file, 'w') as f:
            json.dump(results, f)
        print(f"Results outputted to {results_json_file}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv_path', type=str,
                        required=True, help='Path to the training CSV file')
    parser.add_argument('--test_csv_path', type=str,
                        required=True, help='Path to the test CSV file')
    parser.add_argument('--results_json_file', type=str,
                        default=None, help='Path to the results JSON file')
    parser.add_argument('--model_type', type=str,
                        default='decision_tree', help='Type of the model')
    parser.add_argument('--model_params', type=json.loads,
                        default=None, help='Model parameters in JSON format')
    parser.add_argument('--sample_size', type=float,
                        default=None, help='Sample size (fraction)')
    args = parser.parse_args()

    train_test_model(
        train_csv_path=args.train_csv_path,
        test_csv_path=args.test_csv_path,
        results_json_file=args.results_json_file,
        model_type=args.model_type,
        model_params=args.model_params,
        sample_size=args.sample_size
    )

    # print(
    #     train_test_model(
    #         '../results/vehiclesec2024/small-scale/csv/netshare_car-hacking-fuzzy-bits-sessionized_20231210215334875529513.csv',
    #         '../data_selected/car_hacking/Fuzzy_dataset_aligned_test.csv',
    #         model_type='decision_tree'
    #     ))

    # print(
    #     train_test_model(
    #         '../data_selected/car_hacking/RPM_dataset_aligned_train.csv',
    #         '../data_selected/car_hacking/RPM_dataset_aligned_test.csv',
    #         model_type='decision_tree'
    #     ))

    # print(
    #     train_test_model(
    #         '../data_selected/car_hacking/RPM_dataset_aligned_train.csv',
    #         '../data_selected/car_hacking/RPM_dataset_aligned_test.csv',
    #         model_type='logistic_regression'
    #     ))

    # print(
    #     train_test_model(
    #         '../data_selected/car_hacking/RPM_dataset_aligned_train.csv',
    #         '../data_selected/car_hacking/RPM_dataset_aligned_test.csv',
    #         model_type='naive_bayes'
    #     ))

    # print(
    #     train_test_model(
    #         '../data_selected/car_hacking/RPM_dataset_aligned_train.csv',
    #         '../data_selected/car_hacking/RPM_dataset_aligned_test.csv',
    #         model_type='mlp'
    #     ))

    # print(
    #     train_test_model(
    #         '../data_selected/car_hacking/RPM_dataset_aligned_train.csv',
    #         '../data_selected/car_hacking/RPM_dataset_aligned_test.csv',
    #         model_type='random_forest'
    #     ))
