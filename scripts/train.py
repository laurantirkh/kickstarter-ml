import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
import pickle


def split_data(df):
    # define features and target
    X = df.drop('state', axis=1)
    y = df['state']

    # split into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)

    # save splits into dictionary
    data = {"train": {"X": X_train, "y": y_train},
            "test": {"X": X_test, "y": y_test}}

    return data

def train_model(data, args):
    rf_model = RandomForestClassifier(**args)
    rf_model.fit(data["train"]["X"], data["train"]["y"])
    return rf_model

def get_model_metrics(model, data):
    preds = model.predict(data["test"]["X"])
    f1 = f1_score(data["test"]["y"], preds)
    metrics = {"f1_score": f1}
    return metrics


def main():
    # load data
    df = pd.read_csv("data/df_preprocessed_log_dummies.csv")

    # split data into train and test set
    data = split_data(df)

    # train model
    args = {
        'n_estimators': 200,
        'min_samples_split': 200,
        'min_samples_leaf': 50,
        'max_features': 'sqrt',
        'max_depth': 50,
        'bootstrap': False
        }

    rf = train_model(data, args)

    # get model metrics
    metrics = get_model_metrics(rf, data)
    print(metrics)

    # save model
    model_name = 'models/random_forest_model.sav'
    pickle.dump(rf, open(model_name, 'wb'))

if __name__ == '__main__':
    main()