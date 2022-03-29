import argparse
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from azureml.core import Workspace, Dataset, Run



def main():
    
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-data", type=str)
    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run = Run.get_context()
    ws = run.experiment.workspace

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    df = Dataset.get_by_id(ws, id=args.input_data)
    df = df.to_pandas_dataframe()
    df = df.drop(['stem-root', 'veil-type', 'veil-color'], axis=1).fillna('Other')
    
    x, y = df.drop('class', axis=1), df['class']

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.5, random_state=42)

    enc = OneHotEncoder(handle_unknown='ignore')
    x_train_cat, x_train_num = x_train.select_dtypes(include='object'), x_train.select_dtypes(exclude='object').to_numpy()
    x_test_cat, x_test_num = x_test.select_dtypes(include='object'), x_test.select_dtypes(exclude='object').to_numpy()
    x_train_enc = np.concatenate((x_train_num, enc.fit_transform(x_train_cat).toarray()), axis=1)
    x_test_enc = np.concatenate((x_test_num, enc.transform(x_test_cat).toarray()), axis=1)

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train_enc, y_train)

    accuracy = model.score(x_test_enc, y_test)
    run.log("Accuracy", np.float(accuracy))
    joblib.dump(model, './outputs/hyperdrive_model.joblib')
    

if __name__ == '__main__':
    main()