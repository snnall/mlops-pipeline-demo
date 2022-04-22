
# """Evaluation script"""

# import json
# import logging
# import os
# import pickle
# import tarfile

# import pandas as pd
# import xgboost

# logger = logging.getLogger()
# logger.setLevel(logging.INFO)
# logger.addHandler(logging.StreamHandler())
# from sklearn.metrics import accuracy_score, classification_report, roc_auc_score,precision_recall_curve,precision_recall_curve,auc

# if __name__ == "__main__":
#     model_path = "/opt/ml/processing/model/model.tar.gz"
#     with tarfile.open(model_path) as tar:
#         tar.extractall(path=".")

#     logger.debug("Loading xgboost model.")
#     model = pickle.load(open("xgboost-model", "rb"))

#     print("Loading test input data")
#     test_path = "/opt/ml/processing/test/test.csv"
#     df = pd.read_csv(test_path, header=None)

#     logger.debug("Reading test data.")
#     y_test = df.iloc[:, 0].to_numpy()
#     df.drop(df.columns[0], axis=1, inplace=True)
#     X_test = xgboost.DMatrix(df.values)

#     logger.info("Performing predictions against test data.")
#     predictions = model.predict(X_test)

#     print("Creating classification evaluation report")
#     acc = accuracy_score(y_test, predictions.round())
#     #auc = roc_auc_score(y_test, predictions.round())

#     # Data to plot precision - recall curve
#     precision, recall, thresholds = precision_recall_curve(y_test, predictions.round())
    
#     # Use AUC function to calculate the area under the curve of precision recall curve
#     pr_auc = auc(recall, precision)
    
#     report_dict = {
#         "binary_classification_metrics": {
#             "accuracy": {
#                 "value": acc,
#                 "standard_deviation": "NaN",
#             },
#             "pr_auc": {
#                 "value": pr_auc,
#                 "standard_deviation": "NaN",
#             },
#         }
#     }
    
#     print("Classification report:\n{}".format(report_dict))    
#     logger.info("Writing out evaluation report with accuracy: %f", acc)
#     output_dir = "/opt/ml/processing/evaluation"
#     evaluation_path = f"{output_dir}/evaluation.json"
#     with open(evaluation_path, "w") as f:
#         f.write(json.dumps(report_dict))

import json
import pathlib
import pickle
import tarfile
import joblib
import numpy as np
import pandas as pd
import xgboost
import datetime as dt
from sklearn.metrics import roc_curve,auc


if __name__ == "__main__":
    
    #Read Model Tar File
    model_path = f"/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")
    model = pickle.load(open("xgboost-model", "rb"))
    
    #Read Test Data using which we evaluate the model
    test_path = "/opt/ml/processing/test/test.csv"
    df = pd.read_csv(test_path, header=None)
    y_test = df.iloc[:, 0].to_numpy()
    df.drop(df.columns[0], axis=1, inplace=True)
    X_test = xgboost.DMatrix(df.values)
    
    #Run Predictions
    predictions = model.predict(X_test)
    
    #Evaluate Predictions
    fpr, tpr, thresholds = roc_curve(y_test, predictions)
    auc_score = auc(fpr, tpr)
    report_dict = {
        "classification_metrics": {
            "auc_score": {
                "value": auc_score,
            },
        },
    }
    
    #Save Evaluation Report
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))