
# """Data Processing and Feature engineering  ."""

# import argparse
# import logging
# import pathlib

# import boto3
# import numpy as np
# import pandas as pd

# logger = logging.getLogger()
# logger.setLevel(logging.INFO)
# logger.addHandler(logging.StreamHandler())

# if __name__ == "__main__":
#     logger.info("Starting preprocessing.")
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input-data", type=str, required=True)
#     args = parser.parse_args()

#     base_dir = "/opt/ml/processing"
#     pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
#     input_data = args.input_data
#     print(input_data)
#     bucket = input_data.split("/")[2]
#     key = "/".join(input_data.split("/")[3:])

#     logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
#     fn = f"{base_dir}/data/raw-data.csv"
#     s3 = boto3.resource("s3")
#     s3.Bucket(bucket).download_file(key, fn)

#     logger.info("Reading downloaded data.")

#     # read in csv
#     df = pd.read_csv(fn)

#     # drop the "Phone" feature column
#     df = df.drop(["Phone"], axis=1)

#     # Change the data type of "Area Code"
#     df["Area Code"] = df["Area Code"].astype(object)

#     # Drop several other columns
#     df = df.drop(["Day Charge", "Eve Charge", "Night Charge", "Intl Charge"], axis=1)

#     # Convert categorical variables into dummy/indicator variables.
#     model_data = pd.get_dummies(df)

#     # Create one binary classification target column
#     model_data = pd.concat(
#         [
#             model_data["Churn?_True."],
#             model_data.drop(["Churn?_False.", "Churn?_True."], axis=1),
#         ],
#         axis=1,
#     )

#     # Split the data
#     train_data, validation_data, test_data = np.split(
#         model_data.sample(frac=1, random_state=1729),
#         [int(0.7 * len(model_data)), int(0.9 * len(model_data))],
#     )

#     pd.DataFrame(train_data).to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
#     pd.DataFrame(validation_data).to_csv(
#         f"{base_dir}/validation/validation.csv", header=False, index=False
#     )
#     pd.DataFrame(test_data).to_csv(f"{base_dir}/test/test.csv", header=False, index=False)


import os
import tempfile
import numpy as np
import pandas as pd
import datetime as dt

if __name__ == "__main__":
    
    base_dir = "/opt/ml/processing"
    #Read Data
    df = pd.read_csv(
        f"{base_dir}/input/storedata_total.csv"
    )
    # convert created column to datetime
    df["created"] = pd.to_datetime(df["created"])
    #Convert firstorder and lastorder to datetime datatype
    df["firstorder"] = pd.to_datetime(df["firstorder"],errors='coerce')
    df["lastorder"] = pd.to_datetime(df["lastorder"],errors='coerce')
    #Drop Rows with Null Values
    df = df.dropna()
    #Create column which gives the days between the last order and the first order
    df['first_last_days_diff'] = (df['lastorder'] - df['firstorder']).dt.days
    #Create column which gives the days between the customer record was created and the first order
    df['created_first_days_diff'] = (df['created'] - df['firstorder']).dt.days
    #Drop columns
    df.drop(['custid', 'created','firstorder','lastorder'], axis=1, inplace=True)
    #Apply one hot encoding on favday and city columns
    df = pd.get_dummies(df, prefix=['favday', 'city'], columns=['favday', 'city'])
    # Split into train, validation and test datasets
    y = df.pop("retained")
    X_pre = df
    y_pre = y.to_numpy().reshape(len(y), 1)
    X = np.concatenate((y_pre, X_pre), axis=1)
    np.random.shuffle(X)
    
    # Split in Train, Test and Validation Datasets
    train, validation, test = np.split(X, [int(.7*len(X)), int(.85*len(X))])
    train_rows = np.shape(train)[0]
    validation_rows = np.shape(validation)[0]
    
    test_rows = np.shape(test)[0]
    train = pd.DataFrame(train)
    test = pd.DataFrame(test)
    validation = pd.DataFrame(validation)
    
    # Convert the label column to integer
    train[0] = train[0].astype(int)
    test[0] = test[0].astype(int)
    validation[0] = validation[0].astype(int)
    
    # Save the Dataframes as csv files
    train.to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    validation.to_csv(f"{base_dir}/validation/validation.csv", header=False, index=False)
    test.to_csv(f"{base_dir}/test/test.csv", header=False, index=False)