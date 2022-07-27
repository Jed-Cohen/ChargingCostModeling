import joblib
import numpy as np
import pandas as pd


def apply_model(infile, outfile, location, features):
    model = joblib.load(location)
    zip_data = pd.read_csv(infile)
    zip_arr = zip_data['ZIP']
    for column in list(zip_data.columns):
        if column not in features:
            zip_data = zip_data.drop(column, axis=1)
    zip_data = zip_data.to_numpy()
    outputs = model.predict(zip_data)
    results = pd.DataFrame()
    results['ZIP'] = zip_arr
    results['Predicted Cost'] = outputs

    results.to_csv(outfile)


if __name__ == '__main__':
    # You can either enter filenames here or during runtime
    input_filename = input("Enter Input File Name: ")
    # input_filename = 'Data/final_data.csv'
    output_filename = input("Enter Input File Name: ")
    # output_filename = 'Data/us_model_output.csv'
    model_file = input("Enter Model File Name: ")
    # model_file = 'USA_model.sav'
    # these are the features that are included in your model
    feature_list = ('Land Value', 'Economic Activity', 'Station Count', 'Sales Tax')
    apply_model(input_filename, output_filename, model_file, feature_list)
