import argparse
import os
import numpy as np
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core import dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import joblib

#get the dataset
web_paths = ['https://raw.githubusercontent.com/Ayoyinka-Sofuwa/Capstone-project/main/Breast_cancer_data.csv']
ds = TabularDatasetFactory.from_delimited_files(path=web_paths, separator=',')

#split data
def split_data(data):
	df = data.to_pandas_dataframe().dropna()
	y_df= df['diagnosis']
	df.drop(['diagnosis'],inplace=True, axis=1)
	x_df = df
	
	return x_df, y_df

x,y = split_data(ds)

run = Run.get_context()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.25, random_state=42)


def main():
	parser= argparse.ArgumentParser()
	
	parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
	parser.add_argument('--max_iter', type=int, default=100, help='Maximum numberof iterations to converge')

	args = parser.parse_args()
	
	run.log('Regularization Strength:', np.float(args.C))
	run.log('Max iterations:', np.int(args.max_iter))

	model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)
	
	accuracy= model.score(x_test,y_test)
	run.log('Accuracy', np.float(accuracy))
	
	os.makedirs('outputs', exist_ok=True)
	joblib.dump(model, 'outputs/model.joblib')
	
if __name__ == '__main__':
    main()
