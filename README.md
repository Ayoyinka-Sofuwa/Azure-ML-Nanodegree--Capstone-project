# Capstone-project
This is the final Udacity Nanodegree Project for the 7-month Machine Learning Engineer with Azure course training.

### Introduction
This project is focused on tuning hyperparameters using the hyperdrive and the automated ML method to train more models faster and automatically 
### Overview
For this experiment I decided to work on a breast cancer dataset which affects a large population of women all around the world. This dataset focuses on the lumps that appear in the breast, exploring the properites in size, texture, smoothness etc to predict if it is cancerous or not, which is the description of the diagnosis column
This data was gotten from [Kaggle](https://www.kaggle.com/merishnasuwal/breast-cancer-prediction-dataset)
I used two different kinds of experiments to make this prediction and compared which model performed best.
The two experiments are the [Automated Machine learning](https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-first-experiment-automated-ml) and the [Hyperdrive](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive?view=azure-ml-py) experiments.


### Architectural diagram
<p align="center">
  <img src="https://github.com/Ayoyinka-Sofuwa/Azure-ML-Nanodegree--Capstone-project/blob/main/architectural%20diagram.jpg">
</p>


### Task
This task is a classification task in a supervised learning operation that predicts whether the diagnosis will either be a cancerous lump or a non cancerous lump.
The features of this data include: 

* The mean radius
* mean texture
* mean perimeter
* mean area
* mean smoothness
* diagnosis

It contains 569 observations(rows) and 6 features(columns)

I am seeking to predict the diagnosis column which is the labelled data containing details of lumps that have been recorded as cancerous(1) or not cancerous(0).

### Access
I accessed the data using the URL, using the delimited files method from the Tabular Dataset Factory and I registered it in my workspace using the code:

`data = 'https://raw.githubusercontent.com/Ayoyinka-Sofuwa/Capstone-project/main/Breast_cancer_data.csv'
dataset = Dataset.Tabular.from_delimited_files(data)        
dataset = dataset.register(workspace=ws,name=key,description=description_text)`


## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remember to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with its parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search
Hyperparameters are adjustable parameters we choose for model training that guide the training process. The HyperDrive package helps to automate choosing these parameters.
For my logistic regression experiment, the parameters I used in the search space are C and max_iter. I ran a RansomSampling method over the search space because it iterates much faster over the search space, my primary metric was the accuracy metric and 
My parameter search space was defined using the C(continuous) and the max_iter (discrete)

For example, you can define the parameter search space as discrete or continuous, and a sampling method over the search space as random, grid, or Bayesian. Also, you can specify a primary metric to optimize in the hyperparameter tuning experiment, and whether to minimize or maximize that metric. You can also define early termination policies in which poorly performing experiment runs are canceled and new ones started

### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
