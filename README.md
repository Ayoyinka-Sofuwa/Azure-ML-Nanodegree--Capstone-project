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

In my automl settings, I set the experiment to time out at 30 minutes, run 4 experiments at a time and the primary metric to be highlighted will be the accuracy of each model generated.
It is a classification experiment and my target column to be predicted is the diagnosis column. And I configured the automated ML experiment to be Onnx compatible for deployment.

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?
My best AutoMl model is the VotingEnsemble with an accuracy of 94%, 

*TODO* Remember to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with its parameters.

## Hyperparameter Tuning

Hyperparameters are adjustable parameters we choose for model training that guide the training process. The HyperDrive package helps to automate choosing these parameters.
For my logistic regression experiment, the parameters I used in the search space are C and max_iter. I ran a RansomSampling method over the search space because it iterates much faster than the GridSearch method, my primary metric was the accuracy metric.
My parameter search space was defined using the C(continuous values, ranging uniformly between 0.2 and 0.5) and the max_iter (discrete values, ranging by choice between 2 and 50), with smaller values to get a stronger regularization for the hyperparameter and the maximum number of iterations for the classification algorithm. My estimator is the SKLearn estimator which I used to call the script into the experiment from the directory, and the define the compute target/cluster to be used. 

My hyperdrive configuration included the estimator, the policy, the parameter sampler, the primary metric as "Accuracy" and to maximize it as the goal, maximum concurrent runs, and the total runs for the experiment. I submitted this configuration for the hyperdrive and began the experiment run.


### Results
My best performing model had an accuracy of 91.6%. I could have improved the performance of this model by increasing the range of values in my C and max_iter parameter search space. The parameters were independently set at: regularization strength of 0.2368 and maximum iteration value of 50 

<p align="center">
  <img src="https://raw.githubusercontent.com/Ayoyinka-Sofuwa/Azure-ML-Nanodegree--Capstone-project/5000ac6963fe744349c5aaebf0edc0922eb194cc/screenshots/run%20details%20hyper.png">
</p>

<p align="center">
  <img src="https://github.com/Ayoyinka-Sofuwa/Azure-ML-Nanodegree--Capstone-project/blob/main/screenshots/run%20details%20hyper.png">
</p>

This is the screenshot of the best trained model with its parameters
<p align="center">
  <img src="https://github.com/Ayoyinka-Sofuwa/Azure-ML-Nanodegree--Capstone-project/blob/main/screenshots/best%20run%20hyper.png">
</p>

<p align="center">
  <img src="https://github.com/Ayoyinka-Sofuwa/Azure-ML-Nanodegree--Capstone-project/blob/main/screenshots/best%20run%20details.png">
</p>




## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
