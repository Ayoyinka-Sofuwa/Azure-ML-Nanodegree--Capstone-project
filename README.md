# Capstone-project
This is the final Udacity Nanodegree Project for the 7-month Machine Learning Engineer with Azure training.

### Introduction
This project is focused on generating models by tuning hyperparameters using the hyperdrive and training models with Automated ML faster and automatically, and then deploying and testing the best performing model.

### Overview
For this experiment I decided to work on a breast cancer dataset which affects a large population of women all around the world. This dataset focuses on the lumps that appear in the breast, exploring the properites in size, texture, smoothness etc to predict if it is cancerous or not, which is the description of the diagnosis column.

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

`data = 'https://raw.githubusercontent.com/Ayoyinka-Sofuwa/Capstone-project/main/Breast_cancer_data.csv'`


## Automated ML

(Popularly called AutoML) is the automatic process of model algorithm selection, hyper parameter tuning, iterative modelling, and model assessment.

In my automl settings, I set the experiment to time out at 30 minutes, run 4 experiments at a time and the primary metric to be optimized for model selection will be the accuracy of each model generated.
It is a classification experiment and my target column to be predicted is the diagnosis column. 
The AutoML experiment trains a number of modeld in a short time frame and chooses the best performing model of all the models traied. This is why AutoML is recommended to optimize compute, save time and perform so much better and I configured this automated ML experiment to be Onnx compatible for deployment which makes it easily accessible.

### Results
* The AutoML experiment successfully trained a total of 38 models which generally performed well and the best AutoMl model is the VotingEnsemble with an accuracy of 0.9384 which could've been improved by increasing the run time and if it ran on the compute target.
* Another possible way to improve the output here is to use `AUC_weighted` as my primary metric because of the size of my dataset which isn't very large.

<p align="center">
  <img src="https://github.com/Ayoyinka-Sofuwa/Azure-ML-Nanodegree--Capstone-project/blob/main/screenshot/automl%20run%20widget%201.png">
</p>

<p align="center">
  <img src="https://github.com/Ayoyinka-Sofuwa/Azure-ML-Nanodegree--Capstone-project/blob/main/screenshot/automl%20run%20widget%202.png">
</p>

<p align="center">
  <img src="https://github.com/Ayoyinka-Sofuwa/Azure-ML-Nanodegree--Capstone-project/blob/main/screenshot/automl%20best%20model.png">
</p>


## Hyperparameter Tuning

Hyperparameters are adjustable parameters we choose for model training that guide the training process. The HyperDrive package helps to automate choosing these parameters.

For my logistic regression experiment, the parameters I used in the search space are C and max_iter which was defined using C (continuous values, ranging uniformly between 0.2 and 0.5), (I chose smaller values to get a stronger and better regularization of the model) and the max_iter (discrete values, ranging by choice between 2 and 50), which is the maximum number of iterations for the classification algorithm.

I ran a RansomSampling method over the search space because it iterates much faster than the GridSearch method, my primary metric is the accuracy metric.
The estimator is the SKLearn estimator which was used to call the script into the experiment from the directory, and defined the compute target/cluster to be used.

The early stopping Policy which is defined to delay evaluation until after two iterations, starting from the 5th, is to terminate any model that doesn't perform up to a 91% of the recorded best model.

Overall, my hyperdrive configuration included the estimator, the policy, the parameter sampler, the primary metric as "Accuracy" and to maximize it as the goal, maximum concurrent runs, and the total runs for the experiment. I submitted this configuration for the hyperdrive and began the experiment run.


### Results
My best performing model had an accuracy of 0.9090. I could have improved the performance of this model by increasing the run time and the range of values in the max_iter parameter search space. The parameters were independently and automatically set at: regularization strength of 0.35 and maximum iteration value of 50 

<p align="center">
  <img src="https://github.com/Ayoyinka-Sofuwa/Azure-ML-Nanodegree--Capstone-project/blob/main/screenshot/hyperdrive%20run%20details%201.png">
</p>

<p align="center">
  <img src="https://github.com/Ayoyinka-Sofuwa/Azure-ML-Nanodegree--Capstone-project/blob/main/screenshot/hyperdrive%20run%20details%202.png">
</p>

This is the screenshot of the best trained model with its parameters
<p align="center">
  <img src="https://github.com/Ayoyinka-Sofuwa/Azure-ML-Nanodegree--Capstone-project/blob/main/screenshot/hyperdrive%20bestrun%20studio.png">
</p>


## Model Deployment
The AutoML experiment performed best of the two experiments [Hyperdrive and Automated Machine Learning].

* First, register the model.
Registering a model allows you to store, version, and track metadata about models in your workspace.
The model is registered by defining the model name and description.

On deployment of my AutoML model, I deployed to the Azure Container Instance(ACI)storage which can be accessed by key activated through authentication[Key-based Authentication];
* To set the deployment configuration:
Azure container instance (ACI) webservice takes the deploy_configuration method to define authentication, number of CPU cores and the memory.

* To define the inference configuration:
The inference configuration defines the environment used to run the deployed model. The inference configuration references the following entities, which are used to run the model when it's deployed:
a. An entry script i.e. [the score.py script](https://github.com/Ayoyinka-Sofuwa/Azure-ML-Nanodegree--Capstone-project/blob/main/score.py), which loads the model when the deployed service starts. This script is also responsible for receiving data, passing it to the model, and then returning a response.
b. An Azure Machine Learning environment. An environment defines the software dependencies needed to run the model and entry script.

Then the model is deployed using the workspace, deployment name, model, inference and deployment config.
I then enabled application insight after deployment and retrieved my logs. 

<p align="center">
  <img src="https://github.com/Ayoyinka-Sofuwa/Azure-ML-Nanodegree--Capstone-project/blob/main/screenshot/deployment.png">
</p>


To check how well my model was doing at the endpoint, I wrote a [script](https://github.com/Ayoyinka-Sofuwa/Azure-ML-Nanodegree--Capstone-project/blob/main/endpoint.py) which takes in the scoring_uri, the primary key and the data. The output was a json file that indicates how our target feature or variable is uspposed to turn out.

Then I deleted the compute instance.

## Screen Recording
https://youtu.be/zuqittNSbxY

## Standout Suggestions
1. I tested the deployed model in the studio by inputing a random observation from our data without the target column and it produced an accurate result which could be indicative that the model is performing well and has a low bias.

<p align="center">
  <img src="https://github.com/Ayoyinka-Sofuwa/Azure-ML-Nanodegree--Capstone-project/blob/main/screenshot/testing%20the%20deployed%20model.png">
</p>

2. I converted my AutoML model to an ONNX format to make it easy to convert from one frame to another and to easily deploy the model
3. I enabled logging in my deployed model and viewed metrics in the studio about the occurences during the operation

<p align="center">
  <img src="https://github.com/Ayoyinka-Sofuwa/Azure-ML-Nanodegree--Capstone-project/blob/main/screenshot/logging%201.png">
</p>

<p align="center">
  <img src="https://github.com/Ayoyinka-Sofuwa/Azure-ML-Nanodegree--Capstone-project/blob/main/screenshot/logging%202.png">
</p>


REFERENCE:
###### https://docs.microsoft.com/
