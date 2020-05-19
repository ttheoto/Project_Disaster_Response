# Project_Disaster_Response
This project consists of a disaster response web-app. It is part of the Udacity's Data Science Nanodegree. 

### Table of Contents

1. [Project Motivation](#motivation)
2. [Installation](#installation)
3. [File Descriptions](#files)
4. [Instructions](#instructions)
5. [Results](#results)
6. [Licensing, Authors, and Acknowledgements](#licensing)

## Project Motivation<a name="motivation"></a>

TBD

## Installation <a name="installation"></a>

The code should run using any Python versions 3.*.. The following libraries are required:
* [plotly](https://plotly.com/)
* [flask](https://flask.palletsprojects.com/en/1.1.x/)
* [pandas](https://pandas.pydata.org/)
* [nltk](nltk.org)
* [sklearn](https://scikit-learn.org/stable/)
* [sqlalchemy](https://www.sqlalchemy.org/)

If running the web app (run.py file), a virtual environment is recommended. Please follow [these straightfoward instructions](https://pythonforundergradengineers.com/new-virtual-environment-with-conda.html) if using the Anaconda Package. Python's [official documentation](https://docs.python.org/3/tutorial/venv.html) can be quite helpful too.


## File Descriptions <a name="files"></a>

process_data.py - ETL Pipeline
train_classifier.py - NPL Pipeline
run.py - Flask Web App

## Instructions <a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app. Use a virtual enviroment.
    `python run.py`

3. Go to http://localhost:3001/ 

## Results<a name="results"></a>
https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568

The exploration of the dataset revealed its imbalance (change to chart with percentages). More than half of the categories have comparatively small (<5% of the messages) representation in the dataset. One category (child_alone) did not have any example sentence.
So, even though the first Machine Learning Model had a good accuracy, its results were rather "naive". It could not assign simple sentences (e.g."I need water!") to the right category.   

![Percentages](message_dist.png)

In order 

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

TBD

