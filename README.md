# Disaster Response Pipeline Project

### Project Overview
In this project we analyze disaster data to build a machine learning model using scikit-learn for an API that classifies disaster messages. The data set  is made up of real messages that were sent during disaster events. We create a machine learning pipeline to categorize these events so that the messages can be sent to an appropriate disaster relief agency.

This projects include a web app where users can input new messages and get classification results in several categories. The web app also display some visualizations of the data.

### Relevant Files:
* [process_data](data/process_data.py) ETL Pipeline that:
  * Loads the messages and categories datasets
  * Merges the two datasets
  * Cleans the data
  * Stores it in a SQLite database

* [train_classifier](models/train_classifier.py) ML Pipeline that:
  * Loads data from the SQLite database
  * Splits the dataset into training and test sets
  * Builds a text processing and machine learning pipeline
  * Trains and tunes a model using GridSearchCV
  * Outputs results on the test set
  * Exports the final model as a pickle file

* [run](app/run.py): Start the server for the Flask Web App

### Instructions
1. Run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python app/run.py`

3. Go to http://0.0.0.0:3001/