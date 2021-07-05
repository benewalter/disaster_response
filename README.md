### Table of Contents - Disaster Response Pipeline Project

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

The code should run with no issues using Python version 3.0 or older.

Libraries that were used for the analysis are Pandas, Numpy, Scikit-learn, SQLlchemy, NLTK, RE, Pickle, Json, Plotly, Flask.


## Project Motivation<a name="motivation"></a>

As part of the Udacity Data Scientist Nanodegree, I developed this ETL and ML pipeline that takes messages of disaster responses as inputs and then classifies each message as different categories (multilabel classification).

The ETL and ML pipeline can be used to run a Flask app. The Flask app can be fed disaster messages which are then classified into different categories. 


## File Descriptions <a name="files"></a>

The following files are contained in this repository:
1. process_data.py - ETL pipeline that loads the messages and response categories and preprocesses them so that they are ready for the ML pipeline. The results are saved in a database. 
2. train_classifier.py - Machine Learning pipeline that takes messages and response categories as inputs, tokenizes the messages and outputs a trained and optimized ML model
3. run.py - Python file that creates a flask app
4. Jupyter notebookes - These notebooks were used to create process_data.py and train_classifier.py
5. Pikle files - represent the trained ML models that can be used to classify disaster response messages
6. CSV files - disaster_messages.csv and disaster_categories.csv represent the messages and categories used for training the ML model
7. DisasterResponse.db - database file that contains the cleaned messages and categories resulting from running process_data.py


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

Please note that running the app should only work via a Udacity classroom.
However, process_data.py and train_classifier.py can run without Udacity. You can specify different file paths to test and train the model with different data.

## Results<a name="results"></a>

The above mentioned scripts contain the results of the analysis. They can be used to classify other disaster messages.

![image](https://user-images.githubusercontent.com/62476751/124501862-9fd71900-ddc2-11eb-9469-fd5ec2d5e830.png)
![image](https://user-images.githubusercontent.com/62476751/124501955-ce54f400-ddc2-11eb-8a5c-22ef1f5ff82f.png)


## Licensing, Authors, Acknowledgements<a name="licensing"></a>

I want to give credit to Udacity for the data. Also, I relied on numerous posts on Stack Overflow when stuck with any question. 
I also want to thank the Udacity team for providing this code, the very useful instructions and guidelines for completing this project. 
Otherwise, feel free to use the code here as you would like!


