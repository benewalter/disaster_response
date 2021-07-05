import sys

# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.tree import DecisionTreeClassifier


def load_data(database_filepath):
    '''
    INPUT - database_filepath - filepath of the database
    OUTPUT - X - messages df
           - Y - labels df
           - category_names - list of the categories/labels
    '''
    
    # load data from database
    engine = create_engine('sqlite:///' + str(database_filepath))
    df = pd.read_sql_table('Messages_and_Categories',
        con=engine)
    
    # Create data frame with messages and data frame with labels
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    Y = Y.astype(int)
    
    # Get category names
    category_names = Y.columns
    
    #print(category_names)
    #print(X.head())
    #print(Y.head())
    
    return X,Y, category_names


def tokenize(text):
    '''
    INPUT - text - disaster response messages
    OUTPUT - normalized, tokenized, and lemmed disaster response messages
    '''
    
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Tokenize text
    words = word_tokenize(text)
    # Remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    # Reduce words to their root form
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    # Lemmatize verbs by specifying pos
    words = [WordNetLemmatizer().lemmatize(w, pos='v') for w in lemmed]
    
    return words


def build_model():
    '''
    INPUT - 
    OUTPUT - GridSearchCV object
    '''
    
    # ML pipeline
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('muc', MultiOutputClassifier(DecisionTreeClassifier(random_state=0)))
    ])
    
    # Grid of parameters
    parameters = {
    'muc__estimator__min_samples_leaf': [50,100],
    'muc__estimator__max_depth': [2,4]
    }   
     # Grid search object
    cv = GridSearchCV(pipeline, parameters, cv=2, n_jobs = -1)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT - model - trained and optimized ML model
          - X_test - messages from test set
          - Y_test - labels from test set
          - category_names - labels
    OUTPUT - classification report of test set
    '''
    
    # Apply model to test set
    Y_pred = model.predict(X_test)
    # Convert results into dataframe
    Y_pred_df = pd.DataFrame(Y_pred,columns=Y_test.columns)
    
    # Evaluate performance
    for column in Y_test.columns:
        print('Model Performance - Category: ' + str(column))
        print(classification_report(Y_test[column], Y_pred_df[column]))
    


def save_model(model, model_filepath):
    '''
    INPUT - model - trained and optimized ML model
          - model_filepath - name and path of the (to be) saved ML model file
    OUTPUT - saved model as picke file
    '''
    
    # save the model to disk
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()