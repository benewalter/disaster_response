# Importing relevant packages
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    INPUT - messages_filepath - path to file that contains messages
            categories_filepath - path to file that contains categories
    OUTPUT - 
            df - data frame that contains merged messages and categories
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories, how = 'inner', on = 'id')
    
    return df


def clean_data(df):
    '''
    INPUT - df - data frame that contains merged messages and categories
    OUTPUT - 
            df - cleaned data frame - removed duplicates, each category in one column, appropriate column names
    '''
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand = True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories
    category_names = row.str.split('-', expand = False).str[0]
    # Derived column names are saved as a list
    category_colnames = category_names.values.tolist()
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.strip().str[-1]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    # Ensure that no values > 1 exist
    categories = categories.applymap(lambda x:1 if x>1 else x) 
    
    # drop the original categories column from `df`
    df.drop('categories', axis= 1,inplace = True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    
    # drop duplicates
    df.drop_duplicates(inplace = True)
    
    return df


def save_data(df, database_filename):
    '''
    INPUT - df - cleaned data frame with messages and categories
            database_filename - intended name for the database to be created
    OUTPUT - 
            (saved) database containing the cleaned messages and categories
    '''
    # Create SQL engine
    engine = create_engine('sqlite:///' + str(database_filename))
    # Save data frame as table to database
    df.to_sql('Messages_and_Categories', engine,if_exists = 'replace', index=False) 


def main():
    '''
    Main function that contains the functions to load, clean and save the disaster response data
    
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
