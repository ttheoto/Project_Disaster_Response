import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    INPUT:
    x - string to be cleaned '$*,*'
    
    OUTPUT:
    x - if x is already cleaned and saved as a float
    x_clean - x cleaned and saved as float
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath) 
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, how='left', on=('id'))
    return df

def clean_data(df):
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(";", expand=True)
    # select the first row of the categories dataframe
    row = categories.loc[0]
    # use this row to extract a list of new column names for categories
    category_colnames = list(row.str.split("-", expand = True)[0])
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split("-").str[1]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    # drop the original categories column from `df`
    df = df.drop(columns=['categories'])
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    # remove wrong values from related column
    df.loc[df['related']==2, 'related'] = 1
    # drop duplicates
    df = df.drop_duplicates()
    # remove column with no values
    df = df.drop(columns=['child_alone'])
    return df

def save_data(df, database_filepath):
    # create engine with database path
    engine = create_engine('sqlite:///' + database_filepath)
    # save dataframe to sql database
    df.to_sql('Messages', engine, index=False, if_exists='replace')  

def main():
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