import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    INPUT: messages_filepath - the filepath of the message.csv file
           categories_filepath - the filepath of the the category.csv file
    OUTPUT: df - dataframe with the message.csv file and the category.csv file merged together
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories, how='outer', on='id', indicator=False)
    return df

def clean_data(df):
    '''
    INPUT: df - dataframe with the message.csv file and the category.csv file merged together
    OUTPUT: df - cleaned df containing category volumns 
    '''
    # create a dataframe of the category columns called categories_df
    categories_df = df['categories'].str.split(pat=';', expand=True)
    
    # rename the columns in categories_df
    row = categories_df.loc[0]
    category_colnames = row.apply(lambda x: x[:-2]).values
    categories_df.columns = category_colnames
    
    # convert category values to just numbers 0 or 1 in categories_df
    categories_df = categories_df.apply(lambda x: x.str.split('-').str.get(1).astype('int64'))
    
    # replace categories column in df with new category columns and concat it with categories_df
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories_df], axis=1)
    
    # drop duplicates in df_clean
    df = df.drop_duplicates()
        
    return df
    
def save_data(df, database_filename):
    '''
    INPUT: df - cleaned df from def clean_data
           database_filename -name of the database file
    OUTPUT: None 
    '''    
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df.to_sql(database_filename, engine, index=False, if_exists='replace')  


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