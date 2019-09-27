import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

nltk.download('punkt')
nltk.download(['punkt', 'wordnet', 'stopwords'])


def load_data(database_filepath):
    '''
    INPUT: the filepath of the clean dataset
    OUTPUT: X - predicting variables for the model
            Y - response variables for the model, with values other than 0 or 1 replace by 1
            category_names - a list of the category names
    '''
    
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df = pd.read_sql_table(database_filepath, con=engine)
    
    # set X and Y 
    X = df['message'].values
    Y = df.drop(['id', 'message', 'original', 'genre' ], axis=1)
    
    category_names = Y.columns
    
    Y = Y.values
    
    # replace non-zero and non_one values in Y with 1, because it's binary
    for i, col in enumerate(Y):
        for j, row in enumerate(col):
            if col[j] not in [0, 1]:
                Y[i][j] = 1
          
    return X, Y, category_names


def tokenize(text):
    '''
    INPUT: text - text data to be tokenized
    OUTPUT: clean_tokens - a list of clean tokens to be used in CountVectorizer for building the model
    '''
    
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    tokens = word_tokenize(text)
    
    clean_tokens = [lemmatizer.lemmatize(word).strip().lower() for word in tokens if word not in stop_words]
    return clean_tokens

def build_model():
    '''
    INPUT: None
    OUTPUT: model - a model using RandomForestClassifier as the estimator for the pipeline and following grid search 
    '''
    # initialize the vectorizer, transformer, estimator, and classifier for building the pipeline    
    vect = CountVectorizer(tokenizer=tokenize)
    tfidf = TfidfTransformer()
    forest = RandomForestClassifier(random_state=1)
    clf = MultiOutputClassifier(forest, n_jobs=1)

    # build the pipeline
    pipeline = Pipeline([('vect', vect), ('tfidf', tfidf), ('clf', clf)])
    
    # use grid search to improve the model
 
    parameters = {

    'clf__estimator__n_estimators': [50, 100, 150],

    'clf__estimator__min_samples_split': [2, 3, 4]

     }

    model = GridSearchCV(pipeline, param_grid=parameters, n_jobs=4, verbose=2)
 
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT: model - the model built in def build_model
           X_test, Y_test - test data from using train_test_split(X, Y, test_size=0.2)
           category_names - category names in Y_test
    OUTPUT: result - results showing the precision, recall, f1-score, support for the model
    '''
    Y_pred = model.predict(X_test)
    
    result = classification_report(Y_pred, Y_test)
    
    return result


def save_model(model, model_filepath):
    '''
    INPUT: model - the model built in def build_model
           model_filepath - the filepath to save the model
    OUTPUT: None
    '''
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