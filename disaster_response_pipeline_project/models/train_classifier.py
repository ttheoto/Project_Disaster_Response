'''
NLP Pipeline - train_classifier.py
'''
# import libraries
import sys
import pandas as pd
import nltk
import pickle
import time
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report


# download nltk packages
nltk.download(['punkt','stopwords','wordnet'])

def load_data(database_filepath):
    '''
    INPUT:
    database_filepath - str, path to the database to be loaded 
    
    OUTPUT:
    X - pandas dataframe, features of the ML model
    y - pandas dataframe, target of the ML model
    category_names - list, name of each classification category
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('SELECT * FROM Messages', engine)
    X = df['message']
    y = df.drop(columns=['id', 'message', 'original','genre'])
    category_names = list(y.columns)
    return X, y, category_names

def tokenize(text):
    '''
    INPUT:
    text - str to be normalized, tokenized, have its stopwords removed, and
            be lemmatized
    
    OUTPUT:
    tokens - list with processed tokens
    '''
    # normalize and tokenize
    tokens = word_tokenize(text.lower())
    # remove stop words
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    # lemmatize nouns
    tokens = [WordNetLemmatizer().lemmatize(w) for w in tokens]
    # lemmatize verbs
    tokens = [WordNetLemmatizer().lemmatize(w, pos = 'v') for w in tokens]
    # remove trailing space
    tokens = list(map(str.strip, tokens))
    return tokens

def build_model():
    '''
    OUTPUT:
    cv - model with NLP pipeline, whose parameters were found through Grid
        Search
    '''
    # parameters defined in order to reduce computational effort
    # previous GridSearch runs in notebook were needed
    sgd = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultiOutputClassifier(SGDClassifier(loss='hinge', 
                                                                alpha=1e-3, 
                                                                random_state=42, 
                                                                max_iter=5, 
                                                                tol=None)))])
    # grid search for penalty
    parameters = {'clf__estimator__penalty': ['l1', 'l2']}
    # build model
    cv = GridSearchCV(sgd, parameters)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT:
    model - model to be evaluated
    X_test - array containing test features
    Y_test - array containing test targets
    category_names - list, name of each classification category
    
    OUTPUT:
    prints out the classification report for each category
    '''
    y_pred = model.predict(X_test)
    # change format of y_pred to return column names
    test_pred = pd.DataFrame(y_pred, columns= category_names)
    # reset index of y_test to allow comparison
    test_y = Y_test.reset_index(drop=True)
    for col in category_names:
        print(col)
        print (classification_report(test_y[col], test_pred[col], zero_division=0))
    acc = (test_pred == test_y).mean().mean()
    print('Total Accuracy: {:.4f}'.format(acc))

def save_model(model, model_filepath):
    '''
    INPUT:
    model - model to be pickled
    model_filepath - str, path where model should be saved
    
    OUTPUT:
    saved ML model in .pkl format
    '''
    # save the model to disk
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,
                                                            random_state=10)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        start = time.time()
        model.fit(X_train, Y_train)
        end = time.time()
        print(end - start)
        
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