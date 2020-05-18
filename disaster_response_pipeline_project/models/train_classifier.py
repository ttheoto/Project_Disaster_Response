# import libraries
import sys
import pandas as pd
import nltk
import pickle

from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
nltk.download(['punkt','stopwords','wordnet'])

def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('SELECT * FROM Messages', engine)
    X = df['message']
    y = df.drop(columns=['id', 'message', 'original','genre'])
    category_names = list(y.columns)
    return X, y, category_names

def tokenize(text):
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
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {'clf__estimator__n_estimators': [100,200],
                  'clf__estimator__criterion': ['gini', 'entropy']}
    cv = GridSearchCV(pipeline, parameters)
    return cv   

def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    test_pred = pd.DataFrame(y_pred, columns= category_names)
    test_y = Y_test.reset_index(drop=True)
    for col in category_names:
        print(col)
        print (classification_report(test_y[col], test_pred[col]))
    acc = (test_pred == test_y).mean().mean()
    print('Total Accuracy: {:.4f}'.format(acc))

def save_model(model, model_filepath):
    # save the model to disk
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                                                            test_size=0.7,
                                                            random_state=10)
        
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