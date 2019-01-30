import sys
from sqlalchemy import create_engine
import pandas as pd

import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import pickle


def load_data(database_filepath):
    """
    Loads and prepare the data splitting by features and labels


    Parameters:
    database_filepath (str): Path pointing to the database

    Returns:
    Pandas.DataFrame: DataFrame containing the features
    Pandas.DataFrame: DataFrame containing the labels
    Pandas.DataFrame: DataFrame containing the labels names
    """

    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table(database_filepath, engine)
    X = df.message
    y = df.drop(['message', 'original', 'id', 'genre'], axis=1)
    return X, y, y.columns


def tokenize(text):
    """
    Tokenize the specified text.

    Parameters:
    text (str): The target sentence

    Returns:
    str[]: the tokens
    """

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []

    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Builds a classifier optimizing hyperparameters by using pipelines and relying on a GridSearchCV model

    Returns:
    GridSearchCV: the model
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        # 'vect__ngram_range': ((1, 1), (1, 2)),
        # 'vect__max_df': (0.5, 0.75, 1.0),
        # 'vect__max_features': (None, 5000),
        # 'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [10, 50],
        'clf__estimator__min_samples_split': [2, 3, 4]
    }

    return GridSearchCV(pipeline, param_grid=parameters, verbose=3)


def evaluate_model(model, X_test, y_test, category_names):
    """
    Display f1 score, precision and recall for each category of the dataset

    Parameters:
    model (GridSearchCV): The target model
    X_test (float[]): Test features
    y_test (float[]): Test labels
    category_names (str[]): The names of the categories
    """

    predictions = model.predict(X_test)
    predictions_t = predictions.T

    trues_t = y_test.values.T

    for index, prediction in enumerate(predictions_t):
        print(category_names[index])
        print(classification_report(trues_t[index], prediction))


def save_model(model, model_filepath):
    """
    Persist the model by serialising it in a pickle file

    Parameters:
    model (GridSearchCV): The target model
    model_filepath (str): The path to save the model
    """

    pickle.dump(model, open(model_filepath, 'wb'))


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
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
