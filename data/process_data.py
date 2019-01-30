import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load and merge the specified datasets into a DataFrame

    Parameters:
    messages_filepath (str): Path pointing to the messages dataset
    categories_filepath (str): Path pointing to the categories dataset

    Returns:
    Pandas.DataFrame: The merged DataFrame
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return messages.merge(categories, on='id')


def clean_data(df):
    """
    Cleans the specified dataset

    Parameters:
    df (Pandas.DataFrame): The DataFrame to clean

    Returns:
    Pandas.DataFrame: The cleaned DataFrame
    """

    categories = df.categories.str.split(";", expand=True)

    row = categories.iloc[0]
    categories.columns = [value.split('-')[0] for value in row.values]

    for column in categories:
        categories[column] = [int(value.split('-')[1]) for value in categories[column].values]

    df = pd.concat([df.drop('categories', axis=1), categories], axis=1)

    return df.drop_duplicates()


def save_data(df, database_filename):
    """
    Persist DataFrame as an relational database

    Parameters:
    df (Pandas.DataFrame): The DataFrame to persist
    database_filename (str): The database name
    """

    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql(database_filename, engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'.format(messages_filepath, categories_filepath))

        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')

        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))

        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
