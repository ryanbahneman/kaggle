#!/anaconda/bin/python

""" Writing my first randomforest code.
Author : Ryan Bahneman
Date : 21 April 2015
Revised : 21 April 2015

""" 
import IPython as ipy
import pandas as pd
import numpy as np
import csv as csv
import os as os
import pylab as P
from sklearn.ensemble import RandomForestClassifier


# Data cleanup
#   Convert all strings to integer classifiers.
#   Fill in the missing values of the data and make it complete.
def clean_data(data):
    # female = 0, Male = 1
    data['Gender'] = data['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    # Embarked from 'C', 'Q', 'S'
    for port in ['C', 'Q', 'S']:
        data['Embarked_%s'%port] = data['Embarked'].map(lambda x: x == port).astype(int)

    # All missing Embarked -> just make them embark from most common place
    if len(data.Embarked[ data.Embarked.isnull() ]) > 0:
        most_common_port = data.Embarked.dropna().mode().values[0]
        data.loc[data.Embarked.isnull(), 'Embarked_%s' % most_common_port] = 1

    # All the ages with no data -> make the median of all Ages in that ticket class
    ticket_classes = np.unique(data.Pclass.values)
    median_ages = dict()
    for ticket_class in ticket_classes:
        median_for_class = data[data.Pclass == ticket_class].dropna().Age.median()
        median_ages[ticket_class] = median_for_class

    for ticket_class in ticket_classes:
        if len(data.Age[ data.Age.isnull() & (data.Pclass == ticket_class)]) > 0:
            data.loc[ (data.Age.isnull() & (data.Pclass == ticket_class)), 'Age'] \
                    = median_ages[ticket_class]

    # All the missing Fares -> assume median of their respective class
    if len(data.Fare[ data.Fare.isnull() ]) > 0:
        median_fare = np.zeros(3)
        for f in range(0,3): # loop 0 to 2
            median_fare[f] = data[ data.Pclass == f+1 ]['Fare'].dropna().median()
        for f in range(0,3): # loop 0 to 2
            data.loc[ (data.Fare.isnull()) & (data.Pclass == f+1 ), 'Fare'] = median_fare[f]

    # Add colums for titles 
    data['Clergy'] = data.Name.map(lambda x: 'Rev.' in x).astype(int)
    data['Military'] = data.Name.map(lambda x: 'Col.' in x or 'Major' in x).astype(int)
    data['Nobility'] = data.Name.map(lambda x: 'Count' in x).astype(int)
    data['Mr.'] = data.Name.map(lambda x: 'Mr.' in x).astype(int)
    data['Mrs.'] = data.Name.map(lambda x: 'Mrs.' in x).astype(int)
    data['Miss'] = data.Name.map(lambda x: 'Miss' in x).astype(int)
    data['Master'] = data.Name.map(lambda x: 'Master' in x).astype(int)

    # Remove the non-numeric colums
    data = data.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId', 'Embarked'], axis=1) 

    return data.values

# TRAIN DATA
train_df = pd.read_csv('../data/train.csv', header=0)        # Load the train file into a dataframe

train_data = clean_data(train_df)


# TEST DATA
test_df = pd.read_csv('../data/test.csv', header=0)        # Load the test file into a dataframe

# Collect the test data's PassengerIds before dropping it
ids = test_df['PassengerId'].values

test_data = clean_data(test_df)


print 'Training...'
forest = RandomForestClassifier(n_estimators=10000)
forest = forest.fit( train_data[0::,1::], train_data[0::,0] )


print 'Predicting...'
output = forest.predict(test_data).astype(int)


prediction_filepath = "../prediction/myfirstforest.csv"
prediction_dir = os.path.dirname(prediction_filepath)
if not os.path.exists(prediction_dir):
    os.makedirs(prediction_dir)
predictions_file = open(prediction_filepath, "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'
