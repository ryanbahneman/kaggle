#!/anaconda/bin/python

""" First attempt to solve the rain problem
Author : Ryan Bahneman
Date : 22 April 2015
Revised : 22 April 2015

""" 
import IPython as ipy
import pandas as pd
import numpy as np
import csv as csv
import os as os
import pylab as P
from sklearn import svm



def extract_composite_data(data):
    if type(data) == float:
        result = [data]
        print(data)
    else:
        result = data.split(' ')
    return np.array(result).astype(np.float)

# Data cleanup
#   Convert all strings to integer classifiers.
#   Fill in the missing values of the data and make it complete.
def clean_data(data):

    new_data = pd.DataFrame()

    for i, raw_sample in data.iterrows():

        radar_dists = extract_composite_data(raw_sample.DistanceToRadar)
        max_reflectivities = extract_composite_data(raw_sample.Composite)
        hybrid_scans = extract_composite_data(raw_sample.HybridScan)
        hydrometeor_types = extract_composite_data(raw_sample.HydrometeorType)
        kdps = extract_composite_data(raw_sample.Kdp)
        log_water_volumes = extract_composite_data(raw_sample.LogWaterVolume)
        mass_weighted_means = extract_composite_data(raw_sample.MassWeightedMean)
        mass_weighted_sd = extract_composite_data(raw_sample.MassWeightedSD)
        rr1s = extract_composite_data(raw_sample.RR1)
        rr2s = extract_composite_data(raw_sample.RR2)
        rr3s = extract_composite_data(raw_sample.RR3)
        radar_quality_indices = extract_composite_data(raw_sample.RadarQualityIndex)
        reflectivities = extract_composite_data(raw_sample.Reflectivity)
        reflectivity_qcs = extract_composite_data(raw_sample.ReflectivityQC)
        rho_hvs = extract_composite_data(raw_sample.RhoHV)
        time_to_ends = extract_composite_data(raw_sample.TimeToEnd)
        velocities = extract_composite_data(raw_sample.Velocity)
        zdrs = extract_composite_data(raw_sample.Zdr)

        raw_sample = raw_sample.drop(['HydrometeorType']) 
        for j in xrange(len(radar_dists)):
            new_raw_sample = raw_sample.copy()

            new_raw_sample.Composite = max_reflectivities[j]
            new_raw_sample.DistanceToRadar = radar_dists[j]
            new_raw_sample.RR1 = rr1s[j]
            new_raw_sample.RR2 = rr2s[j]
            new_raw_sample.RR3 = rr3s[j]
            new_raw_sample.Kdp = kdps[j]
            new_raw_sample.HybridScan = hybrid_scans[j]
            new_raw_sample.RadarQualityIndex = radar_quality_indices[j]
            new_raw_sample.Reflectivity = reflectivities[j]
            new_raw_sample.ReflectivityQC = reflectivity_qcs[j]
            new_raw_sample.LogWaterVolume = log_water_volumes[j]
            new_raw_sample.MassWeightedMean = mass_weighted_means[j]
            new_raw_sample.MassWeightedSD = mass_weighted_sd[j]
            new_raw_sample.Zdr = zdrs[j]
            new_raw_sample.RhoHV = rho_hvs[j]
            new_raw_sample.Velocity = velocities[j]
            new_raw_sample.TimeToEnd = time_to_ends[j]


            new_data = new_data.append(new_raw_sample)


        if i == 10:
            break

    ipy.embed()



    # Remove the non-numeric colums
    #data = data.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId', 'Embarked'], axis=1) 

    return data
"""
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

    """


# TRAIN DATA
print 'Loading training data...'
train_df = pd.read_csv('../data/train_2013.csv', header=0) 

print 'Cleaning the training data...'
# Drop the readings that are unreasonably large
max_valid_reading = 69
train_df = train_df[train_df.Expected <= max_valid_reading]

train_data = clean_data(train_df)



#Clean and normailze the data (don't normalize the first colum)
#train_data = clean_data(train_df)
#norm_train_data, norm_mins, norm_maxs = normalize(train_data[:,1:])

# TEST DATA
#test_df = pd.read_csv('../data/test.csv', header=0)

# Collect the test data's PassengerIds before dropping it
#ids = test_df['PassengerId'].values

# Clean and normalize the data
#test_data = clean_data(test_df)
#norm_test_data = normalize(test_data, norm_mins, norm_maxs)[0]

print 'Training...'
#classifier = svm.SVC(kernel='rbf')
#classifier = classifier.fit( norm_train_data, train_data[0::,0] )

print 'Predicting...'
#output = classifier.predict(norm_test_data).astype(int)


#prediction_filepath = "../prediction/svm_rbf.csv"
#prediction_dir = os.path.dirname(prediction_filepath)
#if not os.path.exists(prediction_dir):
#    os.makedirs(prediction_dir)
#predictions_file = open(prediction_filepath, "wb")
#open_file_object = csv.writer(predictions_file)
#open_file_object.writerow(["PassengerId","Survived"])
#open_file_object.writerows(zip(ids, output))
#predictions_file.close()
print 'Done.'
