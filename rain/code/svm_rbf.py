#!/anaconda/bin/python

""" First attempt to solve the rain problem
Author : Ryan Bahneman
Date : 22 April 2015
Revised : 22 April 2015

""" 
import IPython as ipy
import pandas as pd
import numpy as np
import csv
import os
import sys
import pylab as P
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier





def split_data(infile, outfile, overwrite=False):

    ### Skip splitting if the file exists ###
    if (not overwrite) and os.path.isfile(outfile):
        print "Split file alread exists. Skipping step..."
        return

    ### Load the data ###
    print 'Loading data...'
    csv_file_object = csv.reader(open(infile, 'rb')) 

    header = csv_file_object.next()  

    data=[]                          
    for row in csv_file_object:      
        data.append(row)             
    data_len = len(data)
    

    ### Split the data ###
    print 'Splitting the data...'

    # Get the indices where there are multiple entries
    # Note: this only works if the first row contains more than one sample
    #       and all rows need to be split in the s
    multi_entry_idxs = []
    for i, data_entry in enumerate(data[0]):
        if len(data_entry.split(" ")) > 1:
            multi_entry_idxs.append(i)

    # Build a new set of rows for every old row
    new_data = []
    new_data.append(header)
    for i, raw_samples in enumerate(data):

        # Split the multi entry elements
        for c_idx in multi_entry_idxs:
            raw_samples[c_idx] = raw_samples[c_idx].split(" ")

        # The number of entries in this sample
        sample_count = len(raw_samples[multi_entry_idxs[0]])

        # Create a new sample for each entry in the raw sample
        for j in xrange(sample_count):
            new_samples = []
            # If the element was not multi entry copy the element
            # Else copy the j-th entry of the element
            for k in xrange(len(raw_samples)):
                if not k in multi_entry_idxs:
                    new_samples.append(raw_samples[k])
                else:
                    new_samples.append(raw_samples[k][j])

            # Add the new sample to new dataset
            new_data.append(new_samples)

        # Progress tracker
        if i % 1000 == 0:
            print("Completed row %d of %d" % (i, data_len))

    ### Write the data ###
    print 'Writing the data...'

    f = open(outfile, "w")
    csv_writer = csv.writer(f)
    csv_writer.writerows(new_data)
    f.close()

# Data cleanup
def clean_data(data):
    if type(data) == float:
        result = [data]
    else:
        result = data.split(' ')
    return np.array(result).astype(np.float)
    
        #composite_datatypes = ["DistanceToRadar", "Composite", "HybridScan", "HydrometeorType", "Kdp", "LogWaterVolume", "MassWeightedMean", "MassWeightedSD", "RR1", "RR2", "RR3", "RadarQualityIndex", "Reflectivity", "ReflectivityQC", "RhoHV", "TimeToEnd", "Velocity", "Zdr", "HydrometeorType"]

    return data


### Split The Data ###
raw_training_data = '../data/train_2013.csv'
split_training_data = '../data/train_split.csv'
print 'Splitting training data'
split_data(raw_training_data, split_training_data)

raw_test_data = '../data/test_2014.csv'
split_test_data = '../data/test_split.csv'
print 'Splitting test data'
split_data(raw_test_data, split_test_data)


# TRAIN DATA
print 'Loading training data...'
train_df = pd.read_csv(split_training_data, header=0)


print 'Cleaning the training data...'
# Drop the readings that are unreasonably large
max_valid_reading = 69
train_df = train_df[train_df.Expected <= max_valid_reading]

# Reset the indexing
train_df = train_df.reset_index()

# Set the missing values to 0
train_df.MassWeightedMean = train_df.MassWeightedMean.replace(np.nan, 0)
train_df.MassWeightedSD = train_df.MassWeightedSD.replace(np.nan, 0)

train_df.RR1 = train_df.RR1.replace(["-99900", "-99901", "-99903", "999"], np.nan)
train_df.RR2 = train_df.RR2.replace(["-99900", "-99901", "-99903", "999"], np.nan)
train_df.RR3 = train_df.RR3.replace(["-99900", "-99901", "-99903", "999"], np.nan)

# Create a radar best guess estiamte colum
train_df["RRR"] = train_df.RR1.copy()
r2_but_not_r1 = train_df.RR1.isnull() & train_df.RR2.notnull()
train_df.loc[r2_but_not_r1, "RRR"] = train_df.RR2[r2_but_not_r1]
r3_but_not_r2_or_r1 = (train_df.RR1.isnull() & 
                       train_df.RR2.isnull() &
                       train_df.RR3.notnull() )
train_df.loc[r3_but_not_r2_or_r1, "RRR"] = train_df.RR3[r3_but_not_r2_or_r1]

# Drop the colums that don't have a radar estimate
train_df = train_df[train_df.RRR.notnull()]
# Reset the indexing
train_df = train_df.reset_index()

exptected = train_df.Expected.values
train_ids = train_df.Id.values

# Keeping 'DistanceToRadar', 'RRR', 'MassWeighteMean', 'MassWeightedSD'
train_df = train_df.drop(['TimeToEnd', 'level_0', 'index', 'Id', 'Composite', 'HybridScan', 'HydrometeorType', 'Kdp', 'RR1','RR2', 'RR3', 'Reflectivity', 'ReflectivityQC', 'RhoHV', 'Velocity', 'Zdr', 'LogWaterVolume', 'Expected', 'RadarQualityIndex'], axis=1) 

train_data = train_df.values

print 'Training...'
forest = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
forest = forest.fit( train_data, exptected)

ipy.embed()
sys.exit(0)






print 'Predicting...'
output = forest.predict(test_data).astype(int)





ipy.embed()
sys.exit(0)


ipy.embed()

sys.exit(0)




train_data = clean_data(train_df)


# TEST DATA
test_df = pd.read_csv(split_test_data, header=0)

#Clean and normailze the data (don't normalize the first colum)
#train_data = clean_data(train_df)
#norm_train_data, norm_mins, norm_maxs = normalize(train_data[:,1:])

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
