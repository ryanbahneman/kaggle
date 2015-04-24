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
    # Set the missing values to 0
    data.MassWeightedMean = data.MassWeightedMean.replace(np.nan, 0)
    data.MassWeightedSD = data.MassWeightedSD.replace(np.nan, 0)

    data.RR1 = data.RR1.replace(["-99900", "-99901", "-99903", "999"], np.nan)
    data.RR2 = data.RR2.replace(["-99900", "-99901", "-99903", "999"], np.nan)
    data.RR3 = data.RR3.replace(["-99900", "-99901", "-99903", "999"], np.nan)

    # Create a radar best guess estiamte colum
    data["RRR"] = data.RR1.copy()
    r2_but_not_r1 = data.RR1.isnull() & data.RR2.notnull()
    data.loc[r2_but_not_r1, "RRR"] = data.RR2[r2_but_not_r1]
    r3_but_not_r2_or_r1 = (data.RR1.isnull() & 
                           data.RR2.isnull() &
                           data.RR3.notnull() )
    data.loc[r3_but_not_r2_or_r1, "RRR"] = data.RR3[r3_but_not_r2_or_r1]

    # Drop the colums that don't have a radar estimate
    data = data.loc[data.RRR.notnull()]
    # Reset the indexing
    data = data.reset_index(drop=True)

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
train_df = train_df.loc[train_df.Expected <= max_valid_reading]
# Reset the indexing
train_df = train_df.reset_index(drop=True)

# Clean the data
train_df = clean_data(train_df)

exptected = train_df.Expected.values
it_rained = np.array(map(lambda x: x > 0, exptected))
train_ids = train_df.Id.values


# When it did rain
rain_train_df = train_df.loc[train_df.Expected > 0]
rain_data = rain_train_df.Expected.values

# Calculate the percent of rain
rain_percent = []
rain_samples = len(rain_data)
for i in xrange(70):
    samples_in_bin = map(lambda x: x < (i+1), rain_data)
    rain_percent.append(np.array(samples_in_bin).astype(np.int).sum())
rain_percent = np.array(rain_percent) / (1.0 * rain_samples)

# Keeping 'DistanceToRadar', 'RRR', 'MassWeighteMean', 'MassWeightedSD'
train_df = train_df.drop(['TimeToEnd', 'Id', 'Composite', 'HybridScan', 'HydrometeorType', 'Kdp', 'RR1','RR2', 'RR3', 'Reflectivity', 'ReflectivityQC', 'RhoHV', 'Velocity', 'Zdr', 'LogWaterVolume', 'Expected', 'RadarQualityIndex'], axis=1) 

train_data = train_df.values

print 'Training...'
forest = RandomForestClassifier(n_estimators=10, n_jobs=7)
#forest = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
forest = forest.fit( train_data, it_rained)


# TEST DATA
print 'Loading test data...'
test_df = pd.read_csv(split_test_data, header=0)


print 'Getting unique ids...'
unique_test_ids = test_df.Id.values.astype(np.int)
unique_test_ids = np.unique(unique_test_ids)

test_id_idxs = dict()
for i, test_id in enumerate(unique_test_ids):
    test_id_idxs[test_id] = i

print 'Cleaning test data...'
test_df = clean_data(test_df)
test_ids = test_df.Id.values


# Keeping 'DistanceToRadar', 'RRR', 'MassWeighteMean', 'MassWeightedSD'
test_df = test_df.drop(['TimeToEnd', 'Id', 'Composite', 'HybridScan', 'HydrometeorType', 'Kdp', 'RR1','RR2', 'RR3', 'Reflectivity', 'ReflectivityQC', 'RhoHV', 'Velocity', 'Zdr', 'LogWaterVolume', 'RadarQualityIndex'], axis=1) 

test_data = test_df.values

print 'Predicting...'
output = forest.predict(test_data).astype(int)

print 'Counting predictions...'
prediction_count_for_id = np.zeros(unique_test_ids.shape).astype(np.float)
predictions_at_id = np.zeros(unique_test_ids.shape).astype(np.float)
for i, test_id in enumerate(test_ids): 
    idx = test_id_idxs[test_id]

    prediction_count_for_id[idx] += 1
    predictions_at_id[idx] += output[i]

print 'Averaging predictions...'
prediction_count_for_id = map(lambda x: max(x,1.0), prediction_count_for_id)
average_predictions = predictions_at_id / prediction_count_for_id;

print 'Building prediction output...'
all_predictions = []
for avg_pre in average_predictions:
    if avg_pre < 0.5:
        #assume no rain
        predicion = [1]*70
    else:
        predicion = rain_percent

    all_predictions.append(predicion)

# re insert the id column
for i, test_id in enumerate(unique_test_ids):
    predictions_for_test_id = all_predictions[i]
    all_predictions[i] = [test_id]
    all_predictions[i].extend(predictions_for_test_id)

header = ["Id"]
for i in xrange(70):
    header.append("Predicted%d" % i)

all_predictions.insert(0, header)

prediction_filepath = "../prediction/first.csv"
prediction_dir = os.path.dirname(prediction_filepath)
if not os.path.exists(prediction_dir):
    os.makedirs(prediction_dir)
predictions_file = open(prediction_filepath, "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerows(all_predictions)
predictions_file.close()

print 'Done.'
ipy.embed()
sys.exit(0)




train_data = clean_data(train_df)



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
