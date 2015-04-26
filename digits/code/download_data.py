#!/anaconda/bin/python

import os
import curl
import webbrowser
import time

data_dir = "../data"

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

train_data_file = "train.csv"
test_data_file = "test.csv"
data_url = "http://www.kaggle.com/c/digit-recognizer/data"

train_data_path = os.path.join(data_dir, train_data_file)
test_data_path = os.path.join(data_dir, test_data_file)

if (not os.path.isfile(train_data_path)) or (not os.path.isfile(test_data_path)):
    print ("Opening data webpage...")
    print ("Download %s and %s into %s" \
            % (train_data_file, test_data_file, data_dir))
    time.sleep(3)
    webbrowser.open(data_url,new=2)
else:
    print ("Data files exist in %s. Skipping download." % data_dir)

