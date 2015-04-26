#!/usr/bin/env python

import pandas as pd
import IPython as ipy
import PIL

np.set_printoptions(threshold='nan')

training_data_path = "../data/train.csv"

train_df = pd.read_csv(training_data_path, header=0)

images_data = train_df.drop(["label"], axis=1).values

ipy.embed()


