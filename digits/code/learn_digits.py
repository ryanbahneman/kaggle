#!/usr/bin/env python

import pandas as pd
import numpy as np
import IPython as ipy
from PIL import Image


training_data_path = "../data/train.csv"

train_df = pd.read_csv(training_data_path, header=0)

images_data = train_df.drop(["label"], axis=1).values


# Plot an image
for i in xrange(10):
    a = images_data[i].reshape((28,28)).astype(np.uint8)
    img = Image.fromarray(a, mode="L")
    img.show()

ipy.embed()


