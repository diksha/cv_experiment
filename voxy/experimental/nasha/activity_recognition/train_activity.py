#
# Copyright 2020-2021 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import preprocessing
import pickle

PATH = "/home/nasha_voxelsafety_com/voxel/experimental/nasha/activity_recognition/"
activity = "lift"
all_files = os.listdir(PATH)
raw_data = []
for filename in all_files:
    if activity in filename and "csv" in filename:
        df = pd.read_csv(os.path.join(PATH, filename), index_col=None,header=None)
        raw_data.append(df)

frame = pd.concat(raw_data, axis=0, ignore_index=True)

# index_names = frame[frame.iloc[:,-1] == f"good_{activity}"].index

# frame.drop(index_names, inplace = True)

X, y = frame.iloc[:, 1:-1], frame.iloc[:, -1]

# pose_classes are [0: "bad lift", 1: "good lift", 2: "random pose"]
label_enc = preprocessing.LabelEncoder()
label_enc.fit(["random", f"bad_{activity}", f"good_{activity}"])
y = label_enc.transform(y)

# split dataset into train and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) 
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Classsification Report:\n",metrics.classification_report(y_test, y_pred))

filename = '/home/nasha_voxelsafety_com/voxel/experimental/nasha/models/'+f'{activity}_classifer_01282022.sav'
pickle.dump(clf, open(filename, 'wb'))

