# Load pickled data
import pickle
import matplotlib.pyplot as plt
import numpy as np
import csv

# TODO: Fill this in based on where you saved the training and testing data

training_file = './traffic-signs-data/train.p'
validation_file = './traffic-signs-data/valid.p'
testing_file = './traffic-signs-data/test.p'
sign_names_file = 'signnames.csv'
with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
with open(sign_names_file, mode='r') as f:
    reader = csv.reader(f)
    sign_names = {rows[0]:rows[1] for rows in reader}

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

# Number of training examples
n_train = X_train.shape[0]
# Number of testing examples.
n_test = X_test.shape[0]
# shape of an traffic sign
image_shape = X_train.shape[1:3]
# unique classes/labels there are in the dataset.
n_classes = len(sign_names)
print(X_train.shape)
print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
fig, axes = plt.subplots(6, 8)
fig.subplots_adjust(hspace=0.3, wspace=0.05)
sign_classes = sign_names.values()
last_label = -1
r = 0
c = 0
for feature, label in zip(list(X_train), list(y_train)):
    try:
        sign = sign_names.pop(str(label))
        print(c, ", ", r, "\t", sign)
        if(c > 7): 
            c = 0
            r += 1
        axes[r,c].imshow(feature)
        axes[r,c].set_title(sign)
        c += 1
    except KeyError:
        pass
plt.show()