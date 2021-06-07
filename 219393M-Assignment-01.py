from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy import unique
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from numpy import argmax
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

df= pd.read_csv(r'crx.data', header = None, na_values="?" )
df.head()
df.columns = ['A1','A2','A3','A4','A5','A6','A7','A8', 'A9', 'A10','A11','A12','A13','A14','A15','Class']
df = df.to_csv (r'Assignment01.csv', index=None);
df=pd.read_csv("Assignment01.csv");

print('\n\n data types', df.dtypes);
print('\n\nsome of null', df.isnull().sum());
print('\n\nshape of data', df.shape);
df["A14"].value_counts();
print('\n\ndifferent type in A1 column',df["A1"].value_counts());
df.dropna(inplace=True)
print("\n\nShape of the data after removing the null values" , df.shape);
#label encoding
df["A1"] = df["A1"].astype('category');
df["A1"] = df["A1"].cat.codes

df["A9"] = df["A9"].astype('category');
df["A9"] = df["A9"].cat.codes
df["A10"] = df["A10"].astype('category');
df["A10"] = df["A10"].cat.codes

#categorical data
categorical_cols = ['A4', 'A5', 'A6', 'A7','A13', 'A12']
#import pandas as pd
df = pd.get_dummies(df, columns = categorical_cols)

df["Class"] = df["Class"].map({
    "-": 0,
    "+": 1
}).astype(int);
print('\n\nnew data types', df.dtypes);
print(df.head());
print('Final column namses' , df.columns);
print('after preprocessing the dataset', df.head(10));
print('\n\nafter preprocessing the data types', df.shape);

#using best check point we are going to test the data
def make_submission(actual, prediction, sub_name):
  my_submission = pd.DataFrame({ 'actual': actual, 'Class':prediction})
  my_submission.to_csv('{}.csv'.format(sub_name),index=False)
  print('A submission file has been made')

X = df.loc[:, df.columns != 'Class']

X = np.asarray(X).astype(np.float32);
sc= StandardScaler()
X= sc.fit_transform(X)
y = df['Class']
# encode strings to integer
y = LabelEncoder().fit_transform(y)
n_class = len(unique(y))
n_features = X.shape[1]
kf = KFold(n_splits = 5)
# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
NN_model = Sequential()
# The Input Layer :
NN_model.add(Dense(200, kernel_initializer='normal',input_dim = n_features, activation='relu'))
# The Hidden Layers :
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
# The Output Layer :
NN_model.add(Dense(n_class, activation='softmax'))
# Compile the network :
NN_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
NN_model.summary()
checkpoint_name = 'Weights-classification-{epoch:03d}--{val_loss:.5f}.hdf5'
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]
NN_model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split = 0.25, callbacks=callbacks_list)
print("\nLoad wights file of the best model :\n")
wights_file = 'Weights-classification-031--0.40587.hdf5' # choose the best checkpoint
NN_model.load_weights(wights_file) # load it
# NN_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
NN_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# evaluate on test set
y_pred = NN_model.predict(X_test, batch_size=64, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred_bool))
###here we are saving the output ##########
make_submission(y_test,y_pred_bool, 'Final_ANN_Classification_Prediction');

