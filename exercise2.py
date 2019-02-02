import numpy as np 
from sklearn import preprocessing

input_data=np.array([[1,4,2,3],[5,3,4,2],[6,4,8,9]])
print (input_data)

print(input_data.mean(axis=0))
data_standardised = preprocessing.scale(input_data)

print(data_standardised)

print("Mean : ",data_standardised.mean(axis=0))
print("Standard Deviation : ",data_standardised.std(axis=0))

data_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
data_scaled = data_scaler.fit_transform(input_data)
print("MinMax Scale :", data_scaled)