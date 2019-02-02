import numpy as np 
from sklearn import preprocessing

input_data=np.array([[1,2],[3,4],[5,6]])
print (input_data) 
# printing sample array

print(input_data.mean(axis=0))
data_standardised = preprocessing.scale(input_data)
print(data_standardised)
# Standardised array

print("Mean : ",data_standardised.mean(axis=0))
print("Standard Deviation : ",data_standardised.std(axis=0))
# Standardised array mean and standard deviation

data_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
data_scaled = data_scaler.fit_transform(input_data)
print("MinMax Scale :", data_scaled)
# Standardised array using MinMaxScaler in range 0-1

data_normalised_l1 = preprocessing.normalize(input_data, norm='l1')
print("L1 normalised data :", data_normalised_l1)

data_normalised_l2 = preprocessing.normalize(input_data, norm='l2')
print("L2 normalised data :", data_normalised_l2)


