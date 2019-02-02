import numpy as np 
from sklearn import preprocessing

input_classes=['suzuki','ford','suzuki','toyota','ford','bmw']
label_encoder=preprocessing.LabelEncoder()
label_encoder.fit(input_classes)
for i,item in enumerate (label_encoder.classes_):
        print("item -->", i)

encoded_labels = label_encoder.transform (input_classes)
#Transforming labels
print("Labels =", input_classes)
print ("Encoded labels =", list (encoded_labels))

decoded_labels = label_encoder.inverse_transform (encoded_labels) 
# Inverse tranforming labels

print ("Encoded labels =", encoded_labels)
print ("Decoded labels =", list(decoded_labels))

