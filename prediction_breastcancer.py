import numpy as np
import pickle

# loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

#input data and transform into numpy array
in_data= np.asarray(tuple(map(float,input().rstrip().split(','))))

#reshape and scale the input array
in_data_re = in_data.reshape(1,-1)

#print the predicted output for input array
print("Breast Cancer Detected (Malignant)" if loaded_model.predict(in_data_re) else "No Breast Cancer Detected (Benign)")
