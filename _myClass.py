import numpy as np
import pandas as pd

class LogisticRegression:

  def __init__(self):
    self.dict_crop={}#class variable
  
  def sigmoid(self,z):
    #z=np.int256(z)
    a= 1 / (1 + np.exp(-z))
    return a

  def prediction(self,X):
    crop_enigma={'rice': 1, 'maize': 2, 'chickpea': 3, 'kidneybeans': 4, 'pigeonpeas': 5, 'mothbeans': 6, 'mungbean': 7, 'blackgram': 8, 'lentil': 9, 'pomegranate': 10, 'banana': 11, 'mango': 12, 'grapes': 13, 'watermelon': 14, 'muskmelon': 15, 'apple': 16, 'orange': 17, 'papaya': 18, 'coconut': 19, 'cotton': 20, 'jute': 21, 'coffee': 22}
    #inv_dict = {v: k for k, v in crop_enigma.items()}
    #print(inv_dict)
    crops=self.dict_crop.keys()
    m,n=np.shape(X)
    max_crop=[]
    max_value=[0]*m
    for crop in crops:

     for y in self.sigmoid(np.dot(X.T,self.dict_crop[crop][0])+self.dict_crop[crop][1]):
      if  y > max_value[-1]:
        max_crop.append(crop)
        max_value.append(y)


    return (crop_enigma[max_crop[-1]])
