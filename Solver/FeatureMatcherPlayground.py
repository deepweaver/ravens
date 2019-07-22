#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 
from PIL import Image 
import matplotlib.pyplot as plt 
import numpy as np 


# In[4]:


from utils import visualize 
import pickle 
with open("./data.pkl", 'rb') as f: 
    dataset = pickle.load(f)


# In[5]:


data_ab = dataset["4cells"] # list of list of numpy matrices 


# In[6]:


data_cde = dataset["9cells"] 


# In[7]:


visualize(data_ab[0])


# In[8]:


# Brute-Force Matching with ORB Descriptors 
img1 = data_ab[0][0]
img2 = data_ab[0][9]
# Initiate ORB detector
orb = cv2.ORB_create()
# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)


# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1,des2)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
print(sum([matches[i].distance for i in range(10)]))
# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()


# In[23]:


print(type(kp1))
print(type(des1))


# In[25]:


print(len(kp1)) 
print(type(kp1[0]))


# In[26]:


print(kp1)


# In[27]:


print(des1.shape)


# In[28]:


plt.imshow(des1), plt.show()


# In[29]:


print(img1.shape)


# In[30]:


print(type(matches))
print(len(matches))


# In[31]:


print(type(matches[0]))


# In[32]:


print(matches[0])


# In[45]:


print(matches[10].distance)


# In[9]:


img1 = data_ab[0][0]
img2 = data_ab[0][9] 
# Initiate SIFT detector 
sift = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()


# In[10]:


sift = cv2.xfeatures2d.SIFT_create()


# In[ ]:




