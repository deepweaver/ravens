#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 
from PIL import Image, ImageChops


# In[2]:


import numpy as np 
import matplotlib.pyplot as plt 


# In[3]:


img = Image.open("../SPMpadded/a1.png") 
img2 = Image.open("../SPMpadded_lined/a1.png") 
print(type(img))


# In[4]:


img.show()
img2.show()


# In[5]:


npimg = np.array(img) 
npimg2 = np.array(img2)


# In[6]:


print(npimg.shape)


# In[7]:


img.show()


# In[8]:



Limg = img.copy().convert("L")
npLimg = np.array(Limg)
npLimg[npLimg < 128] = 0
npLimg[npLimg >= 128] = 255
i = Image.fromarray(npLimg).convert('1')
i.save("tmp.png") 


# In[9]:


def converter(image,targetpath):
    x = np.array(image.copy().convert("L")) 
    x[x<128] = 0 
    x[x>=128] = 255 
    i = Image.fromarray(x).convert('1') 
    i.save(targetpath)
# converter(img)
import os 
for filename in os.listdir("../SPMpadded_lined/"):
    if filename.endswith("png"):
        im = Image.open("../SPMpadded_lined/"+filename)
        converter(im, "../SPMbinarized_lined/"+filename)
    


# In[10]:


# type(npLimg[0,0])


# In[11]:


# print(npLimg[300:310,400:410])


# In[12]:


# plt.imshow(npLimg)


# In[13]:


# plt.show()


# In[ ]:





# In[ ]:





# In[14]:


mat = cv2.imread('../SPMbinarized/a1.png', cv2.IMREAD_GRAYSCALE) 
print(type(mat))


# In[15]:


print(mat.shape)


# In[16]:


print(type(mat[0,0]))


# In[17]:


im_gray = cv2.imread("../SPMpadded/a1.png", cv2.IMREAD_GRAYSCALE)
(thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# determines the threshold automatically from the image using Otsu's method
cv2.imwrite('./tmp.png', im_bw) 
print(thresh)


# In[18]:


print(im_bw.shape)


# In[19]:


print(im_bw[0,0])
print(type(im_bw[0,0]))


# In[20]:


im_bw_inv = np.zeros(im_bw.shape, dtype=np.uint8) 
im_bw_inv[im_bw == 255] = 0 
im_bw_inv[im_bw == 0] = 255
contours = cv2.findContours(im_bw_inv.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] 
# def findContours(image, mode, method, contours=None, hierarchy=None, offset=None):
# """
# 检测二值图像的轮廓信息
# Argument:
#     image: 待检测图像
#     mode: 轮廓检索模式
#     method: 轮廓近似方法
#     contours: 检测到的轮廓；每个轮廓都存储为点矢量
#     hierarchy: 
#     offset: 轮廓点移动的偏移量
# """
image = cv2.imread("../SPMpadded/a1.png") 
image_RGB = image.copy() 

for c in contours:
    cv2.drawContours(image, [c], -1, (255, 0, 0), 2)
    break
    
    
# display BGR image
plt.subplot(1, 3, 1)
plt.imshow(image_RGB)
plt.axis('off')
plt.title('image_BGR')

# display binary image
plt.subplot(1, 3, 2)
plt.imshow(im_bw, cmap='gray')
plt.axis('off')
plt.title('image_binary')

# display contours
plt.subplot(1, 3, 3)
plt.imshow(image)
plt.axis('off')
plt.title('{} contours'.format(len(contours)))

plt.show()


# In[21]:


cv2.imwrite("./contour.png", image)


# In[22]:


print(np.sum(im_bw / 255))


# In[23]:


print(np.prod(im_bw.shape))


# In[24]:


print(len(contours))


# In[25]:


print(len(contours[0]))


# In[26]:


print(contours[0])


# In[27]:


print(contours[0].shape)


# In[28]:


for i in range(len(contours)):
    print(contours[i].shape)


# In[29]:


with open("../SPM coordinates.txt", "r") as f: 
    for line in f:
        print(line)


# In[39]:


coordinates = [] 
with open("../SPM coordinates.txt", "r") as f: 
    for i,line in enumerate(f):
#         if i % 10 == 0:
            
        coordinates.append([list(map(int,line.split()))])
print(coordinates[0])
print(len(coordinates)) 


# In[40]:


print(coordinates[21]==[[]])


# In[41]:


tmp = ["a", "b", "c", "d", "e"] 
out = "[" 
for i, v in enumerate(tmp):
    for j in range(1,13): 
        out += "'" + v + str(j) + ".png" + "', " 
out += "]"
print(out)


# In[42]:





import os 
data = [] 
dirs = sorted(os.listdir("../SPMpadded/"),)
filenames = ['a1.png', 'a2.png', 'a3.png', 'a4.png', 'a5.png', 'a6.png', 'a7.png', 'a8.png', 'a9.png', 'a10.png', 'a11.png', 'a12.png', 'b1.png', 'b2.png', 'b3.png', 'b4.png', 'b5.png', 'b6.png', 'b7.png', 'b8.png', 'b9.png', 'b10.png', 'b11.png', 'b12.png', 'c1.png', 'c2.png', 'c3.png', 'c4.png', 'c5.png', 'c6.png', 'c7.png', 'c8.png', 'c9.png', 'c10.png', 'c11.png', 'c12.png', 'd1.png', 'd2.png', 'd3.png', 'd4.png', 'd5.png', 'd6.png', 'd7.png', 'd8.png', 'd9.png', 'd10.png', 'd11.png', 'd12.png', 'e1.png', 'e2.png', 'e3.png', 'e4.png', 'e5.png', 'e6.png', 'e7.png', 'e8.png', 'e9.png', 'e10.png', 'e11.png', 'e12.png']
print(dirs)
for filename in filenames:
#     if filename.endswith("png"):
#         im = Image.open("../SPMpadded_lined/"+filename)
#         converter(im, "../SPMbinarized_lined/"+filename)
    im_gray = cv2.imread("../SPMpadded/"+filename, cv2.IMREAD_GRAYSCALE) 
    (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    data.append(im_bw) 




# im_gray = cv2.imread("../SPMpadded/a1.png", cv2.IMREAD_GRAYSCALE)
# (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# # determines the threshold automatically from the image using Otsu's method
# cv2.imwrite('./tmp.png', im_bw) 
# print(thresh)


# In[43]:


print(len(data)) 


# In[44]:


plt.imshow(data[0]) 
plt.show()


# In[45]:


grouped_coor = [] 
group = []
for i,v in enumerate(coordinates):
    if v == [[]]:
        grouped_coor.append(group)
        group = []
        continue
    group.append(v)
print(len(grouped_coor))


# In[46]:


print(len(grouped_coor[0]))
print(len(grouped_coor[23]))
print(len(grouped_coor[24]))
print(len(grouped_coor[-1]))
print(grouped_coor[-2])


# In[47]:



dataset = {"4cells":[], "9cells":[]}

idx = 0
for i in range(24):
    
    dataEntry = []
    for j in range(10):
        x, y, w, h = grouped_coor[i][j][0]
        dataEntry.append(data[i][y:y+h,x:x+w].copy())
    dataset["4cells"].append(dataEntry)
    
    
for i in range(24,60):
    dataEntry = [] 
    for j in range(17):
        x, y, w, h = grouped_coor[i][j][0]
        dataEntry.append(data[i][y:y+h,x:x+w].copy()) 
    dataset["9cells"].append(dataEntry)
    


# In[48]:


plt.imshow(dataset["4cells"][0][0])
plt.show()


# In[49]:


import pickle

with open("data.pkl","wb") as f: 
    pickle.dump(dataset,f) 


# In[50]:


import pickle
with open("data.pkl", 'rb') as f: 
    dataset2 = pickle.load(f) 
    


# In[51]:


print(len(dataset2["4cells"]))


# In[52]:


from utils import visualize 
visualize(dataset2["4cells"][0])


# In[ ]:




