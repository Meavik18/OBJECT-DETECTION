#!/usr/bin/env python
# coding: utf-8

# In[114]:


get_ipython().system('pip3 install opencv-python')


# In[115]:


import os
os.sys.path


# # IMPORT LIBRARIES , MODEL AND FILES

# In[1]:


import cv2
import matplotlib.pyplot as plt


# In[2]:


config_file='sssmobilenet-objectdetecting'
frozen_model='frozen_inference_graph.pb'


# In[3]:


model = cv2.dnn_DetectionModel(frozen_model,config_file)


# In[4]:


classLabels=[]
file_name='object.txt'
with open(file_name,'rt') as fpt:
    classLabels=fpt.read().rstrip('\n').split('\n')


# In[5]:


print(classLabels)


# In[6]:


print(len(classLabels))


# In[7]:


model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)


# # IMAGE PART

# In[8]:


img=cv2.imread('eyewear.jpg')


# In[9]:


plt.imshow(img)


# In[10]:


plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


# In[11]:


ClassIndex, confidece, bbox=model.detect(img,confThreshold=0.5)
print(ClassIndex)


# In[12]:


font_scale =3
font =cv2.FONT_HERSHEY_PLAIN
for ClassInd, conf ,boxes in zip(ClassIndex.flatten(), confidece.flatten(), bbox):
    cv2.rectangle(img,boxes,(255,0,0),2)
    cv2.putText(img,classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40),font,fontScale=font_scale,color=(0,255,0),thickness=3)


# In[13]:


plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


# In[14]:


img=cv2.imread('laptop.jpeg')


# In[15]:


plt.imshow(img)


# In[16]:


plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


# In[17]:


ClassIndex, confidece, bbox=model.detect(img,confThreshold=0.5)


# In[18]:


print(ClassIndex)


# In[19]:


font_scale =3
font =cv2.FONT_HERSHEY_PLAIN
for ClassInd, conf ,boxes in zip(ClassIndex.flatten(), confidece.flatten(), bbox):
    cv2.rectangle(img,boxes,(255,0,0),2)
    cv2.putText(img,classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40),font,fontScale=font_scale,color=(0,255,0),thickness=3)


# In[20]:


plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


# In[21]:


img=cv2.imread('VKCAR.jpeg')


# In[22]:


plt.imshow(img)


# In[23]:


plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


# In[24]:


ClassIndex, confidece, bbox=model.detect(img,confThreshold=0.5)


# In[25]:


print(ClassIndex)


# In[26]:


font_scale =3
font =cv2.FONT_HERSHEY_PLAIN
for ClassInd, conf ,boxes in zip(ClassIndex.flatten(), confidece.flatten(), bbox):
    cv2.rectangle(img,boxes,(255,0,0),2)
    cv2.putText(img,classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40),font,fontScale=font_scale,color=(0,255,0),thickness=3)


# In[27]:


plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


# In[28]:


img=cv2.imread('images1.jpg')


# In[29]:


plt.imshow(img)


# In[30]:


plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


# In[31]:


ClassIndex, confidece, bbox=model.detect(img,confThreshold=0.5)
print(ClassIndex)


# In[32]:


font_scale =3
font =cv2.FONT_HERSHEY_PLAIN
for ClassInd, conf ,boxes in zip(ClassIndex.flatten(), confidece.flatten(), bbox):
    cv2.rectangle(img,boxes,(255,0,0),2)
    cv2.putText(img,classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40),font,fontScale=font_scale,color=(0,255,0),thickness=3)
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


# In[33]:


img=cv2.imread('avik.jpeg')


# In[34]:


plt.imshow(img)


# In[35]:


plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


# In[36]:


ClassIndex, confidece, bbox=model.detect(img,confThreshold=0.5)
print(ClassIndex)


# In[37]:


font_scale =3
font =cv2.FONT_HERSHEY_PLAIN
for ClassInd, conf ,boxes in zip(ClassIndex.flatten(), confidece.flatten(), bbox):
    cv2.rectangle(img,boxes,(255,0,0),2)
    cv2.putText(img,classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40),font,fontScale=font_scale,color=(0,255,0),thickness=3)


# In[38]:


plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


# In[39]:


img=cv2.imread('elephant.jpg')


# In[40]:


plt.imshow(img)


# In[41]:


plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


# In[42]:


ClassIndex, confidece, bbox=model.detect(img,confThreshold=0.5)
print(ClassIndex)


# In[43]:


font_scale =3
font =cv2.FONT_HERSHEY_PLAIN
for ClassInd, conf ,boxes in zip(ClassIndex.flatten(), confidece.flatten(), bbox):
    cv2.rectangle(img,boxes,(255,0,0),2)
    cv2.putText(img,classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40),font,fontScale=font_scale,color=(0,255,0),thickness=3)


# In[44]:


plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


# # VIDEO PART

# In[66]:


cap=cv2.VideoCapture('testingvid.mp4')

if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open video")
    
font_scale=3
font=cv2.FONT_HERSHEY_PLAIN


while True:
    ret,frame=cap.read()
    ClassIndex,confidece,bbox=model.detect(frame,confThreshold=0.55)
    
    
    print(ClassIndex)
    if(len(ClassIndex)!=0):
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(),confidece.flatten(),bbox):
            if(ClassInd<=80):
                cv2.rectangle(frame,boxes,(255,0,0),2)
                cv2.putText(frame,classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40),font,fontScale=font_scale,color=(0,255,0),thickness=3)
            
        cv2.imshow('object detection', frame)
        if cv2.waitKey(2)  & 0xFF== ord('q'):
            break
            cap.release()
            cv2.destroyAllWindow()


# In[ ]:





# In[ ]:


cap=cv2.VideoCapture('vdo2.mp4')

if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open video")
    
font_scale=3
font=cv2.FONT_HERSHEY_PLAIN


while True:
    ret,frame=cap.read()
    ClassIndex,confidece,bbox=model.detect(frame,confThreshold=0.55)
    
    
    print(ClassIndex)
    if(len(ClassIndex)!=0):
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(),confidece.flatten(),bbox):
            if(ClassInd<=80):
                cv2.rectangle(frame,boxes,(255,0,0),2)
                cv2.putText(frame,classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40),font,fontScale=font_scale,color=(0,255,0),thickness=3)
            
        cv2.imshow('object detection', frame)
        if cv2.waitKey(2)  & 0xFF== ord('q'):
            break
            cap.release()
            cv2.destroyAllWindow()


# In[ ]:





# In[45]:


cap=cv2.VideoCapture('vdo3.mp4')


if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open video")
    
font_scale=3
font=cv2.FONT_HERSHEY_PLAIN


while True:
    ret,frame=cap.read()
    ClassIndex,confidece,bbox=model.detect(frame,confThreshold=0.55)
    
    
    print(ClassIndex)
    if(len(ClassIndex)!=0):
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(),confidece.flatten(),bbox):
            if(ClassInd<=80):
                cv2.rectangle(frame,boxes,(255,0,0),2)
                cv2.putText(frame,classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40),font,fontScale=font_scale,color=(0,255,0),thickness=3)
            
        cv2.imshow('object detection', frame)
        if cv2.waitKey(2)  & 0xFF== ord('q'):
            break
            cap.release()
            cv2.destroyAllWindow()


# # WEBCAM

# In[ ]:


cap=cv2.VideoCapture(1)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcame")
    
font_scale=3
font=cv2.FONT_HERSHEY_PLAIN


while True:
    ret,frame=cap.read()
    ClassIndex,confidece,bbox=model.detect(frame,confThreshold=0.55)
    
    
    print(ClassIndex)
    if(len(ClassIndex)!=0):
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(),confidece.flatten(),bbox):
            if(ClassInd<=80):
                cv2.rectangle(frame,boxes,(255,0,0),2)
                cv2.putText(frame,classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40),font,fontScale=font_scale,color=(0,255,0),thickness=3)
            
        cv2.imshow('object detection', frame)
        if cv2.waitKey(2)  & 0xFF== ord('q'):
            break
            cap.release()
            cv2.destroyAllWindow()


# In[ ]:




