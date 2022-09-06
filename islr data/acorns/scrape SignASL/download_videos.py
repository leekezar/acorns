#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
plf = pd.read_csv("plf_lexica.csv")


# In[3]:


videos = pd.read_csv("videos.csv")


# In[4]:


import json
videos_json = {}
words = list(plf.iloc[:,0])

for row in videos.iterrows():
    word = str(row[1][0]).lower()
    
    if word not in words:
        continue
    
    print(row)
    
    if word not in videos_json.keys():
        videos_json[word] = { 
            "urls" : [], 
            "plfs" : list(plf.loc[plf["Entry ID"] == word].to_numpy()[0,1:]) }
    
    videos_json[word]["urls"].append(row[1][2])


# In[ ]:


videos_json["hello"]


# In[ ]:


json.dump(videos_json,open("videos.json","w+"),indent=True)


# In[6]:


import json
videos_json = json.load(open("videos.json"))


# In[10]:


import os
num_vids = 0
for word in videos_json.keys():
    if num_vids > 100:
            break
            
    url = videos_json[word]["urls"][0]
    
    print(word, url)
    if "youtube" in url:
        os.system(f'youtube-dl {url} -o ./videos/{word}.mp4')
    else:
        os.system(f'wget {url} -O ./videos/{word}.mp4')

    num_vids += 1
        


# In[ ]:




