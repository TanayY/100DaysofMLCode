# Note --> Readme is not completely updated yet. Stay Tuned.

## T-Shirt Recommendation System

In this blog I will take you through the steps that I went through while making the T-shirt Recommendation System.

Basically I learnt to make this system back in summer by following the Introductory course by [Applied AI Course](https://www.appliedaicourse.com/)

The course is awesome and it takes you from the basic concepts like Linear Algebra to some very good concepts like bag of words and TF-IDF used in Machine Learning.

The basic libraries used in making of this system are :

1] scikit learn <br/>
2] pandas <br/>
3] numpy <br/>
4] matplotlib <br/>
5] keras

The following steps make up the whole system :

1] Creating a bag of words and finding Euledian Distance between every title <br/>
2] Creating a TF-IDF system and then finding Eucledian Distance between every title <br/>
3] Using the VGG16 pre-trained Neural Network to do image comparison and then applying the Eucledian distance method. <br/>
4] Finally merging all the above methods and giving weights to all of the above methods and finding the best solution.

### Importing all the libraries needed :

```python
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import math
import time
import re
import os
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity  
from sklearn.metrics import pairwise_distances
from matplotlib import gridspec
from scipy.sparse import hstack
import plotly
import plotly.figure_factory as ff
from plotly.graph_objs import Scatter, Layout

plotly.offline.init_notebook_mode(connected=True)
warnings.filterwarnings("ignore")
```

### Data Cleaning and Formatting :

The main thing that matters the most in Machine Learning is the data that you have.<br/>
If you have a good amount of data that is useful for the problem that you are solving then your system is surely to produce better results.<br/>
We do all of this data formatting and preprocessing using the **pandas** library in python.

The first step is to get the data of T-shirts and storing it into a pandas data-frame.<br/>
We get this data from Amazon's API Service.<br/>
You can get the data file by clicking on this [link](https://drive.google.com/file/d/0BwNkduBnePt2LTRsVEg1WjJiSFk/view?usp=sharing).<br/>
It is a json file and you can store it in a data-framw using the following code :

```python

data = pd.read_json('tops_fashion.json')

```
Here pd refers to pandas which we imported in the beginning as pd.<br/>

You can now find the shape of all the data points and variables using the following code :

```python

print ('Number of data points : ', data.shape[0], \
       'Number of features/variables:', data.shape[1])
       
```

The output will be :
```diff
-Number of data points :  183138 
-Number of features/variables: 19
```

The further steps are all about cleaning the data. As this blog does not focus that much on data cleaning and formatting and mainly focuses on the concepts used in recommendation systems, I am going to skip all these steps and jump to the main part.<br/>

The preprocessed data is avvailable on my github repository inside the visual similarity folder.<br/>
The file in named as **16k_preprocessed_data**<br/>
Though here are a few things you need to know before skipping to the main part of the system.<br/><br/>
After all the prepreocessing thats been done you have the following features left in the dataset :<br/>

1. asin
2. brand
3. color
4. image url
5. product type
6. title
7. price

The number of data points remaining after reducing the data from 180k to 16k is : 16042

Now we are good to go.

### Similarity based on Text Comparison :

The first thing is to storing the preprocessed data in a pandas data-frame :

```python
data = pd.read_pickle('pickels/16k_apperal_data_preprocessed')
```

Following are some utility functions used to plot graphs and images : <br/>

#### Note --> The blog will not explain these utility functions as they are not related to any ML Concepts.

```python
# Utility Functions 


#Display an image
def display_img(url,ax,fig):
    # we get the url of the apparel and download it
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    # we will display it 
    plt.imshow(img)
  
#plotting code to understand the algorithm's decision.
def plot_heatmap(keys, values, labels, url, text):
        # keys: list of words of recommended title
        # values: len(values) ==  len(keys), values(i) represents the occurence of the word keys(i)
        # labels: len(labels) == len(keys), the values of labels depends on the model we are using
                # if model == 'bag of words': labels(i) = values(i)
                # if model == 'tfidf weighted bag of words':labels(i) = tfidf(keys(i))
                # if model == 'idf weighted bag of words':labels(i) = idf(keys(i))
        # url : apparel's url

        # we will devide the whole figure into two parts
        gs = gridspec.GridSpec(2, 2, width_ratios=[4,1], height_ratios=[4,1]) 
        fig = plt.figure(figsize=(25,3))
        
        # 1st, ploting heat map that represents the count of commonly ocurred words in title2
        ax = plt.subplot(gs[0])
        # it displays a cell in white color if the word is intersection(lis of words of title1 and list of words of title2), in black if not
        ax = sns.heatmap(np.array([values]), annot=np.array([labels]))
        ax.set_xticklabels(keys) # set that axis labels as the words of title
        ax.set_title(text) # apparel title
        
        # 2nd, plotting image of the the apparel
        ax = plt.subplot(gs[1])
        # we don't want any grid lines for image and no labels on x-axis and y-axis
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # we call dispaly_img based with paramete url
        display_img(url, ax, fig)
        
        # displays combine figure ( heat map and image together)
        plt.show()
    
def plot_heatmap_image(doc_id, vec1, vec2, url, text, model):

    # doc_id : index of the title1
    # vec1 : input apparels's vector, it is of a dict type {word:count}
    # vec2 : recommended apparels's vector, it is of a dict type {word:count}
    # url : apparels image url
    # text: title of recomonded apparel (used to keep title of image)
    # model, it can be any of the models, 
        # 1. bag_of_words
        # 2. tfidf
        # 3. idf

    # we find the common words in both titles, because these only words contribute to the distance between two title vec's
    intersection = set(vec1.keys()) & set(vec2.keys()) 

    # we set the values of non intersecting words to zero, this is just to show the difference in heatmap
    for i in vec2:
        if i not in intersection:
            vec2[i]=0

    # for labeling heatmap, keys contains list of all words in title2
    keys = list(vec2.keys())
    #  if ith word in intersection(lis of words of title1 and list of words of title2): values(i)=count of that word in title2 else values(i)=0 
    values = [vec2[x] for x in vec2.keys()]
    
    # labels: len(labels) == len(keys), the values of labels depends on the model we are using
        # if model == 'bag of words': labels(i) = values(i)
        # if model == 'tfidf weighted bag of words':labels(i) = tfidf(keys(i))
        # if model == 'idf weighted bag of words':labels(i) = idf(keys(i))

    if model == 'bag_of_words':
        labels = values
    elif model == 'tfidf':
        labels = []
        for x in vec2.keys():
            # tfidf_title_vectorizer.vocabulary_ it contains all the words in the corpus
            # tfidf_title_features[doc_id, index_of_word_in_corpus] will give the tfidf value of word in given document (doc_id)
            if x in  tfidf_title_vectorizer.vocabulary_:
                labels.append(tfidf_title_features[doc_id, tfidf_title_vectorizer.vocabulary_[x]])
            else:
                labels.append(0)
    elif model == 'idf':
        labels = []
        for x in vec2.keys():
            # idf_title_vectorizer.vocabulary_ it contains all the words in the corpus
            # idf_title_features[doc_id, index_of_word_in_corpus] will give the idf value of word in given document (doc_id)
            if x in  idf_title_vectorizer.vocabulary_:
                labels.append(idf_title_features[doc_id, idf_title_vectorizer.vocabulary_[x]])
            else:
                labels.append(0)

    plot_heatmap(keys, values, labels, url, text)


# this function gets a list of wrods along with the frequency of each 
# word given "text"
def text_to_vector(text):
    word = re.compile(r'\w+')
    words = word.findall(text)
    # words stores list of all words in given string, you can try 'words = text.split()' this will also gives same result
    return Counter(words) # Counter counts the occurence of each word in list, it returns dict type object {word1:count}



def get_result(doc_id, content_a, content_b, url, model):
    text1 = content_a
    text2 = content_b
    
    # vector1 = dict{word11:#count, word12:#count, etc.}
    vector1 = text_to_vector(text1)

    # vector1 = dict{word21:#count, word22:#count, etc.}
    vector2 = text_to_vector(text2)

    plot_heatmap_image(doc_id, vector1, vector2, url, text2, model)
 ```
 You will have to copy this code to give a visual output.<br/>

#### Note --> The blog will not explain these utility functions as they are not related to any ML Concepts.

