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

