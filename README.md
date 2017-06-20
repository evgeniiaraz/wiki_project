# Wikipedia project

Summer 2017 project at CLSP, JHU


## Data 

Wikipedia Dump of april 2016
(so that it is comparable with Jered's implementation based on categories)

The corpus is English Wikipedia snapshot of april 2016. It contains 4010649 documents, with 591398 different word tokens in them. 

## Pre-processing

1) Create bag-of-words model from the whole corpus - lemmatize the words: obtain their stem and mark with their POS tag
  
  *bag-of-words* model is stored in ```data/wiki-en_bow.mm ```

2) When creating a dictionary:
  * discard the words which occurred in more than 10% of the documents(~ corpus related stopwords) and less than 20 documents(too rare for such a large corpus)

  *dictionary* is stored in ``` data/wiki-en_wordids.txt.bz2 ```
  
                                                                              created in ```Gensim's make_wikicorpus.py``` script

## Transformations
           
  



