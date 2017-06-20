# Wikipedia project

Summer 2017 project at CLSP, JHU


## Data 

Wikipedia Dump of april 2016
(so that it is comparable with Jered's implementation based on categories)

The corpus is English Wikipedia snapshot of april 2016. It contains 4010649 documents, with 591398 different word tokens in them. 

## Pre-processing

1) Create bag-of-words model from the whole corpus - lemmatize the words: obtain their stem and mark with their POS tag
  
  *bag-of-words* model is stored in ```data/wiki-en_bow.mm.bz2 ```

2) When creating a dictionary:
  * discard the words which occurred in more than 10% of the documents(~ corpus related stopwords) and less than 20 documents(too rare for such a large corpus)

  *dictionary* is stored in ``` data/wiki-en_wordids.txt.bz2 ```
  
                                      created in Gensim's make_wikicorpus.py script

## Transformations

#### TF-IDF

Train TF-IDF model(```data/wiki-en.tfidf_model```) on the Bag-of-words model and dictionary created above

Transform the corpus into TF-IDF vectors. The transformed corpus is stored in ```data/wiki-en_tfidf.mm.bz2```. 


#### LDA

(I have tried to run hierarchical Dirichlet process on the whole corpus not to limit the number of topics, but it would take around a week :()

To find the order of number of topics in the Wikipedia corpus, the hierarchical Dirichlet process was run on the latest Simple Wiki dump(it has around 1/4 of the number of documents in the whole corpus). The overall nummber of topics determined is 147. (The resulting model: ```data/simplewiki.hdp_model```)

The LDA model is trained on 300 topics (also - read up on the Web: for large collections 100-500 should be the right number). (**further experiments can run here to determine the best number of topics**)






           
  



