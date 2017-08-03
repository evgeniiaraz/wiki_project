# Wikipedia project

Summer 2017 project at CLSP, JHU

## Prerequisites

Install **gensim** package

Install **numpy** package

Install **pickle** package

Install **os** package

Install **sys** package

Install **logging** package

Install **collections** package

Install **multiprocessing** package

## Data 

Wikipedia Dump of april 2016
(so that it is comparable with Jered's implementation based on categories)

The corpus is English Wikipedia snapshot of april 2016. It contains 4010649 documents, with 591398 different word tokens in them. 

(stored in ``/export/b08/erazumo/library/data``. File: ``pages_articles_en.xml.bz2``, 12GB)

The corpus which was used to test the codes is the SimpleWikipedia dump which has around 81000 articles. 

(stored in ```/export/b08/erazumo/library/data```. File: ```simplewiki-latest-pages-articles.xml.bz2```, 121 MB)


## Wikipedia corpus Representations

#### Bag of words
Create bag-of-words model from the whole corpus - lemmatize the words: obtain their stem and mark with their POS tag
  
*bag-of-words* model is stored in ```/export/b08/erazumo/library/representations/bow```
 
to load the BOW representation: ```   mm = gensim.corpora.MmCorpus(bz2.BZ2File('/export/b08/erazumo/library/representations/bow/wiki-en_bow.mm.bz2'))   ```, 2.5 GB
  
                                      created in Gensim's make_wikicorpus.py script

#### TF-IDF representation

TF-IDF representation of wikipedia corpus is stored in ```/export/b08/erazumo/library/representations/tfidf```

to load the TF-IDF representation:```mm = gensim.corpora.MmCorpus('/export/b08/erazumo/library/representations/tfidf/wiki-en_tfidf.mm') ```, 19GB

((((Train TF-IDF model(```data/wiki-en.tfidf_model```) on the Bag-of-words model and dictionary created above

Transform the corpus into TF-IDF vectors. The transformed corpus is stored in ```data/wiki-en_tfidf.mm.bz2```. ))))


#### LDA

LDA representation is stored in ```/export/b08/erazumo/library/representations/lda```

LDA representation has 300 features(~topics).

to load the LDA representation: ```mm = gensim.corpora.MmCorpus.load('/export/b08/erazumo/representations/lda/lda_index.mm')```, 4.5 GB with the index

```lda_index.mm.index.npy``` is a numpy array with all the representations 

#### Doc2Vec representation

The doc2vec representation is stored in ```/export/b08/erazumo/library/representations/doc2vec```

It is stored as a numpy array, so, it can be loaded as ```mm = numpy.load(/export/b08/erazumo/library/representations/doc2vec/doc2vec_vecs.npy)```, 4.5 GB

It has 300 features analogously to LDA. 

**NB:** the representations stored as a dense MmCorpus can be turned into numpy array using ```gensim.matutils.corpus2dense(corpus, num_terms = number_of_features)```. You can also provide length of corpus to the ```num_docs``` parameter for better memory efficiency. 


(((((I have tried to run hierarchical Dirichlet process on the whole corpus not to limit the number of topics, but it would take around a week :()

To find the order of number of topics in the Wikipedia corpus, the hierarchical Dirichlet process was run on the latest Simple Wiki dump(it has around 1/4 of the number of documents in the whole corpus). The overall nummber of topics determined is 147. (The resulting model: ```data/simplewiki.hdp_model```)

The LDA model is trained on 300 topics (also - read up on the Web: for large collections 100-500 should be the right number). (**further experiments can run here to determine the best number of topics**) (The resulting model: ```data/wiki-en.lda_model```)

Then, the wiki corpus was transformed into LDA space stored in ```data/wiki-en_lda.mm```))))))




  

## Indices

#### Dictionary
  * discard the words which occurred in more than 10% of the documents(~ corpus related stopwords) and less than 20 documents(too rare for such a large corpus)

Dictionary gives a word<->id mapping. For exampe, if the bag of words vector states the word number 3000 occurs 5 times in a document, this dictionary will tell you what the word 3000 is.  

Dictionary is stored in ``` /export/b08/erazumo/library/indices/wiki-en_wordids.txt.bz2 ```

to load the dictionary: ```dict = gensim.corpora.dictionary.Dictionary.load_from_text('/export/b08/erazumo/library/indices/wiki-en_wordids.txt.bz2')```
 
#### Title to id 

Title-to-id dictionary gives the mapping between a title of an article and its id. If the transformations are turned into numpy array and you take raw 3000, it will be the transformation for article with id 3000. 

Mapping is stored in ``` /export/b08/erazumo/library/indices/titles_to_id.pickle ```

to load it:  ```dict = pickle.load(open('/export/b08/erazumo/library/indices/titles_to_id.pickle'))```

#### Id to title
Reverse mapping.

Stored in ```/export/b08/erazumo/library/indices/ids_to_titles.cpickle```

to load it: ```dict = pickle.load(open('/export/b08/erazumo/library/indices/ids_to_titles.cpickle'))```


## Scripts

### Clustering

*/export/b08/erazumo/library/clustering/clustering_minibk.py*

Input: 

numpy array with vectors of articles;
number of clusters to cluster in. 

Output:
two files: 
numpy array 'labels_*batch size*_mni_*maximum no improvement parameter*.npy' with a label for each article (kmeans.labels_)
numpy array 'centroids_*batch size*_mni_*maximum no improvement parameter*.npy' with coordinates of the centroids for the n clusters

it also prints the total inertia 

Script which flatly clusters the vectors for articles in *n* clusters. 
It is completed using minibatch k-means with the settings that make the clustering run the fastest(***change those in the script itself***)

*/export/b08/erazumo/library/clustering/hierarchical_kmeans.py*

Input:
vectorisation of Wikipedia dump stored as numpy array
cluster assignment from flat clustering
title to id mapping between articles
id to title mapping between articles

Output:
'article_clustering.npy' dictionary where each entry is of the form title_of_article:[list of integers] where the integers represent the clusters the article has been clustered into so that it is a hierarchy. 


The script implements the hierarchical MiniBatch K-Means, top-down clustering. It is parallelised using multiprocessing.Pool. 
It uses the same batch settings as the *clustering_minibk.py* flat clustering and recursively divides all the clusters is two. 
It does not take the input as arguments, you would need to put them in the script, but I can modify the script so that it does take them. If you need it modified, let me know. 



#### Testing and evaluation

*/export/b08/erazumo/library/clustering/testing/checking_categories.py*

Inputs:
label assigned to each document in a ```numpy``` array

csv table with information on which document goes in which category formatted as:
it is a csv file containing clusterid, list of articles where each article is a tuple consisting of (article title, vector) so row[2] will get you the list for each cluster and than list[index][0] will get you the title at index 0
                                                                              (from message from Jered, 12 July)
(a sample csv file can be found in '/export/b08/dmcinerney/ForZhenya/dataset.csv')


[optional: the title_to_id index as a pickled dictionary; default: the index in ```/export/b08/erazumo/library/indices/titles_to_id.pickle```]

Output:

"clustering.txt" file which looks like:
Category Philosophy
1
1
3
5
...

where 1,1,3,5,... are clusters to which the articles in Philosophy category are assigned. 

The aim: 
to see the composition of categories labelled by humans in Wikipedia in terms of unsupervised clusters created. 

ideally, we want the following kind of mapping: category "Philosophy": 1,1,1,1,1,1,1, so that all the articles assigned to one cluster fall into one category and vice versa - that articles from cluster 1 do not appear in any other category.


(For my task: that one to one mapping did not happen, but the misclassification made sense. The summary of the categories and misclassifications(~qualitative evaluation) is in ```/export/b08/erazumo/library/clustering/testing/category_summary.txt```;
output of the actual cheching_categories.py code is in the same folder in  ```clustering.txt``` file)



*/export/b08/erazumo/library/scripts/clustering/testing/clustering_titles.py*

Input:
numpy array with labels of clusters for each article
[pickled dictionary with title-to-id mapping; if not given, ```/export/b08/erazumo/library/indices/titles_to_id.pickle``` is used] 

Output:
Text files with titles of articles assigned to each cluster.


I used the output of this script to understand what is the content of each cluster (and if it made sense) 
(my qualitative judgement is given in ```/export/b08/erazumo/library/clustering/testing/category_summary.txt```)

*/export/b08/erazumo/library/scripts/clustering/testing/distance_centroids.py*
Input:
numpy array with coordinates of centroids('kmeans.centroids_' saved to a numpy array)

Output:
'distances.npy' which stores pairwise distance between all centroids

This script is for quantitatively evaluating how good the flat clustering is, seeing, how widespread/close the centroids are. 






#### Side notes:
1) an experiment: compare the documents retrieved when using LDA/LDA after k-Means(or another clustering algorithm)/Jered/just clustering on tf-idf depending on time it takes

2) to future me for evaluation: TREC corpora with (topics, document, relevance); not the same collection for all topics, but overlapping

3) doc2vec: https://markroxor.github.io/gensim/static/notebooks/doc2vec-wikipedia.html

https://cs.stanford.edu/~quocle/paragraph_vector.pdf


/export/b08/dmcinerney/ForZhenya
           
use DBOW - it is better for computing similarity 

https://arxiv.org/pdf/1607.05368.pdf



