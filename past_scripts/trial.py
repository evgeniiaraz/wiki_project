import logging
import itertools

import numpy as np
import gensim

import time

start = time.time()

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO  # ipython sometimes messes up the logging setup; restore

def head(stream, n=10):
    """Convenience fnc: return the first `n` elements of the stream, as plain list."""
    return list(itertools.islice(stream, n))


from gensim.utils import smart_open, simple_preprocess
from gensim.corpora.wikicorpus import _extract_pages, filter_wiki
from gensim.parsing.preprocessing import STOPWORDS

def tokenize(text):
    return [token for token in simple_preprocess(text) if token not in STOPWORDS]

def iter_wiki(dump_file):
    """Yield each article from the Wikipedia dump, as a `(title, tokens)` 2-tuple."""
    ignore_namespaces = 'Wikipedia Category File Portal Template MediaWiki User Help Book Draft'.split()
    for title, text, pageid in _extract_pages(smart_open(dump_file)):
        text = filter_wiki(text)
        tokens = tokenize(text)
        if len(tokens) < 50 or any(title.startswith(ns + ':') for ns in ignore_namespaces):
            continue  # ignore short articles and various meta-articles
        yield title, tokens

# only use simplewiki in this tutorial (fewer documents)
# the full wiki dump is exactly the same format, but larger
stream = iter_wiki('pages_articles_en.xml.bz2')
for title, tokens in itertools.islice(iter_wiki('pages_articles_en.xml.bz2'), 8):
  print title, tokens[:10]  # print the article title and its first ten tokens

doc_stream = (tokens for _, tokens in iter_wiki('pages_articles_en.xml.bz2'))
id2word_wiki = gensim.corpora.Dictionary(doc_stream, prune_at= 20000000000000)
id2word_wiki.filter_extremes(no_above=0.1)

class WikiCorpus(object):
    def __init__(self, dump_file, dictionary, clip_docs=None):
        """
        Parse the first `clip_docs` Wikipedia documents from file `dump_file`.
        Yield each document in turn, as a list of tokens (unicode strings).

        """
        self.dump_file = dump_file
        self.dictionary = dictionary
        self.clip_docs = clip_docs

    def __iter__(self):
        self.titles = []
        for title, tokens in itertools.islice(iter_wiki(self.dump_file), self.clip_docs):
            self.titles.append(title)
            yield self.dictionary.doc2bow(tokens)

    def __len__(self):
        return self.clip_docs

# create a stream of bag-of-words vectors
wiki_corpus = WikiCorpus('pages_articles_en.xml.bz2', id2word_wiki)

gensim.corpora.MmCorpus.serialize('./data/wiki_bow.mm', wiki_corpus)

mm_corpus = gensim.corpora.MmCorpus('./data/wiki_bow.mm')

tfidf_model = gensim.models.TfidfModel(mm_corpus, id2word=id2word_wiki)

gensim.corpora.MmCorpus.serialize('./data/wiki_tfidf.mm', tfidf_model[mm_corpus])

tfidf_corpus = gensim.corpora.MmCorpus('./data/wiki_tfidf.mm')

tfidf_model.save('./data/tfidf_wiki.model')
id2word_wiki.save('./data/wiki.dictionary')

print time.time() - start
