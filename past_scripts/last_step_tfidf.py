import gensim

mm = gensim.corpora.MmCorpus('./data/wiki-en_bow.mm')
tfidf = gensim.models.TfidfModel.load('./data/wiki-en.tfidf_model')
gensim.corpora.MmCorpus.serialize('wiki-en_tfidf.mm', tfidf[mm], progress_cnt = 10000)
