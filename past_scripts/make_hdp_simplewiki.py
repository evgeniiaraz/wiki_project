#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Copyright (C) 2012 Lars Buitinck <larsmans@gmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""
USAGE: %(program)s WIKI_XML_DUMP OUTPUT_PREFIX [VOCABULARY_SIZE]

Convert articles from a Wikipedia dump to (sparse) vectors. The input is a
bz2-compressed dump of Wikipedia articles, in XML format.

This actually creates three files:

* `OUTPUT_PREFIX_wordids.txt`: mapping between words and their integer ids
* `OUTPUT_PREFIX_bow.mm`: bag-of-words (word counts) representation, in
  Matrix Matrix format
* `OUTPUT_PREFIX_tfidf.mm`: TF-IDF representation
* `OUTPUT_PREFIX.tfidf_model`: TF-IDF model dump

The output Matrix Market files can then be compressed (e.g., by bzip2) to save
disk space; gensim's corpus iterators can work with compressed input, too.

`VOCABULARY_SIZE` controls how many of the most frequent words to keep (after
removing tokens that appear in more than 10%% of all documents). Defaults to
100,000.

If you have the `pattern` package installed, this script will use a fancy
lemmatization to get a lemma of each token (instead of plain alphabetic
tokenizer). The package is available at https://github.com/clips/pattern .

Example: python -m gensim.scripts.make_wikicorpus ~/gensim/results/enwiki-latest-pages-articles.xml.bz2 ~/gensim/results/wiki_en
"""


import logging
import os.path
import sys
import gensim
import bz2
from gensim.corpora import Dictionary, HashDictionary, MmCorpus, WikiCorpus
from gensim.models import TfidfModel


# Wiki is first scanned for all distinct word types (~7M). The types that
# appear in more than 10% of articles are removed and from the rest, the
# DEFAULT_DICT_SIZE most frequent types are kept.
DEFAULT_DICT_SIZE = 8000000


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    online = 'online' in program
    lemmatize = 'lemma' in program
    debug = 'nodebug' not in program
    # initialize corpus reader and word->id mapping
    id2word = gensim.corpora.Dictionary.load_from_text(bz2.BZ2File('data/simplewiki_wordids.txt.bz2'))

    # build tfidf, ~50min
    mm = gensim.corpora.MmCorpus('data/simplewiki_tfidf.mm')

    hdp = gensim.models.HdpModel(mm, id2word)

    hdp.save('data/wiki-en.hdp_model')

    # save tfidf vectors in matrix market format
    # ~4h; result file is 15GB! bzip2'ed down to 4.5GB

    logger.info("finished running %s" % program)
