#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Copyright (C) 2012 Lars Buitinck <larsmans@gmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import logging
import os.path
import sys
from gensim import corpora, models
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
    mm = corpora.MmCorpus('./data/wiki_en_bow.mm.gz')

    # build tfidf, ~50min
    tfidf = models.TfidfModel.load('./data/wiki_en.tfidf_model', mmap = 'r')
    # save tfidf vectors in matrix market format
    # ~4h; result file is 15GB! bzip2'ed down to 4.5GB
    MmCorpus.serialize('./data/wiki_en_tfidf.mm', tfidf[mm], progress_cnt=10000)

    logger.info("finished running %s" % program)
