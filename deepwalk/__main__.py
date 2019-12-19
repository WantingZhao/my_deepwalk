#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import random
from io import open
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import logging
import graph
import numpy as np
import walks as serialized_walks
from gensim.models import Word2Vec
from skipgram import Skipgram
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from six import text_type as unicode
from six import iteritems
from six.moves import range

from sys import path
import scipy.sparse
path.append(r"D:\python\my_mixhop")
from mixhop_dataset import common_load_data


import psutil
from multiprocessing import cpu_count

p = psutil.Process(os.getpid())
try:
    p.set_cpu_affinity(list(range(cpu_count())))
except AttributeError:
    try:
        p.cpu_affinity(list(range(cpu_count())))
    except AttributeError:
        pass

logger = logging.getLogger(__name__)
LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"


def debug(type_, value, tb):
  if hasattr(sys, 'ps1') or not sys.stderr.isatty():
    sys.__excepthook__(type_, value, tb)
  else:
    import traceback
    import pdb
    traceback.print_exception(type_, value, tb)
    print(u"\n")
    pdb.pm()

def deepwalk_get_feature(args, adj_indices,result_path):
    model_path =result_path + '.model'
    if os.path.exists(model_path):
        return Word2Vec.load(model_path)
    G = graph.load_edgelist(adj_indices, undirected=args.undirected)

    print(G)
    print("Number of nodes: {}".format(len(G.nodes())))

    num_walks = len(G.nodes()) * args.number_walks

    print("Number of walks: {}".format(num_walks))

    data_size = num_walks * args.walk_length

    print("Data size (walks*length): {}".format(data_size))

    if data_size < args.max_memory_data_size:
        print("Walking...")
        walks = graph.build_deepwalk_corpus(G, num_paths=args.number_walks,
                                            path_length=args.walk_length, alpha=0, rand=random.Random(args.seed))
        print("Training...")
        model = Word2Vec(walks, size=args.representation_size, window=args.window_size, min_count=0, sg=1, hs=1,
                         workers=args.workers)
    else:
        print("Data size {} is larger than limit (max-memory-data-size: {}).  Dumping walks to disk.".format(data_size,
                                                                                                             args.max_memory_data_size))
        print("Walking...")

        walks_filebase = args.dataset + ".walks"
        walk_files = serialized_walks.write_walks_to_disk(G, walks_filebase, num_paths=args.number_walks,
                                                          path_length=args.walk_length, alpha=0,
                                                          rand=random.Random(args.seed),
                                                          num_workers=args.workers)

        print("Counting vertex frequency...")
        if not args.vertex_freq_degree:
            vertex_counts = serialized_walks.count_textfiles(walk_files, args.workers)
        else:
            # use degree distribution for frequency in tree
            vertex_counts = G.degree(nodes=G.iterkeys())

        print("Training...")
        walks_corpus = serialized_walks.WalksCorpus(walk_files)
        model = Skipgram(sentences=walks_corpus, vocabulary_counts=vertex_counts,
                         size=args.representation_size,
                         window=args.window_size, min_count=0, trim_rule=None, workers=args.workers)

    model.wv.save_word2vec_format(result_path+'.feature')
    model.save(model_path)
    return model

def process(args):
  result_path = 'result/' + args.dataset + (',number-walks=%d' % args.number_walks) \
                  + (',representation-size=%d' % args.representation_size) \
                  + (',walk-length=%d' % args.walk_length) \
                  + (',window-size=%d' % args.window_size)
  dataset_str = '../../my_mixhop/data/ind.' + args.dataset

  num_nodes, edge_sets, metapaths, metapaths_name, train_idx, valid_idx, test_idx, adj_indices, adj_values, allx, ally = common_load_data(dataset_str)
  model = deepwalk_get_feature(args,adj_indices,result_path)

  train_X = model[[str(idx) for idx in train_idx]]
  train_y = [ np.argmax(y) for y in ally[train_idx] ]
  test_X = model[[str(idx) for idx in test_idx]]
  test_y = [ np.argmax(y) for y in ally[test_idx]]

  lr = LogisticRegression()  # 初始化LogisticRegression
  lr.fit(train_X, train_y)  # 使用训练集对测试集进行训练
  lr_y_predit = lr.predict(test_X)  # 使用逻辑回归函数对测试集进行预测
  acc =lr.score(test_X, test_y)
  print('Accuracy of LR Classifier:%f' % acc)  # 使得逻辑回归模型自带的评分函数score获得模型在测试集上的准确性结果


  with open(result_path+'.acc','w') as w:
      w.write('Accuracy of LR Classifier: %f\n' % acc)



def main():
  parser = ArgumentParser("deepwalk",
                          formatter_class=ArgumentDefaultsHelpFormatter,
                          conflict_handler='resolve')

  parser.add_argument("--debug", dest="debug", action='store_true', default=False,
                      help="drop a debugger if an exception is raised.")

  parser.add_argument('--format', default='adjlist',
                      help='File format of input file')

  parser.add_argument('--dataset', default='dblp',
                      help='dataset')

  parser.add_argument("-l", "--log", dest="log", default="INFO",
                      help="log verbosity level")

  parser.add_argument('--matfile-variable-name', default='network',
                      help='variable name of adjacency matrix inside a .mat file.')

  parser.add_argument('--max-memory-data-size', default=1000000000, type=int,
                      help='Size to start dumping walks to disk, instead of keeping them in memory.')

  parser.add_argument('--number-walks', default=10, type=int,
                      help='Number of random walks to start at each node')

  parser.add_argument('--representation-size', default=64, type=int,
                      help='Number of latent dimensions to learn for each node.')

  parser.add_argument('--seed', default=0, type=int,
                      help='Seed for random walk generator.')

  parser.add_argument('--undirected', default=True, type=bool,
                      help='Treat graph as undirected.')

  parser.add_argument('--vertex-freq-degree', default=False, action='store_true',
                      help='Use vertex degree to estimate the frequency of nodes '
                           'in the random walks. This option is faster than '
                           'calculating the vocabulary.')

  parser.add_argument('--walk-length', default=40, type=int,
                      help='Length of the random walk started at each node')

  parser.add_argument('--window-size', default=5, type=int,
                      help='Window size of skipgram model.')

  parser.add_argument('--workers', default=1, type=int,
                      help='Number of parallel processes.')


  args = parser.parse_args()
  numeric_level = getattr(logging, args.log.upper(), None)
  logging.basicConfig(format=LOGFORMAT)
  logger.setLevel(numeric_level)

  if args.debug:
   sys.excepthook = debug

  process(args)

if __name__ == "__main__":
  sys.exit(main())
