# -*- coding:utf-8 -*-
# License: MIT License

import argparse
import logging

import torch.nn as nn

from utils.evolutor import Evaluator
from utils.io import rebuild_dir
from utils.preprocess import PreProcess

__version__ = '0.1.0'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pos', action='store', dest='pos', default='datasets/examples/pos.csv',
                        help='Positive test data.')
    parser.add_argument('--neg', action='store', dest='neg', default='datasets/examples/neg.csv',
                        help='Negative test data.')
    parser.add_argument('--embedding', action='store', dest='embedding',
                        default='datasets/examples/embedding.txt',
                        help='Pre-trained word embedding.')
    parser.add_argument('--outputs', action='store', dest='outputs', default='outputs/default',
                        help='Dir of intermediate files.')
    parser.add_argument('--log-level', dest='log_level', default='info',
                        help='Logging level.')

    opts = parser.parse_args()

    # mkdir outputs if not dir
    rebuild_dir(opts.outputs)

    # logger configure
    LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opts.log_level.upper()))
    logging.info(opts)

    # pre-process dataset
    pre_process = PreProcess(embedding_path=opts.embedding)
    dataset_pos, dataset_neg = pre_process.load_data(opts.pos, opts.neg)

    # pre-trained embedding
    vectors = pre_process.words_field.vocab.vectors
    embedding = nn.Embedding(*vectors.size())
    embedding.weight = nn.Parameter(vectors, requires_grad=False)

    # evaluate synonyms using embedding
    evaluator = Evaluator(embedding=embedding, outputs_dir=opts.outputs)
    evaluator.evaluate(dataset_pos, dataset_neg)
