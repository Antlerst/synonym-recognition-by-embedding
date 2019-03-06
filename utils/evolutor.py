# -*- coding:utf-8 -*-
# License: MIT License

import logging
import os

import numpy as np
import torch.nn as nn
import torchtext
from sklearn.metrics import precision_recall_curve


class Evaluator(object):
    """Evaluate cosine similarity between words tuple using pre-trained word embedding"""

    KEY_WORDS = 'words'
    KEY_COS = 'cos'

    def __init__(self, embedding, outputs_dir):
        self._embedding = embedding
        self._outputs_dir = outputs_dir
        self._cosine_sim = nn.CosineSimilarity(dim=-1)

    def evaluate(self, dataset_pos, dataset_neg):
        pos_eval_dict = self.evaluate_dataset(dataset_pos, label='pos')
        neg_eval_dict = self.evaluate_dataset(dataset_neg, label='neg')
        self.get_precision_recall_curve(pos_eval_dict, neg_eval_dict)

    def evaluate_dataset(self, dataset, label):
        batch_iter = torchtext.data.Iterator(dataset=dataset,
                                             train=False,
                                             sort=False,
                                             batch_size=len(dataset))

        eval_dict = dict()

        # only iterate once
        for batch in batch_iter:
            words = batch.words
            words_embed = self._embedding(words)
            cos = self._cosine_sim(words_embed[:, 0, :], words_embed[:, 1, :])

            eval_dict[Evaluator.KEY_WORDS] = words
            eval_dict[Evaluator.KEY_COS] = cos

        self.save_words_cos(dataset, label, eval_dict)
        return eval_dict

    def save_words_cos(self, dataset, label, eval_dict):
        """save words and its cosine similarity for future analysis"""
        vocab = dataset.fields['words'].vocab

        words = eval_dict[Evaluator.KEY_WORDS]
        cos = eval_dict[Evaluator.KEY_COS]

        words_path = os.path.join(self._outputs_dir, '{}.txt'.format(label))
        with open(words_path, 'w', encoding='utf-8') as f:
            for i in range(len(dataset)):
                f.write('{},{},{:.4f}\n'.format(
                    vocab.itos[words[i][0]],
                    vocab.itos[words[i][1]],
                    cos[i]
                ))
        logging.info('Words and cosine similarity is saved to {}'.format(words_path))

    def get_precision_recall_curve(self, pos_eval_dict, neg_eval_dict):
        """get precision & call tuples using pos & neg cosine similarity"""
        pos_cos = pos_eval_dict[Evaluator.KEY_COS]
        neg_cos = neg_eval_dict[Evaluator.KEY_COS]

        labels = np.hstack([np.ones(len(pos_cos)), np.zeros(len(neg_cos))])
        scores = np.hstack([pos_cos, neg_cos])
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        max_f1_index = np.argmax(f1)

        prf_path = os.path.join(self._outputs_dir, 'prf.txt')
        with open(prf_path, 'w', encoding='utf-8') as f:
            f.write('Precision,recall,f1,threshold\n')
            for i in range(len(precision) - 1):
                f.write('{:.4f},{:.4f},{:.4f},{:.4f}\n'.format(
                    precision[i],
                    recall[i],
                    f1[i],
                    thresholds[i]
                ))
        logging.info('Precision, recall, f1 and threshold is saved to {}'.format(prf_path))
        logging.info('Max F1 is {2:.4f}, and the corresponding item is {0:.4f},{1:.4f},{2:.4f},{3:.4f}'.format(
            precision[max_f1_index],
            recall[max_f1_index],
            f1[max_f1_index],
            thresholds[max_f1_index]))
