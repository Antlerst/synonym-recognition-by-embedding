# -*- coding:utf-8 -*-
# License: MIT License

import logging

import torchtext
from torchtext.vocab import Vectors


class PreProcess(object):
    """pre-process the synonym words and embedding"""
    words_field = torchtext.data.Field(sequential=True,
                                       batch_first=True,
                                       pad_token=None,
                                       unk_token=None,
                                       tokenize=(lambda s: s.split(',')))

    def __init__(self, embedding_path):
        self._embedding_path = embedding_path

    def load_data(self, *filenames):
        """load dataset once"""
        if not len(filenames):
            raise RuntimeError('Function [load_data] need at least one path of dataset.')

        vectors = Vectors(name=self._embedding_path)

        datasets = []
        for filename in filenames:
            dataset = torchtext.data.TabularDataset(filename,
                                                    fields=[('words', self.words_field)],
                                                    format='tsv')
            logging.info('Dataset {}:'.format(filename))
            logging.info('\tTotal Length: {}'.format(len(dataset.examples)))
            self._filter_not_existed_embedding(dataset, vectors.stoi)
            logging.info('\tExisted in embedding Length: {}'.format(len(dataset.examples)))

            datasets.append(dataset)

        self.words_field.build_vocab(*datasets, vectors=vectors)

        return datasets if len(datasets) > 1 else datasets[0]

    @staticmethod
    def _filter_not_existed_embedding(dataset, stoi):
        dataset.examples = list(filter(
            lambda x: x.words[0] in stoi and x.words[1] in stoi, dataset.examples
        ))
