# synonym-recognition-by-embedding

![Language](https://img.shields.io/github/languages/top/cloudyyyyy/synonym-recognition-by-embedding.svg?style=flat)
![Licence](https://img.shields.io/github/license/cloudyyyyy/synonym-recognition-by-embedding.svg?style=flat)

## Relation Resource

 - Synonym dataset `datasets/synonyms/*` is built on Chinese Synonym Dataset: [同义词词林](https://www.ltp-cloud.com/download#down_cilin).
 - Pre-train word embedding: 
    1. [Word2vec or Fasttext](https://github.com/Kyubyong/wordvectors)
    2. [Wikipedia2vec](https://wikipedia2vec.github.io/wikipedia2vec/pretrained)
    3. [Tencent AI Lab Embedding Corpus for Chinese Words and Phrases](https://ai.tencent.com/ailab/nlp/embedding.html)
    4. ···
    
## Get Started

### Prepare for synonym dataset

You can use `datasets/synonyms/*` or dataset else you built.

### Download the embedding

Download from the above Pre-train word embedding.

### Dependencies

```
numpy==1.15.4
torch==1.0.1
torchtext==0.3.1
scikit-learn==0.20.3
```

You can install these by:

```shell
python install -r requirements.txt
```

### Run

```shell 
python main.py --pos datasets/synonyms/pos.csv \
               --neg datasets/synonyms/neg.csv \
               --embedding /path/to/file 
```

## License
@MIT
