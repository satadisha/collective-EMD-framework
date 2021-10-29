import os

# Project directory
# _ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# _ROOT_DIR = '/content/gdrive/My Drive/gaguilar'
_ROOT_DIR = ''

############################################################
# Data Files

_DATA_DIR = _ROOT_DIR + 'data/'
#emerging.test.conll.preproc.url emerging.test.conll.preproc.url.postag
TRAIN = _DATA_DIR + 'emerging.train.conll'
DEV   = _DATA_DIR + 'emerging.dev.conll'
# TEST  = _DATA_DIR + '0_emerging.test.conll'
# TEST  = _DATA_DIR+'NIST/0208/'
# TEST  = _DATA_DIR+'stsbenchmark/'

# TEST  = _DATA_DIR+'tweets_3K/'
# TEST  = _DATA_DIR+'deduplicated/'

TEST  = _DATA_DIR+'covid/'

# TEST  = _DATA_DIR+'venezuela/'
# TEST  = _DATA_DIR+'roevwade/'
# TEST  = _DATA_DIR+'billdeblasio/'
# TEST  = _DATA_DIR+'ripcity/'
# TEST  = _DATA_DIR+'billnye/'
# TEST  = _DATA_DIR+'pikapika/'

# TEST  = _DATA_DIR+'wnut-test/'
# TEST  = _DATA_DIR+'broad_twitter_corpus/'

TRAIN_POSTAG             = TRAIN + '.postag'
TRAIN_PREPROC_URL        = TRAIN + '.preproc.url'
TRAIN_PREPROC_URL_POSTAG = TRAIN + '.preproc.url.postag'

DEV_POSTAG               = DEV   + '.postag'
DEV_PREPROC_URL          = DEV   + '.preproc.url'
DEV_PREPROC_URL_POSTAG   = DEV   + '.preproc.url.postag'

# TEST_POSTAG              = TEST  + '.postag'
# TEST_PREPROC_URL         = TEST  + '.preproc.url'
# TEST_PREPROC_URL_POSTAG  = TEST  + '.preproc.url.postag'

############################################################
# Embedding Files

_EMBEDDINGS_DIR   = _ROOT_DIR + 'embeddings'

W2V_TWITTER_EMB_GODIN = _EMBEDDINGS_DIR + '/twitter/word2vec_twitter_model.bin'
GAZET_EMB_ONE_CHECK   = _EMBEDDINGS_DIR + '/gazetteers/one.token.check.emb'

############################################################
# Global Tokens

URL_TOKEN   = '<URL>'
TAG_TOKEN   = '<TAG>'
PUNCT_TOKEN = '<PUNCT>'
EMOJI_TOKEN = '<EMOJI>'
UNK_TOKEN   = '<UNK>'
PAD_TOKEN   = '<PAD>'

#########################################################
PREDICTIONS_DIR = _ROOT_DIR + 'predictions/'
# PREDICTIONS_DIR = '/raid/data/gustavoag/ner/emnlp17/predictions/'

NN_PREDICTIONS  = PREDICTIONS_DIR + 'network.tsv'
CRF_PREDICTIONS = PREDICTIONS_DIR + 'crfsuite.tsv'

############################################################
def _test_paths():
    assert os.path.isdir(_DATA_DIR)
    assert os.path.isfile(TRAIN)
    assert os.path.isfile(DEV)
    assert os.path.isfile(TEST)

    print(TRAIN)
    print(DEV)
    print(TEST)

if __name__ == '__main__':
    _test_paths()
