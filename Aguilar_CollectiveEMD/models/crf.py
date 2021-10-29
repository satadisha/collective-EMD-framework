import pycrfsuite as crf

from pycrfsuite import ItemSequence
from common import utilities as utils

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

def _get_xseq(model, matrix):
    features= model.predict(matrix)
    print('got features...')
    xseq = [{'feat{}'.format(i):float(w) for i,w in enumerate(list(feature))}
            for feature
            in features]
    print('got xseq...')
    # return ItemSequence(xseq)
    return xseq


def train_with_fextractor(nn_model, x_train, y_train):
    # nn_model = Model(inputs=model.input, outputs=model.get_layer('common_dense_layer').output)
    # x_train = [x_gaze_train,
    #            x_ortho_twitter_train,
    #            x_word_twitter_train,
    #            x_postag_train]

    xseq_train = _get_xseq(nn_model, x_train)
    yseq_train = utils.flatten(y_train)
    
    # pycrfsuite portion

    trainer = crf.Trainer(verbose=False)
    trainer.append(xseq_train, yseq_train)
    trainer.set_params({
        'c1': 1.0,                            # L1 penalty
        'c2': 1e-3,                           # L2 penalty
        'max_iterations': 100,                # stop earlier
        'feature.possible_transitions': True  # possible transitions, but not observed
    })
    trainer.train('weights.pycrfsuite')
    
    # clf = sklearn_crfsuite.CRF(
    # algorithm='lbfgs',
    # c1=0.1,
    # c2=1e-3, #0.1,
    # max_iterations=100,
    # all_possible_transitions=True)
    # clf.fit(xseq_train, yseq_train)
    
    # with open('saved/crf_model.pkl','wb') as f:
    #     pickle.dump(clf,f)


def predict_with_fextractor(nn_model, x_test):
    # x_test = [x_gaze_test,
    #           x_ortho_twitter_test,
    #           x_word_twitter_test,
    #           x_postag_test]
    
    
    print('starting crf predictor')

    # pycrfsuite portion
    tagger = crf.Tagger()
    tagger.open('weights.pycrfsuite')
    
    # Predicting test data
    xseq=_get_xseq(nn_model, x_test)
    decoded_predictions=[]
    slice_size= 200
    for i in range(0, len(xseq), slice_size):
        if((i+slice_size)<=len(xseq)):
            curr_slice=xseq[i:(i+slice_size)]
        else:
            curr_slice= xseq[i:]
        itemseq=ItemSequence(curr_slice)
        # print('got itemseq...')
        
        decoded_predictions += tagger.tag(itemseq)
    # decoded_predictions = tagger.tag(itemseq)
    
    # with open('saved/crf_model.pkl', 'rb') as f:
    #     clf = pickle.load(f)
    
    # decoded_predictions = clf.predict(_get_xseq(nn_model, x_test))
    print('crf prediction done')
    return xseq, decoded_predictions






