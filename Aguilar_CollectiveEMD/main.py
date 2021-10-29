import sys
import numpy as np
seed_number = 42
np.random.seed(seed_number)
import time

from common import utilities as utils
from common import representation as rep
from models import network
from models import crf
from settings import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pickle
import pycrfsuite as pycrf

def main():
    ###############################
    ## LOADING DATA
    ###############################
    
    print ('Argument List:', str(sys.argv))
    global TEST
    TEST=TEST+str(sys.argv[1])
    
    TEST_POSTAG              = TEST  + '.postag'
    TEST_PREPROC_URL         = TEST  + '.preproc.url'
    TEST_PREPROC_URL_POSTAG  = TEST  + '.preproc.url.postag'
    
    print(TEST_PREPROC_URL)
    print(TEST_PREPROC_URL_POSTAG)
    
    start_time_train=time.time()
    ## TWEETS
    print("Loading tweets...")
    (tweets_train, labels_train), (tweets_test, labels_test) = utils.read_datasets(TEST_PREPROC_URL)

    ## POS TAGS
    print("Loading pos tags...")
    postag_train, postag_test = utils.read_and_sync_postags(tweets_train, tweets_test,TEST_PREPROC_URL_POSTAG)


    ###############################
    ## LOADING EMBEDDINGS
    ###############################

    ## TWITTER
    print("Loading twitter embeddings...")
    # twitter_embeddings, word2index = utils.read_twitter_embeddings(tweets_train + tweets_test)
    twitter_embeddings, word2index = utils.read_twitter_embeddings(tweets_train)

    ## GAZETTERS
    print("Loading gazetteers embeddings...")
    gaze_embeddings, gaze2index = utils.read_gazetteer_embeddings()


    ###############################
    ## GENERATING ENCODING
    ###############################
    start_time_test=time.time()
    print("Generating encodings...")
    ## WORDS (X)
    radius = 1
    x_word_twitter_train = rep.encode_tweets(word2index, tweets_train, radius)
    x_word_twitter_test  = rep.encode_tweets(word2index, tweets_test, radius)

    ## LABELS (Y)
    y_bin_train = rep.encode_bin_labels(labels_train)
    y_cat_train = rep.encode_cat_labels(labels_train)

    ## POS TAGS
    # index2postag = [PAD_TOKEN,UNK_TOKEN] + utils.get_uniq_elems(postag_train + postag_test)
    # print(postag_train[:5])
    index2postag = [PAD_TOKEN,UNK_TOKEN] + utils.get_uniq_elems(postag_train)
    x_postag_train = rep.encode_postags(index2postag, postag_train, radius)
    # 
    x_postag_test  = rep.encode_postags(index2postag, postag_test, radius)

    ## ORTHOGRAPHY
    ortho_dim = 30
    ortho_max_length = 20
    x_ortho_train = rep.encode_orthography(tweets_train, ortho_max_length)
    x_ortho_test  = rep.encode_orthography(tweets_test, ortho_max_length)

    ## GAZETTEERS
    x_gaze_train = rep.encode_gazetteers(gaze2index, tweets_train, radius)
    x_gaze_test  = rep.encode_gazetteers(gaze2index, tweets_test, radius)


    ###############################
    ## BUILD NEURAL NETWORK
    ###############################
    

    print("Building neural network...")
    char_inputs, char_encoded = network.get_char_cnn(ortho_max_length, len(rep.index2ortho), ortho_dim, 'char_ortho')
    word_inputs, word_encoded = network.get_word_blstm(len(index2postag), twitter_embeddings, window=radius*2+1, word_dim=100)
    gaze_inputs, gaze_encoded = network.get_gazetteers_dense(radius*2+1, gaze_embeddings)

    mtl_network = network.build_multitask_bin_cat_network(len(rep.index2category),      # number of category classes
                                                          char_inputs, char_encoded,    # char component (CNN)
                                                          word_inputs, word_encoded,    # word component (BLSTM)
                                                          gaze_inputs, gaze_encoded)    # gazetteer component (Dense)
    # mtl_network.summary()


    # ###############################
    # ## TRAIN NEURAL NETWORK
    # ###############################

    train_word_values = [x_word_twitter_train, x_postag_train]
    train_char_values = [x_ortho_train]
    train_gaze_values = [x_gaze_train]

    x_train_samples = train_gaze_values + train_char_values + train_word_values
    y_train_samples = {'bin_output': y_bin_train, 'cat_output': y_cat_train}

    network.train_multitask_net_with_split(mtl_network, x_train_samples, y_train_samples)
    
    fextractor = network.create_model_from_layer(mtl_network, layer_name='common_dense_layer')
    crf.train_with_fextractor(fextractor, x_train_samples, labels_train)
    
    # end_time_train=time.time()
    
    # print('train+preprocess time_taken:',(end_time_train-start_time_train))


    ###############################
    ## NN PREDICTIONS
    ###############################
    
    # mtl_network.load_weights("saved/My_Custom_Model.h5")

    x_test = [x_gaze_test, x_ortho_test, x_word_twitter_test, x_postag_test]
    inputs = gaze_inputs + char_inputs + word_inputs
    
    

    decoded_predictions = network.predict(mtl_network, inputs, x_test, rep.index2category)
    
    # print('prediction_length:',len(decoded_predictions))

    # print("Classification Report\n")
    # print(classification_report(utils.flatten(labels_test), decoded_predictions))
    # print()
    # print()
    # print("Confusion Matrix\n")
    # print(confusion_matrix(utils.flatten(labels_test), decoded_predictions))

    # Saving predictions in format: token\tlabel\tprediction
    # utils.save_predictions(NN_PREDICTIONS, tweets_test, labels_test, decoded_predictions)


    ###############################
    ## CRF PREDICTIONS
    ###############################
    
    fextractor = network.create_model_from_layer(mtl_network, layer_name='common_dense_layer')
    
    network_features, decoded_predictions = crf.predict_with_fextractor(fextractor, x_test)
    
    print(len(network_features),len(x_test),len(decoded_predictions))
    
    # gazetteer|orthographic|twitter-embedding|pos-tags
    print(len(x_test[0]),len(x_test[1]),len(x_test[2]),len(x_test[3]))
    print(len(network_features[0]))
    # print(type(network_features),type(x_test),type(decoded_predictions))
    
    
    # slice_size= 200
    # start=0
    # decoded_predictions=[]
    # for i in range(0, len(x_test), slice_size):
    #     if((i+slice_size)<=len(x_test)):
    #         curr_slice=x_test[i:(i+slice_size)]
    #     else:
    #         curr_slice= x_test[i:]
    #     decoded_predictions += crf.predict_with_fextractor(tagger, fextractor, curr_slice)
    #     print('end of iteration')
    
    # print(len(decoded_predictions))
    
    # for prediction in decoded_predictions:
    #     print(prediction)

    # print("Classification Report\n")
    # print(classification_report(utils.flatten(labels_test), decoded_predictions))
    # print()
    # print()
    # print("Confusion Matrix\n")
    # print(confusion_matrix(utils.flatten(labels_test), decoded_predictions))

    #TO ACTUALLY PRINT OUTPUTS--------- Saving predictions in format: token label prediction
    token_feature_tuple_list = utils.save_predictions(CRF_PREDICTIONS, tweets_test, labels_test, decoded_predictions, network_features)
    # utils.save_predictions_wo_eval(CRF_PREDICTIONS, tweets_test, labels_test, decoded_predictions)
    
    #FOR EFFICIENCY TEST
    # utils.save_predictions_wo_eval(CRF_PREDICTIONS, tweets_test, labels_test, decoded_predictions)
    
    end_time_test=time.time()
    print('test time_taken:',(end_time_test-start_time_test))
    
    return


if __name__ == '__main__':
    # print("running on the server?")
    main()



