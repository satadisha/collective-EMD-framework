import re
import string
import csv
import numpy as np
import matplotlib.pyplot as plt
import emoji
import pandas as pd
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import gutenberg
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTrainer

from collections import Counter
from collections import defaultdict as ddict
from embeddings.twitter.word2vecReader import Word2Vec
from itertools import groupby
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical # from keras.utils import to_categorical
from settings import *

##############################################################
# General Functions
##############################################################


gutenberg_text = ""
for file_id in gutenberg.fileids():
    gutenberg_text += gutenberg.raw(file_id)
tokenizer_trainer = PunktTrainer()
tokenizer_trainer.INCLUDE_ALL_COLLOCS = True
tokenizer_trainer.train(gutenberg_text)

my_sentence_tokenizer = PunktSentenceTokenizer(tokenizer_trainer.get_params())
my_sentence_tokenizer._params.abbrev_types.add('dr')
my_sentence_tokenizer._params.abbrev_types.add('c.j')
my_sentence_tokenizer._params.abbrev_types.add('u.s')
my_sentence_tokenizer._params.abbrev_types.add('u.s.a')
my_sentence_tokenizer._params.abbrev_types.add('ret.')
my_sentence_tokenizer._params.abbrev_types.add('rep.')
        
def unzip(list_of_tuples):
    return [list(elem) for elem in zip(*list_of_tuples)]

def flatten_rec(l):
    # TODO: fix problem with long lists (maximum recursion depth exceeded)
    if not l:
        return []
    if isinstance(l[0], list):
        return flatten(l[0]) + flatten(l[1:])
    return l[:1] + flatten(l[1:])

def flatten(l):
    """Flatten 2D lists"""
    return [i for sublist in l for i in sublist]

def remove_repeated_elements(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def get_uniq_elems(corpus):
    return list(set(flatten(corpus)))

##############################################################
# Input and output functions
##############################################################

def read_file_as_list_of_tuples(filename, delimiter='\t'):
    """It returns a list of tweets, and each tweet is a tuple of the elements found in the line"""
    with open(filename) as stream:
        reader = csv.reader(stream, delimiter=delimiter, quoting=csv.QUOTE_NONE)
        return [list(tuple(e) for e in g) for k, g in groupby(reader, lambda x: not x) if not k]

def read_file_as_lists(filename, delimiter='\t'):
    
    # f= open(filename,'r')
    # temp_read=f.read()
    # print('for tally:',len(temp_read.split('\n\n')))
    
    with open(filename) as stream:
        reader = csv.reader(stream, delimiter=delimiter, quoting=csv.QUOTE_NONE)
        labeled_tokens = [zip(*g) for k, g in groupby(reader, lambda x: not [s.strip() for s in x if s.strip()]) if not k]
        tokens, labels = zip(*labeled_tokens)
        return [list(t) for t in tokens], [list(l) for l in labels]

def read_test_tweets(filename, delimiter='\t'):
    with open(filename) as stream:
        reader = csv.reader(stream, delimiter=delimiter, quoting=csv.QUOTE_NONE)
        tweets = [list(g) for k, g in groupby(reader, lambda x: not [s.strip() for s in x if s.strip()]) if not k]
        return [flatten(tokens) for tokens in tweets]
    
def read_embeddings(fname, term2index, index2term, embeddings, sep=' ', skipfirst=False):
    with open(fname) as stream:
        if skipfirst:
            next(stream)
        for line in stream:
            # print(line)
            term2vec = line.strip().split(sep)
            term2index[term2vec[0]] = len(term2index)
            index2term.append(term2vec[0])
            embeddings = np.append(embeddings, np.array([term2vec[1:]], dtype=np.float32), axis=0)
        return term2index, index2term, embeddings
    
def pick_embeddings_from_file(filename, vocabulary, sep=' ', skipfirst=False):
    embeddings = []
    index2term = []
    reduced_vocab = {v:i for i,v in enumerate(list(vocabulary))} # To avoid removing elements from the outer list
    with open(filename) as stream:
        if skipfirst:
            next(stream)
        for line in stream:
            term2vec = line.strip().split(sep)
            if term2vec[0] in reduced_vocab:
                del reduced_vocab[term2vec[0]] # reduce the lookup space
                index2term.append(term2vec[0]) 
                embeddings.append(np.array(term2vec[1:], dtype=np.float32))
            if not reduced_vocab:
                break
    return index2term, np.array(embeddings)

def pick_embeddings(vocabulary, index2term, embeddings):
    new_embeddings = []
    new_index2term = []
    reduced_vocab = list(vocabulary) # To avoid removing elements from the outer list
    for index,term in enumerate(index2term):
        if term in reduced_vocab:
            reduced_vocab.pop(reduced_vocab.index(term)) # reduce the lookup space
            new_index2term.append(term) 
            new_embeddings.append(embeddings[index])
        if not reduced_vocab:
            break
    return new_index2term, np.array(new_embeddings)
    
def pick_embeddings_by_indexes(vocabulary, embeddings, term2index):
    embeds, index2term = zip(*[(embeddings[term2index.get(token)], token) 
                               for token in vocabulary 
                               if term2index.get(token)])
    return list(index2term), np.array(embeds)

def left_join_embeddings(vocab, ind2word_1, ind2word_2, embeddings_1, embeddings_2):
    embeddings = []
    index2word = []
    for word in vocab:
        if word in ind2word_1:
            index2word.append(word)
            embeddings.append(embeddings_1[ind2word_1.index(word)])
        elif word in ind2word_2:
            index2word.append(word)
            embeddings.append(embeddings_2[ind2word_2.index(word)])
    return index2word, np.array(embeddings)

def write_file(filename, dataset, delimiter='\t'):
    """dataset is a list of tweets where each token can be a tuple of n elements"""
    with open(filename, '+w') as stream:
        writer = csv.writer(stream, delimiter=delimiter, quoting=csv.QUOTE_NONE, quotechar='')
        for tweet in dataset:
            writer.writerow(list(tweet))

def save_file(filename, tweets, labels):
    """save a file with token, label and prediction in each row"""
    dataset = []
    for n, tweet in enumerate(tweets):
        tweet_data = list(zip(tweet, labels[n])) + [()]
        dataset += tweet_data 
    write_file(filename, dataset)

def write_encoded_tweets(filename, tweets):
    dataset = []
    for n, tweet in enumerate(tweets):
        dataset += tweet + [()]
    write_file(filename, dataset)

# def get_entities(word_tag_tuples):
    
#     mentions=[]
#     candidateMention=''
#     positions=[]
    
#     #emoji.get_emoji_regexp().sub(u'', candidateMention)
#     for tup in word_tag_tuples:
#         candidate=tup[0]
#         tag=tup[1]
#         if(tag=='O'):
#             if(candidateMention):
#                 if((not candidateMention.strip().startswith('#'))&(not candidateMention.strip().startswith('@'))&(not candidateMention.strip().startswith('https:'))):
#                     mention_to_add=emoji.get_emoji_regexp().sub(u'', candidateMention).strip(string.punctuation).lower().strip()
#                     if mention_to_add.endswith("'s"):
#                         li = mention_to_add.rsplit("'s", 1)
#                         mention_to_add=''.join(li)
#                     elif mention_to_add.endswith("’s"):
#                         li = mention_to_add.rsplit("’s", 1)
#                         mention_to_add=''.join(li)
#                     else:
#                         mention_to_add=mention_to_add
#                     if(mention_to_add!=''):
#                         mentions.append(mention_to_add)
#             candidateMention=''
#         else:
#             if (tag=='B'):
#                 if((not candidateMention.strip().startswith('#'))&(not candidateMention.strip().startswith('@'))&(not candidateMention.strip().startswith('https:'))):
#                     mention_to_add=emoji.get_emoji_regexp().sub(u'', candidateMention).strip(string.punctuation).lower().strip()
#                     if mention_to_add.endswith("'s"):
#                         li = mention_to_add.rsplit("'s", 1)
#                         mention_to_add=''.join(li)
#                     elif mention_to_add.endswith("’s"):
#                         li = mention_to_add.rsplit("’s", 1)
#                         mention_to_add=''.join(li)
#                     else:
#                         mention_to_add=mention_to_add
#                     if(mention_to_add!=''):
#                         mentions.append(mention_to_add)
#                 candidateMention=candidate
#             else:
#                 candidateMention+=" "+candidate
#         # if (tag=='B'):
#         #     if((not candidateMention.strip().startswith('#'))&(not candidateMention.strip().startswith('@'))):
#         #         mention_to_add=emoji.get_emoji_regexp().sub(u'', candidateMention).strip(string.punctuation).lower().strip()
#         #         if(mention_to_add):
#         #             mentions.append(mention_to_add)
#         #     candidateMention=candidate
#         # else:
#         #     candidateMention+=" "+candidate
#     if(emoji.get_emoji_regexp().sub(u'', candidateMention).strip(string.punctuation).strip()):
#         if((not candidateMention.strip().startswith('#'))&(not candidateMention.strip().startswith('@'))&(not candidateMention.strip().startswith('https:'))):
#             mention_to_add=emoji.get_emoji_regexp().sub(u'', candidateMention).strip(string.punctuation).lower().strip()
#             if(mention_to_add!=''):
#                 mentions.append(mention_to_add)
#         # mentions.append(emoji.get_emoji_regexp().sub(u'', candidateMention).strip(string.punctuation).lower().strip())
#     # print('extracted mentions:', mentions)
#     return mentions

def get_entities(word_tag_tuples):
    mentions=[]
    candidateMention=''
    positions=[]
    apostrophe_list =["'s",'’s','s']
    
    #emoji.get_emoji_regexp().sub(u'', candidateMention)
    for index, tup in enumerate(word_tag_tuples):
        candidate=tup[0]
        tag=tup[1]
        if(tag=='O'):
            if(candidateMention):
                if((not candidateMention.strip().startswith('#'))&(not candidateMention.strip().startswith('@'))&(not candidateMention.strip().startswith('https:'))):
                    mention_to_add=emoji.get_emoji_regexp().sub(u'', candidateMention).strip(string.punctuation).lower().strip()
                    if mention_to_add.endswith("'s"):
                        li = mention_to_add.rsplit("'s", 1)
                        mention_to_add=''.join(li)
                    elif mention_to_add.endswith("’s"):
                        li = mention_to_add.rsplit("’s", 1)
                        mention_to_add=''.join(li)
                    else:
                        mention_to_add=mention_to_add
                    if(mention_to_add!=''):
                        try:
                            assert len(mention_to_add.split()) == len(positions)
                            mentions.append((mention_to_add,positions))
                        except AssertionError:
                            print(word_tag_tuples)
                            print(mention_to_add,positions)
                            return
            candidateMention=''
            positions=[]
        else:
            if (tag=='B'):
                if((not candidateMention.strip().startswith('#'))&(not candidateMention.strip().startswith('@'))&(not candidateMention.strip().startswith('https:'))):
                    mention_to_add=emoji.get_emoji_regexp().sub(u'', candidateMention).strip(string.punctuation).lower().strip()
                    if mention_to_add.endswith("'s"):
                        li = mention_to_add.rsplit("'s", 1)
                        mention_to_add=''.join(li)
                    elif mention_to_add.endswith("’s"):
                        li = mention_to_add.rsplit("’s", 1)
                        mention_to_add=''.join(li)
                    else:
                        mention_to_add=mention_to_add
                    if(mention_to_add!=''):
                        try:
                            assert len(mention_to_add.split()) == len(positions)
                            mentions.append((mention_to_add,positions))
                        except AssertionError:
                            print(word_tag_tuples)
                            print(mention_to_add,positions)
                            return
                if((candidate.strip() not in string.punctuation)&(emoji.get_emoji_regexp().sub(u'', candidate).strip(string.punctuation).lower().strip()!='')&(candidate.strip().strip(string.punctuation) not in apostrophe_list)):
                    candidateMention=candidate
                    positions=[index]
            else:
                if((candidate.strip() not in string.punctuation)&(emoji.get_emoji_regexp().sub(u'', candidate).strip(string.punctuation).lower().strip()!='')&(candidate.strip().strip(string.punctuation) not in apostrophe_list)):
                    candidateMention+=" "+candidate
                    positions.append(index)
        # if (tag=='B'):
        #     if((not candidateMention.strip().startswith('#'))&(not candidateMention.strip().startswith('@'))):
        #         mention_to_add=emoji.get_emoji_regexp().sub(u'', candidateMention).strip(string.punctuation).lower().strip()
        #         if(mention_to_add):
        #             mentions.append(mention_to_add)
        #     candidateMention=candidate
        # else:
        #     candidateMention+=" "+candidate
    if(emoji.get_emoji_regexp().sub(u'', candidateMention).strip(string.punctuation).strip()):
        if((not candidateMention.strip().startswith('#'))&(not candidateMention.strip().startswith('@'))&(not candidateMention.strip().startswith('https:'))&(candidate.strip().strip(string.punctuation) not in apostrophe_list)):
            mention_to_add=emoji.get_emoji_regexp().sub(u'', candidateMention).strip(string.punctuation).lower().strip()
            if(mention_to_add!=''):
                try:
                    assert len(mention_to_add.split()) == len(positions)
                    mentions.append((mention_to_add,positions))
                except AssertionError:
                    print(word_tag_tuples)
                    print(mention_to_add,positions)
                    return
        # mentions.append(emoji.get_emoji_regexp().sub(u'', candidateMention).strip(string.punctuation).lower().strip())
    # print('extracted mentions:', mentions)
    return mentions

def custom_flatten(mylist, outlist,ignore_types=(str, bytes, int)):
    
    if (mylist !=[]):
        for item in mylist:
            #print not isinstance(item, ne.NE_candidate)
            if isinstance(item, list) and not isinstance(item, ignore_types):
                custom_flatten(item, outlist)
            else:
                item=item.strip(' \t\n\r')
                outlist.append(item)
    return outlist
    
def getWordsII(sentence):
    tweetWordList=[]
    if(sentence):
        tempList=[]
        tempWordList=sentence.split()
        p_dots= re.compile(r'[.]{2,}')
        #print(tempWordList)
        for word in tempWordList:
            if (list(p_dots.finditer(word))):
                matched_spans= list(p_dots.finditer(word)) 
                temp=[]
                next_string_start=0
                for matched_span in matched_spans:
                    matched_start=matched_span.span()[0]
                    this_excerpt=word[next_string_start:matched_start]
                    if(this_excerpt):
                        temp.append(this_excerpt)
                    next_string_start=matched_span.span()[1]
                if(next_string_start<len(word)):
                    last_excerpt=word[next_string_start:]
                    if(last_excerpt):
                        temp.append(last_excerpt)
    #             print(temp)
            elif((word.count('.')==1)&(word.endswith('.'))):
                words=list(filter(lambda elem: elem!='',re.split("(\.)",word)))
                temp=[]
                for token in words:
                    if(token!='.'):
                        temp+=list(filter(lambda elem: elem!='',re.split('([^a-zA-Záéíó@#’\'0-9])',token)))
                    else:
                        temp.append('.')
            else:
                temp=list(filter(lambda elem: elem!='',re.split('([^a-zA-Záéíó@.#’\'0-9])',word)))
            if(temp):
                tempList.append(temp)
        tweetWordList=custom_flatten(tempList,[])
    return tweetWordList
    
def normalize_to_sentences(text):
    # re_url = re.compile(r"((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\
    #                     .([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*",
    #                     re.MULTILINE|re.UNICODE)
    # # replace ips
    # re_ip = re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")

    # # replace URLs
    # text = re_url.sub("URL", text)
    # # replace IPs
    # text = re_ip.sub("IPADDRESS", text)
    
    
    tweetSentences=list(filter (lambda sentence: len(sentence)>1, text.split('\n')))
    tweetSentenceList_inter=custom_flatten(list(map(lambda sentText: my_sentence_tokenizer.tokenize(sentText.lstrip().rstrip()),tweetSentences)),[])
    tweetSentenceList=list(filter (lambda sentence: len(sentence)>1, tweetSentenceList_inter))

    # print(text)
    # setup tokenizer
    #tokenizer = WordPunctTokenizer()

    # sent_text = [sent for sent in nltk.tokenize.sent_tokenize(text)] # this gives us a list of sentences
    # sent_text = []
    # for sentence in tweetSentenceList:
    #     # for sentence in sentences:
    #     sent_text.append(getWordsII(sentence))
        # sent_text.append(nltk.tokenize.word_tokenize(sentence))
    #print('normalizing to sentences: ', tweetSentenceList)
    return tweetSentenceList
    
def save_predictions_wo_eval(filename, tweetsentences, labels, predictions, network_features):
    tweet_to_sentences_w_annotation={}
    sentenceID=0
    test=pd.read_csv("data/deduplicated/deduplicated_test.csv",sep =';',keep_default_na=False)
    all_detected_ne=[]
    sentence_df_dict={} 
    sentence_to_dfIndex_dictKey=0
    
    for row in test.itertuples():
        tweetID=str(row.Index)
        text=str(row.TweetText)
        row_sentences=[]
        row_sentences += normalize_to_sentences(text)
        # print(text)
        
        mentions=[]
        
        # if(row_sentences):
        tweet_to_sentences_w_annotation[tweetID]=((sentenceID,sentenceID+len(row_sentences)),mentions)
        sentenceID+=len(row_sentences)
        
        for sentID, sentence in enumerate(row_sentences):
            sentence_df_dict[sentence_to_dfIndex_dictKey] = (tweetID,sentID,sentence)
            sentence_to_dfIndex_dictKey +=1
        # else:
        #     tweet_to_sentences_w_annotation[tweetID]=((sentenceID,sentenceID+1),mentions)
        #     sentenceID+=1
        # print(sentenceID,len(row_sentences))
    
    dataset, i = [], 0
    ner_arrays=[]
    file_write_text=''
    # token_feature_tuple_list=[]
    
    for n, tweet in enumerate(tweetsentences):
        tweet_data = list(zip(tweet, labels[n], predictions[i:i + len(tweet)]))
        word_tag_tuples=zip(tweet,predictions[i:i + len(tweet)])
        token_feature_tuple = list(zip(tweet,network_features[i:i + len(tweet)]))
        entitiesWPositions_from_sentence=get_entities(word_tag_tuples)
        entities_from_sentence=[tup[0] for tup in entitiesWPositions_from_sentence]
        
        # print(sentence_df_dict[n])
        # print(entitiesWPositions_from_sentence)
        sentence_df_dict[n] = sentence_df_dict[n]+(entitiesWPositions_from_sentence,token_feature_tuple)
        # line_text='\t'.join(entities_from_sentence)
        # file_write_text+=line_text+'\n'
        all_detected_ne.extend(entities_from_sentence)
        ner_arrays.append(entities_from_sentence)
        # token_feature_tuple_list.append(token_feature_tuple)
        i += len(tweet)
        dataset += tweet_data + [()]
    
    print('tally:',sentenceID,len(tweetsentences),len(ner_arrays))
    return sentence_df_dict, tweet_to_sentences_w_annotation

def postprocess_for_sts_dataset(filename, tweetsentences, labels, predictions, network_features):
    
    i = 0
    tweet_to_sentences_w_annotation={}
    sentence_df_dict = {}
    
    for n, tweet in enumerate(tweetsentences):
        
        tweet_to_sentences_w_annotation[n] = ((i,i+1),[])
        tweet_data = list(zip(tweet, labels[n], predictions[i:i + len(tweet)]))
        word_tag_tuples=zip(tweet,predictions[i:i + len(tweet)])
        token_feature_tuple = list(zip(tweet,network_features[i:i + len(tweet)]))
        entitiesWPositions_from_sentence=get_entities(word_tag_tuples)
        entities_from_sentence=[tup[0] for tup in entitiesWPositions_from_sentence]
        
        sentence_df_dict[n] = (n,n,tweet,entitiesWPositions_from_sentence,token_feature_tuple)
        # line_text='\t'.join(entities_from_sentence)
        # file_write_text+=line_text+'\n'
        # all_detected_ne.extend(entities_from_sentence)
        # ner_arrays.append(entities_from_sentence)
        # token_feature_tuple_list.append(token_feature_tuple)
        i += len(tweet)
    
    return sentence_df_dict, tweet_to_sentences_w_annotation
    
def save_predictions(filename, tweetsentences, labels, predictions, network_features):
    """save a file with token, label and prediction in each row"""
    tweet_to_sentences_w_annotation={}
    sentenceID=0
    
    # test=pd.read_csv("data/venezuela/venezuela.csv",sep =',',keep_default_na=False)
    # outputfilename="data/venezuela/venezuela.txt"
    
    # test=pd.read_csv("data/tweets_3K/tweets_3k_annotated.csv",sep =',',keep_default_na=False)
    # outputfilename="data/tweets_3K/tweets_3k_annotated.txt"
    
    test=pd.read_csv("data/covid/covid_2K.csv",sep =',',keep_default_na=False)
    outputfilename="data/covid/covid.txt"
    
    # test=pd.read_csv("data/roevwade/roevwade.csv",sep =',',keep_default_na=False)
    # outputfilename="data/roevwade/roevwade.txt"
    
    # test=pd.read_csv("data/billdeblasio/billdeblasio.csv",sep =',',keep_default_na=False)
    # outputfilename="data/billdeblasio/billdeblasio.txt"
    
    # test=pd.read_csv("data/billnye/billnye.csv",sep =',',keep_default_na=False)
    # outputfilename="data/billnye/billnye.txt"
    
    # test=pd.read_csv("data/ripcity/ripcity.csv",sep =',',keep_default_na=False)
    # outputfilename="data/ripcity/ripcity.txt"
    
    # test=pd.read_csv("data/pikapika/pikapika.csv",sep =',',keep_default_na=False)
    # outputfilename="data/pikapika/pikapika.txt"
    
    # test=pd.read_csv("data/wnut-test/wnut17test.csv",sep =',',keep_default_na=False)
    # outputfilename="data/wnut-test/wnut17test.txt"
    
    # test=pd.read_csv("data/broad_twitter_corpus/broad_twitter_corpus.csv",sep =',',keep_default_na=False)
    # outputfilename="data/broad_twitter_corpus/broad_twitter_corpus.txt"
    
    all_detected_ne=[]
    all_annotated_ne=[]
    sentence_df_dict={} 
    sentence_to_dfIndex_dictKey=0
    
    for row in test.itertuples():
        tweetID=str(row.Index)
        text=str(row.TweetText)
        row_sentences=[]
        row_sentences += normalize_to_sentences(text)
        # print(text)
        
        mentions=[]
        for sentence_level in str(row.mentions_other).split(';'):
            if(sentence_level):
                for mention in sentence_level.split(','):
                    if(mention):
                        mentions.append(mention.lower().strip(string.punctuation).strip())
        mentions=list(filter(lambda element: ((element !='')&(element !='nan')), mentions))
        all_annotated_ne.extend(mentions)
        
        # if(row_sentences):
        tweet_to_sentences_w_annotation[tweetID]=((sentenceID,sentenceID+len(row_sentences)),mentions)
        sentenceID+=len(row_sentences)
        
        for sentID, sentence in enumerate(row_sentences):
            sentence_df_dict[sentence_to_dfIndex_dictKey] = (tweetID,sentID,sentence)
            sentence_to_dfIndex_dictKey +=1
        # else:
        #     tweet_to_sentences_w_annotation[tweetID]=((sentenceID,sentenceID+1),mentions)
        #     sentenceID+=1
        # print(sentenceID,len(row_sentences))
    print('tally:',sentence_to_dfIndex_dictKey, len(tweetsentences))
    dataset, i = [], 0
    ner_arrays=[]
    file_write_text=''
    # token_feature_tuple_list=[]
    
    for n, tweet in enumerate(tweetsentences):
        tweet_data = list(zip(tweet, labels[n], predictions[i:i + len(tweet)]))
        word_tag_tuples=zip(tweet,predictions[i:i + len(tweet)])
        token_feature_tuple = list(zip(tweet,network_features[i:i + len(tweet)]))
        entitiesWPositions_from_sentence=get_entities(word_tag_tuples)
        entities_from_sentence=[tup[0] for tup in entitiesWPositions_from_sentence]
        
        sentence_df_dict[n] = sentence_df_dict[n]+(entitiesWPositions_from_sentence,token_feature_tuple)
        # line_text='\t'.join(entities_from_sentence)
        # file_write_text+=line_text+'\n'
        all_detected_ne.extend(entities_from_sentence)
        ner_arrays.append(entities_from_sentence)
        # token_feature_tuple_list.append(token_feature_tuple)
        i += len(tweet)
        dataset += tweet_data + [()]
    
    print('tally:',sentenceID,len(tweetsentences),len(ner_arrays))
    system_output_mention_list=list(set(all_detected_ne))
    file_write_text='\n'.join(system_output_mention_list)
    f1= open(outputfilename, "w")
    f1.write(file_write_text)
    f1.close()
    
    true_positive_count=0
    false_positive_count=0
    false_negative_count=0
    total_mentions=0
    total_annotation=0
    
    # output_entities=set(all_detected_ne)
    # annotated_entities=set(all_annotated_ne)
    
    # print(all_detected_ne)
    # print(all_annotated_ne)
    
    # print(output_entities)
    # print(annotated_entities)
    
    # true_positive_set=set(output_entities & annotated_entities)
    # true_positive_count=len(list(true_positive_set))
    # false_positive_count=len(list(output_entities-true_positive_set))
    # false_negative_count=len(list(annotated_entities-true_positive_set))
    
    for tweetID in tweet_to_sentences_w_annotation.keys():
        unrecovered_annotated_mention_list=[]
        tp_counter_inner=0
        fp_counter_inner=0
        fn_counter_inner=0
        
        annotated_mention_list=tweet_to_sentences_w_annotation[tweetID][1]
        output_mentions_list=[]
        idRange=tweet_to_sentences_w_annotation[tweetID][0]
        for sentID in range(idRange[0],idRange[1]):
            output_mentions_list+=ner_arrays[sentID]
        # print(tweetID,annotated_mention_list,output_mentions_list)
        all_postitive_counter_inner=len(output_mentions_list)
        while(annotated_mention_list):
            if(len(output_mentions_list)):
                annotated_candidate= annotated_mention_list.pop()
                if(annotated_candidate in output_mentions_list):
                    output_mentions_list.pop(output_mentions_list.index(annotated_candidate))
                    tp_counter_inner+=1
                else:
                    unrecovered_annotated_mention_list.append(annotated_candidate)
            else:
                unrecovered_annotated_mention_list.extend(annotated_mention_list)
                break
        # unrecovered_annotated_mention_list_outer.extend(unrecovered_annotated_mention_list)
        fn_counter_inner=len(unrecovered_annotated_mention_list)
        fp_counter_inner=all_postitive_counter_inner- tp_counter_inner
        
        # print(tp_counter_inner,fp_counter_inner,fn_counter_inner)
        
        true_positive_count+=tp_counter_inner
        false_positive_count+=fp_counter_inner
        false_negative_count+=fn_counter_inner
        
    print('true_positive_count,false_positive_count,false_negative_count:')
    print(true_positive_count,false_positive_count,false_negative_count)
    
    precision=(true_positive_count)/(true_positive_count+false_positive_count)
    recall=(true_positive_count)/(true_positive_count+false_negative_count)
    f_measure=2*(precision*recall)/(precision+recall)
            
    print('precision: ',precision)
    print('recall: ',recall)
    print('f_measure: ',f_measure)
    return sentence_df_dict, tweet_to_sentences_w_annotation

def save_final_predictions(filename, tweets, predictions):
    """save a file with token and its prediction in each row"""
    dataset, i = [], 0
    for n, tweet in enumerate(tweets):
        tweet_data = list(zip(tweet, predictions[i:i + len(tweet)]))
        i += len(tweet)
        dataset += tweet_data + [()]
    write_file(filename, dataset)

def read_datasets():
    tweets_train, labels_train = read_file_as_lists(TRAIN_PREPROC_URL)
    tweets_dev,   labels_dev   = read_file_as_lists(DEV_PREPROC_URL)
    # tweets_test,  labels_test  = read_file_as_lists(TEST_PREPROC_URL)

    # Combining train and dev to account for different domains
    tweets_train += tweets_dev
    labels_train += labels_dev

    return (tweets_train, labels_train) 
    
def read_datasets_test(TEST_PREPROC_URL):
    tweets_test,  labels_test  = read_file_as_lists(TEST_PREPROC_URL)
    return (tweets_test, labels_test)


def read_and_sync_postags(tweets_train):
    pos_tweets_train, pos_labels_train = read_file_as_lists(TRAIN_PREPROC_URL_POSTAG)
    pos_tweets_dev,   pos_labels_dev   = read_file_as_lists(DEV_PREPROC_URL_POSTAG)
    # pos_tweets_test,  pos_labels_test  = read_file_as_lists(TEST_PREPROC_URL_POSTAG)

    # Combining train and dev to account for different domains
    pos_tweets_train += pos_tweets_dev
    pos_labels_train += pos_labels_dev

    # Standarizing tokenization between postags and original tweets
    sync_postags_and_tweets(tweets_train, pos_tweets_train, pos_labels_train)
    # sync_postags_and_tweets(tweets_test, pos_tweets_test, pos_labels_test)

    return pos_labels_train
    
def read_and_sync_postags_test(tweets_test, TEST_PREPROC_URL_POSTAG):
    pos_tweets_test,  pos_labels_test  = read_file_as_lists(TEST_PREPROC_URL_POSTAG)
    
    # Standarizing tokenization between postags and original tweets
    sync_postags_and_tweets(tweets_test, pos_tweets_test, pos_labels_test)
    return pos_labels_test
    


def read_twitter_embeddings(corpus):
    w2v_model = Word2Vec.load_word2vec_format(W2V_TWITTER_EMB_GODIN, binary=True)
    w2v_vocab = {token: v.index for token, v in w2v_model.vocab.items()}

    # Using only needed embeddings (faster this way)
    index2word, embeddings = pick_embeddings_by_indexes(get_uniq_elems(corpus), w2v_model.syn0, w2v_vocab)

    index2word = [PAD_TOKEN, UNK_TOKEN] + index2word
    word2index = ddict(lambda: index2word.index(UNK_TOKEN), {w: i for i, w in enumerate(index2word)})
    embeddings = np.append(np.zeros((2, embeddings.shape[1])), embeddings, axis=0)

    return embeddings, word2index


def read_gazetteer_embeddings():
    gazetteers = read_file_as_list_of_tuples(GAZET_EMB_ONE_CHECK)[0]
    index2gaze, embeddings = zip(*[(data[0], data[1:]) for data in gazetteers])

    index2gaze = [UNK_TOKEN, PAD_TOKEN] + list(index2gaze)
    gaze2index = ddict(lambda: index2gaze.index(UNK_TOKEN), {g: i for i, g in enumerate(index2gaze)})
    embeddings = np.append(np.zeros((2, 6)), embeddings, axis=0)

    return embeddings, gaze2index


def show_training_loss_plot(hist):
    # TODO: save the resulting plot
    train_loss = hist.history["loss"]
    val_loss = hist.history["val_loss"]
    plt.plot(range(len(train_loss)), train_loss, color="red", label="Train Loss")
    plt.plot(range(len(train_loss)), val_loss, color="blue", label="Validation Loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend(loc="best")
    plt.show()

##############################################################
# Representation utilities
##############################################################

def tokenize_tweets(tweets, char_level=False, lower=False, filters=''):
    tokenizer = Tokenizer(filters=filters, lower=lower, char_level=char_level)
    tokenizer.fit_on_texts([' '.join(t) for t in tweets])
    return tokenizer

def get_max_word_length(tweets):
    return max(map(len, flatten(tweets)))

def element2index_dict(elems, offset=0):
    counter = Counter(elems) 
    sorted_elems = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    return dict({e[0]: (i+offset) for i,e in enumerate(sorted_elems)})

def encode_tokens(token2index, tokens):
    return [[token2index[tkn] for tkn in tkns] for tkns in tokens]

def decode_predictions(predictions, idx2label):
    return [idx2label[pred] for pred in predictions]

def build_x_matrix(w, encodings, pad_idx=0):
    """
    w: window size for context
    encodings: list of lists, i.e. each tweet contains a list of tokens;
    return matrix: whose rows (of length w * 2 + 1) represent the tokens
    """
    x_matrix = []
    for context in encodings:
        for i, enc in enumerate(context):
            # Left side of the target word
            lower = max(i - w, 0)
            left = [pad_idx] * (w - (i - lower)) + context[lower:i]

            # Right side of the target word
            upper = min(i + w + 1, len(context))
            right = context[i:upper] + [pad_idx] * (w + 1 - (upper - i))

            # The whole vector (row)
            x_matrix.append(left + right)
    return np.array(x_matrix)

def build_sided_matrix(w, side, encodings, pad_idx=0):
    """
    side can be either 'left' or 'right'
    """
    x_matrix = []
    for context in encodings:
        for i, enc in enumerate(context):
            # Left side of the target word
            if side == 'left':
                lower = max(i - w, 0)
                left = [pad_idx] * (w - (i - lower)) + context[lower:i+1]
                x_matrix.append(left)
                
            # Right side of the target word
            elif side == 'right':
                upper = min(i + w + 1, len(context))
                right = context[i:upper] + [pad_idx] * (w + 1 - (upper - i))
                x_matrix.append(right)
    return np.array(x_matrix)


def vectorize_labels(labels, index2label=None):
    """labels: list of lists, i.e. each tweet has a list of labels"""
    flat_labels = flatten(labels)
    if not index2label:
        label_set = list(set(flat_labels))
        index2label = dict(enumerate(label_set))
    label2index = dict((l, i) for i, l in enumerate(index2label))

    y = [label2index[l] for l in flat_labels]
    y = to_categorical(np.array(y, dtype='int32'))
    return y, label2index, index2label

def build_embedding_matrix(w2v, dim, tokenizer):
    print("Length of word_index:", len(tokenizer.word_index))
    embedding_matrix = [w2v.syn0norm[w2v.vocab.get(word).index, :dim]
                        if w2v.vocab.get(word)
                        else np.zeros(dim)
                        for word, i in tokenizer.word_index.items()]
    print(len(embedding_matrix))
    print(embedding_matrix)
    return np.array(embedding_matrix, dtype='float32')

def orthigraphic_char(ch):
    try:
        if re.match('[a-z]', ch):
            return 'c'
        if re.match('[A-Z]', ch):
            return 'C'
        if re.match('[0-9]', ch):
            return 'n'
        if ch in string.punctuation:
            return 'p'
    except TypeError:
        print('TypeError:',ch)
    return 'x'
    
def orthographic_tweet(tweet):
    return [''.join([orthigraphic_char(ch) for ch in token]) for token in tweet]
    
def orthographic_mapping(tweets):
    return [orthographic_tweet(tweet) for tweet in tweets]

def match_up_to(x, elems):
    # print(x)
    # print(elems)
    acc = [] 
    for e in elems: 
        acc.append(e)
        if x == ''.join(acc):
            # print(acc)
            return len(acc)
    return None
        
def map_equivalent(t):
    equivalences = [('&lt;', '<'), ('&amp;', '&'), ('&gt;', '>'),('&quot;', "\"")]
    for a, b in equivalences:
        if a in t:
            return t.replace(a, b)
    return t

def map_to_iob_labels(labels):
    return [[lbl[0] for lbl in lbls] for lbls in labels]

def sync_postags_and_tweets(tweets, pos_tweets, pos_labels):
    print(len(tweets),len(pos_tweets))
    for row in range(len(tweets)):
        for pos in range(len(tweets[row])):
            t = tweets[row][pos]
            p = pos_tweets[row][pos]
            # print(t)
            # print(p)

            if t != p:
                t = map_equivalent(t)
                up_to = match_up_to(t, pos_tweets[row][pos:])
                
                assert up_to and up_to > 0, "Inconsistency: {} not in {}".format(p, t)
                
                pos_chunk = remove_repeated_elements(pos_labels[row][pos:pos+up_to])
                
                del pos_tweets[row][pos+1:pos+up_to]
                del pos_labels[row][pos+1:pos+up_to]
                
                pos_tweets[row][pos] = t
                pos_labels[row][pos] = ''.join(pos_chunk)
                tweets[row][pos] = t
                
        assert len(tweets[row]) == len(pos_tweets[row]), "\n{}\n{}".format(tweets[row], pos_tweets[row])
        assert len(pos_tweets[row]) == len(pos_labels[row]), "{}\n{}".format(pos_tweets[row], pos_labels[row])


##############################################################
# Metrics functions
##############################################################
def fbeta_score(y_true, y_pred, beta=1):
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F score at 0.
    if c3 == 0:
        return 0.0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Weight precision and recall together as a single scalar.
    beta2 = beta ** 2
    f_score = (1 + beta2) * (precision * recall) / (beta2 * precision + recall)
    return f_score




