capitalized=0
start_of_sentence=1
abbreviation=2
all_capitalized=3
is_csl=4
title=5
has_number=6
date_indicator=7
is_apostrophed=8 #not a boolean feature : extension required
has_intermediate_punctuation=9 #segmentation required
ends_like_verb=10
ends_like_adverb=11
change_in_capitalization=12 #not a boolean feature : #segmentation required
has_topic_indicator=13 #segmentation required
is_quoted=14
'''other features:
         POS_isAdjective
         POS_endwithAdjective
         Title
'''

class NE_candidate:
    """A simple NE_candidate class"""
    
    def __init__(self, phrase, position):
        length=0
        global capitalized
        global start_of_sentence
        global abbreviation
        global all_capitalized
        global is_csl
        global title
        #global is_number
        global has_number
        global date_indicator
        global is_apostrophed
        global has_intermediate_punctuation
        global ends_like_verb
        global ends_like_adverb
        global change_in_capitalization
        global has_topic_indicator
        global is_quoted
        self.phraseText=phrase
#         self.position=position
#         self.date_num_holder=[]
#         self.punctuation_holder=[]
        self.length=len(phrase.split())
#         self.features = [None]*15
        self.sen_index=''
        return
    
    
    def set_feature(self, feature_index, feature_value):
        self.features[feature_index]= feature_value
        return
    
    def set_punctuation_holder(self,holder_in):
        self.punctuation_holder=holder_in
        return
    
    def set_date_num_holder(self,holder_in):
        self.date_num_holder=holder_in
        return
    
    def reset_length(self):
        self.length=len(self.phraseText.split())
        return
    def set_sen_index(self,sen_index):
        self.sen_index=sen_index
    
    def print_obj(self):
        print (self.phraseText+" "+str(self.length)+" "+str(self.position)+" "+str(self.date_num_holder)+" "+str(self.punctuation_holder), end=" ")
        #print self.phraseText+" "+str(self.length),
        for feature in self.features:
            print (feature, end=" ")
        print ("")
        return