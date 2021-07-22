
# coding: utf-8

# In[298]:

import sys
import re
import string
import csv
import random
import time
import emoji
# import regex
#import binascii
#import shlex
import numpy as np
import pandas  as pd
from itertools import groupby
from operator import itemgetter
from collections import Iterable, OrderedDict
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import gutenberg
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTrainer
from scipy import stats
#from datasketch import MinHash, MinHashLSH
import NE_candidate_module as ne
import NE_candidate_module as ne
import Mention
import threading, queue
import time
import datetime
import copy
import trie as trie
import ast

# In[324]:

#---------------------Existing Lists--------------------
cachedStopWords = stopwords.words("english")
tempList=["i","and","or","other","another","across","unlike","anytime","were","you","then","still","till","nor","perhaps","otherwise","until","sometimes","sometime","seem","cannot","seems","because","can","like","into","able","unable","either","neither","if","we","it","else","elsewhere","how","not","what","who","when","where","who's","who’s","let","today","tomorrow","tonight","let's","let’s","lets","know","make","oh","via","i","yet","must","mustnt","mustn't","mustn’t","i'll","i’ll","you'll","you’ll","we'll","we’ll","done","doesnt","doesn't","doesn’t","dont","don't","don’t","did","didnt","didn't","didn’t","much","without","could","couldn't","couldn’t","would","wouldn't","wouldn’t","should","shouldn't","souldn’t","shall","isn't","isn’t","hasn't","hasn’t","wasn't","wasn’t","also","let's","let’s","let","well","just","everyone","anyone","noone","none","someone","theres","there's","there’s","everybody","nobody","somebody","anything","else","elsewhere","something","nothing","everything","i'd","i’d","i’m","won't","won’t","i’ve","i've","they're","they’re","we’re","we're","we'll","we’ll","we’ve","we've","they’ve","they've","they’d","they'd","they’ll","they'll","again","you're","you’re","you've","you’ve","thats","that's",'that’s','here’s',"here's","what's","what’s","i’m","i'm","a","so","except","arn't","aren't","arent","this","when","it","it’s","it's","he's","she's","she'd","he'd","he'll","she'll","she’ll","many","can't","cant","can’t","even","yes","no","these","here","there","to","maybe","<hashtag>","<hashtag>.","ever","every","never","there's","there’s","whenever","wherever","however","whatever","always","although"]
for item in tempList:
    if item not in cachedStopWords:
        cachedStopWords.append(item)
cachedStopWords.remove("don")
cachedStopWords.remove("your")
cachedStopWords.remove("up")
cachedTitles = ["mr.","mr","mrs.","mrs","miss","ms","sen.","dr","dr.","prof.","president","congressman"]
prep_list=["in","at","of","on","v."] #includes common conjunction as well
article_list=["a","an","the"]
conjoiner=["de"]
day_list=["sunday","monday","tuesday","wednesday","thursday","friday","saturday","mon","tues","wed","thurs","fri","sat","sun"]
month_list=["january","february","march","april","may","june","july","august","september","october","november","december","jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
chat_word_list=["nope","gee","hmm","bye","vs","ouch","omw","qt","dj","dm","congrat","haueheuaeh","ahushaush","jr","please","retweet","2mrw","2moro","4get","ooh","reppin","idk","oops","yup","stfu","uhh","2b","dear","yay","btw","ahhh","b4","ugh","ty","cuz","coz","sorry","yea","asap","ur","bs","rt","lmfao","lfmao","slfmao","u","r","nah","umm","ummm","thank","thanks","congrats","whoa","rofl","ha","ok","okay","hey","hi","huh","ya","yep","yeah","fyi","duh","damn","lol","omg","congratulations","fucking","fuck","f*ck","wtf","wth","aka","wtaf","xoxo","rofl","imo","wow","fck","haha","hehe","hoho"]

#string.punctuation.extend('“','’','”')
#---------------------Existing Lists--------------------


# In[300]:

class SatadishaModule():

    def __init__(self):
        print("hello")
        #self.batch=batch
        #self.batch=self.batch[:3000:]
        self.counter=0
        gutenberg_text = ""
        for file_id in gutenberg.fileids():
            gutenberg_text += gutenberg.raw(file_id)
        trainer = PunktTrainer()
        trainer.INCLUDE_ALL_COLLOCS = True
        trainer.train(gutenberg_text)
        self.my_sentence_tokenizer = PunktSentenceTokenizer(trainer.get_params())
        self.my_sentence_tokenizer._params.abbrev_types.add('dr')
        self.my_sentence_tokenizer._params.abbrev_types.add('c.j')
        self.my_sentence_tokenizer._params.abbrev_types.add('u.s')
        self.my_sentence_tokenizer._params.abbrev_types.add('u.s.a')


        #self.extract()

    def getWords(self, sentence):
        tempList=[]
        tempWordList=sentence.split()
        p_dots= re.compile(r'[.]{2,}')
        #print(tempWordList)
        for word in tempWordList:
            temp=[]
            
            if "(" in word:
                temp=list(filter(lambda elem: elem!='',word.split("(")))
                if(temp):
                    temp=list(map(lambda elem: '('+elem, temp))
            elif ")" in word:
                temp=list(filter(lambda elem: elem!='',word.split(")")))
                if(temp):
                    temp=list(map(lambda elem: elem+')', temp))
                # temp.append(temp1[-1])
            # elif (("-" in word)&(not word.endswith("-"))):
            #     temp1=list(filter(lambda elem: elem!='',word.split("-")))
            #     if(temp1):
            #         temp=list(map(lambda elem: elem+'-', temp1[:-1]))
            #     temp.append(temp1[-1])
            elif (("?" in word)&(not word.endswith("?"))):
                temp1=list(filter(lambda elem: elem!='',word.split("?")))
                if(temp1):
                    temp=list(map(lambda elem: elem+'?', temp1[:-1]))
                temp.append(temp1[-1])
            elif ((":" in word)&(not word.endswith(":"))):
                temp1=list(filter(lambda elem: elem!='',word.split(":")))
                if(temp1):
                    temp=list(map(lambda elem: elem+':', temp1[:-1]))
                temp.append(temp1[-1])
            elif (("," in word)&(not word.endswith(","))):
                #temp=list(filter(lambda elem: elem!='',word.split(",")))
                temp1=list(filter(lambda elem: elem!='',word.split(",")))
                if(temp1):
                    temp=list(map(lambda elem: elem+',', temp1[:-1]))
                temp.append(temp1[-1])
            elif (("/" in word)&(not word.endswith("/"))):
                temp1=list(filter(lambda elem: elem!='',word.split("/")))
                if(temp1):
                    temp=list(map(lambda elem: elem+'/', temp1[:-1]))
                temp.append(temp1[-1])
                #print(index, temp)
            # elif "..." in word:
            #     #print("here")
            #     temp=list(filter(lambda elem: elem!='',word.split("...")))
            #     if(temp):
            #         if(word.endswith("...")):
            #             temp=list(map(lambda elem: elem+'...', temp))
            #         else:
            #            temp=list(map(lambda elem: elem+'...', temp[:-1]))+[temp[-1]]
            #     # temp.append(temp1[-1])
            # elif ".." in word:
            #     temp=list(filter(lambda elem: elem!='',word.split("..")))
            #     if(temp):
            #         if(word.endswith("..")):
            #             temp=list(map(lambda elem: elem+'..', temp))
            #         else:
            #             temp=list(map(lambda elem: elem+'..', temp[:-1]))+[temp[-1]]
            #     #temp.append(temp1[-1])
            elif (list(p_dots.finditer(word))):
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
            elif "…" in word:
                temp=list(filter(lambda elem: elem!='',word.split("…")))
                if(temp):
                    if(word.endswith("…")):
                        temp=list(map(lambda elem: elem+'…', temp))
                    else:
                        temp=list(map(lambda elem: elem+'…', temp[:-1]))+[temp[-1]]
            else:
                #if word not in string.punctuation:
                temp=[word]
            if(temp):
                tempList.append(temp)
        tweetWordList=self.flatten(tempList,[])
        return tweetWordList


    def flatten(self,mylist, outlist,ignore_types=(str, bytes, int, ne.NE_candidate)):
    
        if mylist !=[]:
            for item in mylist:
                #print not isinstance(item, ne.NE_candidate)
                if isinstance(item, list) and not isinstance(item, ignore_types):
                    self.flatten(item, outlist)
                else:
                    if isinstance(item,ne.NE_candidate):
                        item.phraseText=item.phraseText.strip(' \t\n\r')
                        item.reset_length()
                    else:
                        if type(item)!= int:
                            item=item.strip(' \t\n\r')
                    outlist.append(item)
        return outlist


    def normalize(self,word):
        strip_op=word
        strip_op=(((strip_op.lstrip(string.punctuation)).rstrip(string.punctuation)).strip()).lower()
        strip_op=(strip_op.lstrip('“‘’”')).rstrip('“‘’”')
        #strip_op= self.rreplace(self.rreplace(self.rreplace(strip_op,"'s","",1),"’s","",1),"’s","",1)
        if strip_op.endswith("'s"):
            li = strip_op.rsplit("'s", 1)
            return ''.join(li)
        elif strip_op.endswith("’s"):
            li = strip_op.rsplit("’s", 1)
            return ''.join(li)
        else:
            return strip_op
            
    # @profile
    def extract(self,batch,batch_number): 
        #df = read_csv('eric_trump.csv', index_col='ID', header=0, encoding='utf-8')
        
        print("Phase I extracting now")
        time_in=time.time()
        self.batch=batch
        #output.csv
        #df_out= DataFrame(columns=('tweetID', 'sentID', 'hashtags', 'user', 'usertype', 'TweetSentence', 'phase1Candidates'))
        # self.df_out= pd.DataFrame(columns=('tweetID', 'sentID', 'hashtags', 'user', 'TweetSentence','tweetwordList', 'phase1Candidates','start_time','entry_batch','annotation'))
        if(self.counter==0):
            #self.df_out= pd.DataFrame(columns=('tweetID', 'sentID', 'hashtags', 'user', 'TweetSentence', 'phase1Candidates','correct_candidates_tweet'))
            #dict1 = {'tweetID':0, 'sentID':0, 'hashtags':'first', 'user':'user', 'TweetSentence':'sentence', 'phase1Candidates':'phase1Out','start_time':'now','entry_batch':'batch_number'}
            self.CTrie=trie.Trie("ROOT")
            self.ME_EXTR=Mention.Mention_Extraction()
            self.phase2stopWordList=[]
            #self.df_out= pd.DataFrame({'tweetID':0, 'sentID':0, 'hashtags':'first', 'user':'user', 'TweetSentence':'sentence', 'phase1Candidates':'phase1Out','start_time':'now','entry_batch':'batch_number'}, index=[0,])
        #%%timeit -o
        #module_capital_punct.main:
        '''I am running this for 100 iterations for testing purposes. Of course you no longer need this for loop as you are
        #running one tuple at a time'''
        #if(self.counter==0):

            #initializing candidateBase with a dummy node
            
            #self.interCWSGap={}
        #candidateBase={}


        #NE_container=DataFrame(columns=('candidate', 'frequency', 'capitalized', 'start_of_sentence', 'abbreviation', 'all_capitalized','is_csl','title','has_number','date_indicator','is_apostrophed','has_intermediate_punctuation','ends_like_verb','ends_like_adverb','change_in_capitalization','has_topic_indicator'))

        count=0
        ne_count=0
        userMention_count=0
        #token_count=0

        NE_list_phase1=[]
        UserMention_list=[]
        

        df_holder=[]
        quickRegex=re.compile("[a-z]+")
        # df= self.batch.filter(['TweetSentence','tweetID','sentID','tweetwordList','phase1Candidates','hashtags','user','entry_batch','annotation','stanford_candidates'])

        #--------------------------------------PHASE I---------------------------------------------------
        # for index, row in self.batch.iterrows():
        for row in self.batch.itertuples():

            index=row.Index

            now = datetime.datetime.now()
            #now=str(now.hour)+":"+str(now.minute)+":"+str(now.second)

            #hashtags=str(row['Discussion'])

            # hashtags=str(row['HashTags'])
            hashtags=str(row.HashTags)

            # user=str(row['User'])
            user=str(row.User)

            #userType=str(row['User Type'])

            # tweetText=str(row['TweetText'])
            tweetText=str(row.TweetText)


            #print(tweetText)
            #correct_candidates_tweet=str(row['Mentions'])
            #print(str(index))

            #annot_raw=str(row['mentions_other'])
            annot_raw=""

            # stanford_candidates=str(row['stanford_candidates'])
            # stanford_candidates=stanford_candidates.split(",")
            # stanford_candidates=list(filter(None, stanford_candidates))
            # stanford_candidates = [candidatee for candidatee in stanford_candidates if str(candidatee) != 'nan']
            stanford_candidates=""

            # ritter_candidates=str(row['ritter_candidates'])
            # ritter_candidates=ritter_candidates.split(",")
            # ritter_candidates=list(filter(None, ritter_candidates))
            # ritter_candidates = [candidatee for candidatee in ritter_candidates if str(candidatee) != 'nan']
            ritter_candidates = ""

            # calai_candidates=str(row['calai_candidates'])
            calai_candidates=""
            #calai_candidates=ast.literal_eval(calai_candidates)
            # print(calai_candidates)



            # split_list=annot_raw.split(";")
            # #split_listFilter=list(filter(lambda element: element.strip()!='', split_list))
            # split_listFilter=list(filter(None, split_list))


            #annotations in list of list structure
            #filtered_2_times=list(map(lambda element: list(filter(None, element.split(','))), split_list))
            
            #capitalization module
            #if all words are capitalized:
            # print(index)

            # if tweetText.isupper():
            #     print(index,tweetText)
            #     dict1 = {'tweetID':str(index), 'sentID':str(0), 'hashtags':hashtags, 'user':user, 'TweetSentence':tweetText, 'phase1Candidates':"nan",'start_time':now,'entry_batch':batch_number,'annotation':filtered_2_times[0]}
            #     df_holder.append(dict1)
            # elif tweetText.islower():
            #     print(index,tweetText)
            #     print("",end="")

            #     dict1 = {'tweetID':str(index), 'sentID':str(0), 'hashtags':hashtags, 'user':user, 'TweetSentence':tweetText, 'phase1Candidates':"nan",'start_time':now,'entry_batch':batch_number,'annotation':filtered_2_times[0]}
            #     df_holder.append(dict1)
            #else:
            ne_List_final=[]
            userMention_List_final=[]
            #pre-modification: returns word list split at whitespaces; retains punctuation
            tweetSentences=list(filter (lambda sentence: len(sentence)>1, tweetText.split('\n')))
            # tweetSentenceList_inter=self.flatten(list(map(lambda sentText: sent_tokenize(sentText.lstrip().rstrip()),tweetSentences)),[])
            tweetSentenceList_inter=self.flatten(list(map(lambda sentText: self.my_sentence_tokenizer.tokenize(sentText.lstrip().rstrip()),tweetSentences)),[])
            tweetSentenceList=list(filter (lambda sentence: len(sentence)>1, tweetSentenceList_inter))



            #filtering nan values 
            # if(len(filtered_2_times[0])==1):
            #     if(filtered_2_times[0][0]=='nan'):
            #         filtered_2_times[0]=[]


            # print(index,filtered_2_times,tweetSentenceList)
            



            for sen_index in range(len(tweetSentenceList)):
                sentence=tweetSentenceList[sen_index]
                # if(index==14155):
                #     print(sentence)

                # uncomment this 
                #modified_annotations=[self.normalize(candidate)for candidate in filtered_2_times[sen_index]]

                annotation=[]
                # for candidate in modified_annotations:
                #     if(candidate=="nan"):
                #         pass
                #     else:
                #         annotation.append(candidate)


                # for i in filtered_2_times[sen_index]:
                #     if(i=="nan"):

                #print(sentence)
                #print(sen_index)
                #tweetWordList= list(filter(lambda word:(word.strip(string.punctuation))!="",sentence.split()))
                p_dots= re.compile(r'[.]{2,}')
                # if p_dots.match(word):
                #     match_lst = p_dots.findall(word)
                #     index= (list( p1.finditer(cap_phrases) )[-1]).span()[1]

                phase1Out=""
                if((not tweetText.isupper()) &(not tweetText.islower())):
                    tempList=[]
                    tempWordList=sentence.split()
                    #print(tempWordList)
                    for word in tempWordList:
                        temp=[]
                        
                            # if(temp1):
                            #     temp=list(map(lambda elem: elem+'..', temp1[:-1]))
                            # temp.append(temp1[-1])
                        if (("?" in word)&(not word.endswith("?"))):
                            temp1=list(filter(lambda elem: elem!='',word.split("?")))
                            if(temp1):
                                temp=list(map(lambda elem: elem+'?', temp1[:-1]))
                            temp.append(temp1[-1])
                        elif ((":" in word)&(not word.endswith(":"))):
                            temp1=list(filter(lambda elem: elem!='',word.split(":")))
                            if(temp1):
                                temp=list(map(lambda elem: elem+':', temp1[:-1]))
                            temp.append(temp1[-1])
                        elif (("," in word)&(not word.endswith(","))):
                            #temp=list(filter(lambda elem: elem!='',word.split(",")))
                            temp1=list(filter(lambda elem: elem!='',word.split(",")))
                            if(temp1):
                                temp=list(map(lambda elem: elem+',', temp1[:-1]))
                            temp.append(temp1[-1])
                        elif (("/" in word)&(not word.endswith("/"))):
                            temp1=list(filter(lambda elem: elem!='',word.split("/")))
                            if(temp1):
                                temp=list(map(lambda elem: elem+'/', temp1[:-1]))
                            temp.append(temp1[-1])
                        # elif (("-" in word)&(not word.endswith("-"))):
                        #     temp1=list(filter(lambda elem: elem!='',word.split("-")))
                        #     if(temp1):
                        #         temp=list(map(lambda elem: elem+'-', temp1[:-1]))
                        #     temp.append(temp1[-1])
                        elif (list(p_dots.finditer(word))):
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
                        elif "…" in word:
                            temp=list(filter(lambda elem: elem!='',word.split("…")))
                            if(temp):
                                if(word.endswith("…")):
                                    temp=list(map(lambda elem: elem+'…', temp))
                                else:
                                    temp=list(map(lambda elem: elem+'…', temp[:-1]))+[temp[-1]]
                        # elif "..." in word:
                        #     #print("here")
                        #     temp=list(filter(lambda elem: elem!='',word.split("...")))
                        #     # if(temp1):
                        #     #     temp=list(map(lambda elem: elem+'...', temp1[:-1]))
                        #     # temp.append(temp1[-1])
                        # elif ".." in word:
                        #     temp=list(filter(lambda elem: elem!='',word.split("..")))
                        #     #print(index, temp)
                        else:
                            #if word not in string.punctuation:
                            if(word!='&;'):
                                temp=[word]
                            else:
                                temp=['&']
                        if(temp):
                            tempList.append(temp)
                    tweetWordList=self.flatten(tempList,[])
                    #print(tweetWordList)

                    #token_count+=len(tweetWordList)
                    #returns position of words that are capitalized
                    #print(tweetWordList)
                    

                    tweetWordList_cappos = list(map(lambda element : element[0], filter(lambda element : self.capCheck(element[1]), enumerate(tweetWordList))))
                    #print(tweetWordList_cappos)

                    hashtags_usermentions = list(filter(lambda word: (word.startswith('#'))|(word.startswith('@')), tweetWordList))

                    #returns list of stopwords in tweet sentence
                    combined_list_here=([]+cachedStopWords+article_list+prep_list+chat_word_list)
                    #combined_list_here.remove("the")
                    tweetWordList_stopWords=list(filter(lambda word: ((word[0].islower()) & (((word.strip()).strip(string.punctuation)).lower() in combined_list_here))|(word.strip() in string.punctuation)|(word.startswith('#'))|(word.startswith('@')), tweetWordList))
                    
                    #returns list of @userMentions
                    userMentionswPunct=list(filter(lambda phrase: phrase.startswith('@'), tweetWordList))
                    userMentions=list(map(lambda mention: mention.rstrip(string.punctuation), userMentionswPunct))
                    
                    userMention_count+=len(userMentions)
                    userMention_List_final+=userMentions  

                    '''#function to process and store @ user mentions---- thread 1
                    #print(userMention_List_final)
                    threading.Thread(target=self.ME_EXTR.ComputeAll, args=(userMention_List_final,)).start()'''

                    #non @usermentions are processed in this function to find non @, non hashtag Entities---- thread 2
                    ne_List_allCheck=[]

                    # if(index==484):
                    #     print(tweetWordList,tweetWordList_cappos)
                    #     print(len(tweetWordList),str(len(tweetWordList_cappos)),len(hashtags_usermentions))
                        # flags = re.findall(r'[^\w\s,]', tweetText)
                        # print([c for c in tweetWordList if c in emoji.UNICODE_EMOJI],flags) 

                    emoji_list = []
                    # # data = regex.findall(r'\X', tweetWordList)
                    for word in tweetWordList:
                        if any(char in emoji.UNICODE_EMOJI for char in word):
                            emoji_list.append(word)
                    # print(emoji_list)

                    # print(index,sentence)
                    # print(tweetWordList,tweetWordList_cappos,emoji_list,hashtags_usermentions)
                    # if((len(tweetWordList))<(len(tweetWordList_cappos)+len(emoji_list)+len(hashtags_usermentions))):
                    #     print(index,sentence)
                    #     print(tweetWordList,tweetWordList_cappos,emoji_list,hashtags_usermentions)
                    

                    if((len(tweetWordList))>(len(tweetWordList_cappos)+len(emoji_list)+len(hashtags_usermentions))):
                        
                        #q = queue.Queue()
                        #threading.Thread(target=self.trueEntity_process, args=(tweetWordList_cappos,tweetWordList,q)).start()
                        initial_elems_to_remove=[]
                        inner_index=0
                        for elem in tweetWordList:
                            if((elem.startswith('@'))|(elem.startswith('#'))):
                                initial_elems_to_remove.append(inner_index)
                                inner_index+=1
                                # print(tweetWordList)
                            else:
                                break
                        tweetWordList_edited=[tweetWordList[index] for index in range(len(tweetWordList)) if index not in initial_elems_to_remove]

                        tweetWordList_justcappos = list(map(lambda element : element[0], filter(lambda element : self.capCheck2(element[1]), enumerate(tweetWordList_edited))))

                        
                        ne_List_allCheck= self.trueEntity_process(index,tweetWordList_justcappos,tweetWordList_edited)
                    #ne_List_allCheck= q.get()
                        
                    ne_count+=len(ne_List_allCheck)
                    ne_List_final+=ne_List_allCheck

                    #write row to output dataframe

                    
                    if(len(tweetWordList)==(len(tweetWordList_cappos)+len(emoji_list)+len(hashtags_usermentions))):
                        phase1Out="nan"

                    if(len(ne_List_allCheck)>0):
                        for candidate in ne_List_allCheck:
                            position = '*'+'*'.join(str(v) for v in candidate.position)
                            position=position+'*'
                            candidate.set_sen_index(sen_index)
                            phase1Out+=(((candidate.phraseText).lstrip(string.punctuation)).strip())+ '::'+str(position)+"||" 
                else:
                    phase1Out="nan"
                    tweetWordList=self.getWords(sentence)

                #print(self.df_out.columns)
                enumerated_tweetWordList=[(token,idx) for idx,token in enumerate(tweetWordList)]
                dict1 = {'tweetID':str(index), 'sentID':str(sen_index), 'hashtags':hashtags, 'user':user, 'TweetSentence':sentence, 'tweetwordList': enumerated_tweetWordList, 'phase1Candidates':phase1Out,'start_time':now,'entry_batch':batch_number,'annotation':annotation,'stanford_candidates':stanford_candidates,'ritter_candidates':ritter_candidates,'calai_candidates':calai_candidates}
                df_holder.append(dict1)
                    #self.df_out.append(outrow)

                    #self.df_out=self.df_out.append(outrow,ignore_index=True)

            for candidate in ne_List_final:
                #self.insert_dict (candidate,self.NE_container,candidateBase,index,candidate.sen_index,batch_number)
                candidateText=(((candidate.phraseText.lstrip(string.punctuation)).rstrip(string.punctuation)).strip(' \t\n\r')).lower()
                candidateText=(candidateText.lstrip('“‘’”')).rstrip('“‘’”')
                candidateText= self.rreplace(self.rreplace(self.rreplace(candidateText,"'s","",1),"’s","",1),"’s","",1)
                # if(index==9423):
                #     print(candidateText)
                combined=[]+cachedStopWords+cachedTitles+prep_list+chat_word_list+article_list+day_list
                if not ((candidateText in combined)|(candidateText.isdigit())|(self.is_float(candidateText))):
                    if(quickRegex.match(candidateText)):
                        self.CTrie.__setitem__(candidateText.split(),len(candidateText.split()),candidate.features,batch_number)
                    self.CTrie.__setitem__(candidateText.split(),len(candidateText.split()),candidate.features,batch_number)
            # if(index==371):
            # #     # print(sentence)
            #     self.printList(ne_List_final)

            #if(userMention_List_final):
            #    print(userMention_List_final)

            NE_list_phase1+=ne_List_final
            UserMention_list+=userMention_List_final
                #print ("\n")


        
        #fieldnames=['candidate','freq','length','cap','start_of_sen','abbrv','all_cap','is_csl','title','has_no','date','is_apostrp','has_inter_punct','ends_verb','ends_adverb','change_in_cap','topic_ind','entry_time','entry_batch','@mention']
        #updated_NE_container=[]

        '''#Updating trie with @mention info
        self.CTrie.updateTrie("",self.ME_EXTR)'''
        time_out=time.time()
        
        #for display purposes Iterating through the trie
        '''candidateBase= self.CTrie.__iter__()
        for node in candidateBase:
            print(node)'''

        '''for key in self.NE_container.keys():
            val=self.NE_container[key]+[str(ME_EXTR.checkInDictionary(key))]
            #index+=1
            #updated_NE_container[key]=val
            dict1 = {'candidate':key, 'freq':val[0],'length':val[1],'cap':val[2],'start_of_sen':val[3],'abbrv':val[4],'all_cap':val[5],'is_csl':val[6],'title':val[7],'has_no':val[8],'date':val[9],'is_apostrp':val[10],'has_inter_punct':val[11],'ends_verb':val[12],'ends_adverb':val[13],'change_in_cap':val[14],'topic_ind':val[15],'entry_time':val[16],'entry_batch':val[17],'@mention':val[18]}
            updated_NE_container.append(dict1)'''


        '''with open('candidate_base.csv', 'w') as output_candidate:
         #with open('candidates.csv', 'w') as output_candidate:
             writer = csv.writer(output_candidate)
             writer.writerow(fieldnames)
             for k, v in updated_NE_container.items():
                 writer.writerow([k] + v)'''

        

        #print("Total number of tokens processed: "+str(token_count))
        #print ("Total number of candidate NEs extracted: "+str(len(candidateBase)))
        
        #print(self.NE_container.items())
        #freqs=pd.read_csv('candidate_base.csv',  encoding = 'utf-8',delimiter=',')
        #freqs = pd.DataFrame(updated_NE_container, columns=fieldnames)
        #freqs = pd.DataFrame()
        #freqs=pd.DataFrame(list(self.NE_container.items()),  orient='index')#columns=fieldnames)
        

        self.append_rows(df_holder)
        self.counter=self.counter+1
        
        #return (copy.deepcopy(self.df_out),copy.deepcopy(freqs),time_in,time_out)
        return (self.df_out,self.CTrie,time_in,time_out,self.phase2stopWordList)

        #return sorted_candidateBase
    # @profile
    def append_rows(self,df_holder):
    
        self.df_out = pd.DataFrame(df_holder,columns=('tweetID', 'sentID', 'hashtags', 'user', 'TweetSentence','tweetwordList', 'phase1Candidates','start_time','entry_batch','annotation'))
        # self.df_out=self.df_out.append(df)
               

        # self.df_out.to_csv('tweet_base.csv' ,sep=',', encoding='utf-8')
    
    def rreplace(self,s, old, new, occurrence):
        if s.endswith(old):
            li = s.rsplit(old, occurrence)
            return new.join(li)
        else:
            return s

    def stopwordReplace(self, candidate):
        combined=cachedStopWords+prep_list+article_list+day_list+chat_word_list

        if(candidate.features[ne.is_quoted]):
            words=self.normalize(candidate.phraseText).split()
            flag=False
            swList=[]
            for word in words:
                if(word in combined):
                    swList.append(word)
                else:
                    flag=True
            #print(candidate.phraseText,swList,flag)
            if(flag):
                self.phase2stopWordList=list(set(self.phase2stopWordList)|set(swList))
                #self.phase2stopWordList.extend(swList)
            else:
                candidate.phraseText=""
            return candidate

        wordlist=list(filter(lambda word: word!='', candidate.phraseText.split()))
        pos=candidate.position

        # print(candidate.phraseText,wordlist,pos)

        start=0
        flag=False
        while(start!=len(pos)):
            if(wordlist[start].lstrip(string.punctuation).rstrip(string.punctuation).strip().lower() not in combined):
                #flag=True
                break
            start+=1
        end=len(pos)-1
        while(end>=0):
            #print(wordlist[end])
            if(wordlist[end].lstrip(string.punctuation).rstrip(string.punctuation).strip() not in combined):
                #flag=True
                break
            end-=1
        #print(start,end)
        updated_pos=pos[start:(end+1)]
        updated_phrase=' '.join(wordlist[start:(end+1)])
        #print(updated_pos,updated_phrase)
        candidate.phraseText=updated_phrase
        candidate.position=updated_pos
        return candidate

# In[301]:

#candidate: 'frequency','length', 'capitalized', 'start_of_sentence', 'abbreviation', 'all_capitalized','is_csl','title','has_number','date_indicator','is_apostrophed','has_intermediate_punctuation','ends_like_verb','ends_like_adverb','change_in_capitalization','has_topic_indicator'
    def is_float(self,string):
        try:
            f=float(string)
            if(f==0.0):
              return True
            else:
              return ((f) and (string.count(".")==1))
      #return True# True if string is a number with a dot
        except ValueError:  # if string is not a number
          return False

    def insert_dict(self,candidate,NE_container,candidateBase,tweetID,sentenceID,batch):
        key=(((candidate.phraseText.lstrip(string.punctuation)).rstrip(string.punctuation)).strip(' \t\n\r')).lower()
        key=(key.lstrip('“‘’”')).rstrip('“‘’”')
        key= self.rreplace(self.rreplace(self.rreplace(key,"'s","",1),"’s","",1),"’s","",1)
        combined=[]+cachedStopWords+cachedTitles+prep_list+chat_word_list+article_list+day_list
        try:
            if ((key in combined)|(key.isdigit())|(self.is_float(key))):
                return
        except TypeError:
            print(key)

        tweetID=str(tweetID)
        sentenceID=str(sentenceID)

        if key in self.NE_container:
            feature_list=self.NE_container[key]
            feature_list[0]+=1
            for index in [0,1,2,3,4,5,6,7,9,10,11,13,14]:
                if (candidate.features[index]==True):
                    feature_list[index+2]+=1
            for index in [8,12]:
                if (candidate.features[index]!=-1):
                    feature_list[index+2]+=1
        else:
            now = datetime.datetime.now()
            now=str(now.hour)+":"+str(now.minute)+":"+str(now.second)
            feature_list=[0]*17
            feature_list[0]+=1
            feature_list[1]=candidate.length
            #call background process to check for non capitalized occurences
            for index in [0,1,2,3,4,5,6,7,9,10,11,13,14]:
                if (candidate.features[index]==True):
                    feature_list[index+2]+=1
            for index in [8,12]:
                if (candidate.features[index]!=-1):
                    feature_list[index+2]+=1
            feature_list.append(now)
            feature_list.append(batch)
            self.NE_container[key] = feature_list
        
        #insert in candidateBase
        '''if key in candidateBase.keys():
            #candidateBase[key]=candidateBase[key]+[str(tweetID)+":"+str(sentenceID)]
            if(tweetID in candidateBase[key]):
                if(sentenceID  in candidateBase[key][tweetID] ):
                    candidateBase[key][tweetID][sentenceID]=candidateBase[key][tweetID][sentenceID]+1

                else:
                    candidateBase[key][tweetID][sentenceID]=1

            else:
                candidateBase[key][tweetID]={}
                candidateBase[key][tweetID][sentenceID]=1
                #c=[(y,str(idx)) for idx,y  in enumerate( a)  if y not in b]
            #candidateBase[key]
        else:
            #candidateBase[key]=[str(tweetID)+":"+str(sentenceID)]
            candidateBase[key]={}
            candidateBase[key][tweetID]={}
            candidateBase[key][tweetID][sentenceID]=1'''
        return
        


# In[302]:

    def printList(self,mylist):
        print("["),
        #print "[",
        for item in mylist:
            if item != None:
                if isinstance(item,ne.NE_candidate):
                    item.print_obj()
                    #print (item.phraseText)
                else:
                    print (item+",", end="")
                    #print item+",",
        #print "]"
        print("]")
        return


# In[303]:




# In[304]:

    def consecutive_cap(self,index,tweetWordList_cappos,tweetWordList):
        output=[]
        #identifies consecutive numbers in the sequence
        #print(tweetWordList_cappos)
        for k, g in groupby(enumerate(tweetWordList_cappos), lambda element: element[0]-element[1]):
            output.append(list(map(itemgetter(1), g)))
        count=0
        if output:        
            final_output=[output[0]]
            for first, second in (zip(output,output[1:])):
                # print(first,second)
                #print(tweetWordList[first[-1]])
                if (((not (tweetWordList[first[-1]]).endswith('"'))&(not (tweetWordList[first[-1]].isdigit()|self.isfloat(tweetWordList[first[-1]])|self.ispercent(tweetWordList[first[-1]]))))&(((second[0]-first[-1])==2)&(not ((tweetWordList[second[0]].isdigit()|self.isfloat(tweetWordList[second[0]])|self.ispercent(tweetWordList[second[0]]))))) & (tweetWordList[first[-1]+1].lower() in prep_list)):
                    (final_output[-1]).extend([first[-1]+1]+second)
                elif (((not (tweetWordList[first[-1]]).endswith('"'))&(not (tweetWordList[first[-1]].isdigit()|self.isfloat(tweetWordList[first[-1]])|self.ispercent(tweetWordList[first[-1]]))))&(((second[0]-first[-1])==2)&(not ((tweetWordList[second[0]].isdigit()|self.isfloat(tweetWordList[second[0]])|self.ispercent(tweetWordList[second[0]]))))) & (tweetWordList[first[-1]+1].lower() in conjoiner)):
                    (final_output[-1]).extend([first[-1]+1]+second)
                elif(((not (tweetWordList[first[-1]]).endswith('"'))&(not (tweetWordList[first[-1]].isdigit()|self.isfloat(tweetWordList[first[-1]])|self.ispercent(tweetWordList[first[-1]]))))&(((second[0]-first[-1])==3)&(not ((tweetWordList[second[0]].isdigit()|self.isfloat(tweetWordList[second[0]])|self.ispercent(tweetWordList[second[0]]))))) & (tweetWordList[first[-1]+1].lower() in prep_list)& (tweetWordList[first[-1]+2].lower() in article_list)):
                    (final_output[-1]).extend([first[-1]+1]+[first[-1]+2]+second)
                else:
                    final_output.append(second)
                    #merge_positions.append(False)
        else:
            final_output=[]

        # if(index==24):
        # #     print('here')
        #     print(output)
        #     print(final_output)
        
        return final_output


# In[305]:

#basically splitting the original NE_candidate text and building individual object from each text snippet
    def build_custom_NE(self,phrase,pos,prototype,feature_index,feature_value):
        #print("Enters")
        position=pos
        custom_NE= ne.NE_candidate(phrase,position)
        for i in range(15):
            custom_NE.set_feature(i,prototype.features[i])
        custom_NE.set_feature(feature_index,feature_value)
        if (feature_index== ne.is_csl) & (feature_value== True):
            custom_NE.set_feature(ne.start_of_sentence, False)
        custom_NE=self.entity_info_check(custom_NE)
        return custom_NE


    # In[306]:

    def abbrv_algo(self,ne_element):
        '''abbreviation algorithm 
        trailing apostrophe:
               |period:
               |     multiple letter-period sequence:
               |         all caps
               | non period:
               |     ?/! else drop apostrophe
        else:
            unchanged
        '''
        phrase= ne_element.phraseText
        #print("=>"+phrase)
        #since no further split occurs we can set remaining features now
        ne_element.set_feature(ne.capitalized, True)
        if ne_element.phraseText.isupper():
            ne_element.set_feature(ne.all_capitalized, True)
        else:
            ne_element.set_feature(ne.all_capitalized, False)
            
        abbreviation_flag=False
        p=re.compile(r'[^a-zA-Z\d\s]$')
        match_list = p.findall(phrase)
        if len(match_list)>0:
            #print("Here")
            if phrase.endswith('.'):
                #print("Here")
                p1= re.compile(r'([a-zA-Z][\.]\s*)')
                match_list = p1.findall(phrase)
                if ((len(match_list)>1) & (len(phrase)<6)):
                    #print ("1. Found abbreviation: "+phrase)
                    abbreviation_flag= True
                else:
                    if (phrase[-2]!=' '):
                        phrase= phrase[:-1]

            else:
                #if phrase.endswith(string.punctuation):
                if (phrase[-2]!=' '):
                    phrase= phrase[:-1]
                #if not (phrase.endswith('?')|phrase.endswith('!')|phrase.endswith(')')|phrase.endswith('>')):
                    #phrase= phrase[:-1]
        else:
            p2=re.compile(r'([^a-zA-Z0-9_\s])')
            match_list = p2.findall(phrase)
            if ((len(match_list)==0) & (phrase.isupper()) & (len(phrase)<7)& (len(phrase)>1)):
                #print ("2. Found abbreviation!!: "+phrase)
                abbreviation_flag= True
            else:
                #print("Here-> "+phrase)
                p3= re.compile(r'([A-Z][.][A-Z])')
                p4= re.compile(r'\s')
                match_list = p3.findall(phrase)
                match_list1 = p4.findall(phrase)
                if ((len(match_list)>0) & (len(match_list1)==0)):
                    abbreviation_flag= True
                    #print ("3. Found abbreviation!!: "+phrase)
                
        #element= ne.NE_candidate(phrase.strip())
        ne_element.phraseText=phrase
        ne_element.reset_length()
        ne_element.set_feature(ne.abbreviation, abbreviation_flag)
        return ne_element
        


    # In[307]:


    def punct_clause(self,tweet_index,NE_phrase_in):
        
        NE_phrases=self.entity_info_check(NE_phrase_in)
        cap_phrases=NE_phrases.phraseText.strip()
        final_lst=[]
        #print (cap_phrases,NE_phrases.features[ne.date_indicator])
        if (re.compile(r'[^a-zA-Z0-9_\s]')).findall(cap_phrases):
            #case of intermediate punctuations: handles abbreviations
            p1= re.compile(r'(?:[a-zA-Z0-9][^a-zA-Z0-9_\s]\s*)')
            match_lst = p1.findall(cap_phrases)
            #print(match_lst)
            if match_lst:
                index= (list( p1.finditer(cap_phrases) )[-1]).span()[1]
            
            p= re.compile(r'[^a-zA-Z\d\s]')
            match_list = p.findall(cap_phrases)

            p2=re.compile(r'[^a-zA-Z\d\s]$') #ends with punctuation

            if ((len(match_list)>0)&(len(match_lst)>0)&((len(match_list)-len(match_lst))>0)):
                if (p2.findall(cap_phrases)):
                    #only strips trailing punctuations, not intermediate ones following letters
                    cap_phrases = cap_phrases[0:index]+re.sub(p, '', cap_phrases[index:])
                    NE_phrases.phraseText= cap_phrases
            
        
        #comma separated NEs
        #lst=filter(lambda(word): word!="", re.split('[,]', cap_phrases))
        #print ("=>"+ cap_phrases)
        start_of_sentence_fix=NE_phrases.features[ne.start_of_sentence]
        #temp=re.split("\...", cap_phrases)
        #inter=self.flatten(list(map(lambda elem: re.split('[,:!…]',elem),temp)),[])
        #print("'''",inter)
        combined=cachedStopWords+prep_list+article_list+day_list+chat_word_list
        splitList=re.split('["‘’“”()/,;:!?…]',cap_phrases)
        splitList=list(filter(lambda word: ((word!="")&(word.lstrip(string.punctuation).rstrip(string.punctuation).strip().lower() not in combined)), splitList))

               
        wordlstU=list(map(lambda word: word.strip().strip(string.punctuation), splitList))
        wordlstU=list(filter(lambda word: (word!="")&(not word.isspace()), wordlstU))
        # print(wordlstU)
        wordlst=list(filter(lambda word: ((word.strip().strip(string.punctuation))[0].isupper()|(word.strip().strip(string.punctuation))[0].isdigit()), wordlstU))

        splitList_wo_comma=re.split('["‘’“”()/;:!?…]',cap_phrases)
        splitList_wo_comma=list(filter(lambda word: ((word!="")&(word.lstrip(string.punctuation).rstrip(string.punctuation).strip().lower() not in combined)), splitList_wo_comma))

        wordlstU_wo_comma=list(map(lambda word: word.strip().strip(string.punctuation), splitList_wo_comma))
        wordlstU_wo_comma=list(filter(lambda word: (word!="")&(not word.isspace()), wordlstU_wo_comma))
        wordlst_wo_comma=list(filter(lambda word: ((word.strip().strip(string.punctuation))[0].isupper()|(word.strip().strip(string.punctuation))[0].isdigit()), wordlstU_wo_comma))

        #print(":::",wordlst)
        # if (tweet_index==159):
        #     print(cap_phrases,"==",splitList,NE_phrases.features[ne.date_indicator])
        if ((NE_phrases.features[ne.date_indicator]==False)):
            #print("hehe")
            if(len(splitList)>1):
                if(len(wordlst)>0):
                    #print("here::")
                    pos=NE_phrases.position
                    combined=[]
                    prev=0
                    for i in range(len(wordlst)):
                        word=wordlst[i]
                        word_len=len(list(filter(lambda individual_word: individual_word!="", re.split('[ ]', word))))
                        word_pos=pos[(prev):(prev+word_len)]
                        prev=prev+word_len
                        combined+=[[word]+word_pos]
                    
                    lst_nsw=list(filter(lambda element: (((str(element[0])).strip(string.punctuation).lower() not in combined)& (not (str(element[0])).strip(string.punctuation).isdigit()) & (len(str(element[0]))>1)) ,combined))
                    #print ("++",lst_nsw)
                    if(lst_nsw):
                        final_lst= list(map(lambda element:self.build_custom_NE(str(element[0]),element[1:],NE_phrases,ne.is_csl,True), lst_nsw))
                        final_lst[0].set_feature(ne.start_of_sentence, NE_phrases.features[ne.start_of_sentence])
                else:
                    final_lst=[]
            else:
                NE_phrases.set_feature(ne.is_csl,False)
                final_lst=[NE_phrases]
        else:
            if(len(splitList_wo_comma)>1):
                if(len(wordlst_wo_comma)>0):
                    #print("here::")
                    pos=NE_phrases.position
                    combined=[]
                    prev=0
                    for i in range(len(wordlst_wo_comma)):
                        word=wordlst_wo_comma[i]
                        word_len=len(list(filter(lambda individual_word: individual_word!="", re.split('[ ]', word))))
                        word_pos=pos[(prev):(prev+word_len)]
                        prev=prev+word_len
                        combined+=[[word]+word_pos]
                    
                    lst_nsw=list(filter(lambda element: (((str(element[0])).strip(string.punctuation).lower() not in combined)& (not (str(element[0])).strip(string.punctuation).isdigit()) & (len(str(element[0]))>1)) ,combined))
                    #print ("++",lst_nsw)
                    if(lst_nsw):
                        final_lst= list(map(lambda element:self.build_custom_NE(str(element[0]),element[1:],NE_phrases,ne.is_csl,True), lst_nsw))
                        final_lst[0].set_feature(ne.start_of_sentence, NE_phrases.features[ne.start_of_sentence])
                else:
                    final_lst=[]
            else:
                NE_phrases.set_feature(ne.is_csl,False)
                final_lst=[NE_phrases]
        
        #check abbreviation
        #print("++",final_lst)
        if(final_lst):
            final_lst= list(map(lambda phrase: self.abbrv_algo(phrase), final_lst))

        
        #print(lst)
        return final_lst


    # In[308]:

    #%%timeit -o
    def f(self,y,sflag,quoteFlag,tweetWordList):

        combined=[]+cachedStopWords+cachedTitles+prep_list+chat_word_list+article_list+day_list
        #print(sflag)
        if sflag:
            left=""
            right=""
            lp=(-1)
            rp=(-1)
            i=0
            j=len(y)-1
            flag1=False
            flag2=False
            x=[]
            while (((flag1==False)|(flag2==False))&((j-i)>0)):
                if(flag1==False):
                    left=(((tweetWordList[y[i]].strip('“‘"’”')).strip("'").lstrip(string.punctuation)).rstrip(string.punctuation)).lower()
                    if(left not in combined):
                        flag1=True
                        lp=i
                    else:
                        i+=1
                if(flag2==False):
                    right=(((tweetWordList[y[j]].strip('“‘"’”')).strip("'").lstrip(string.punctuation)).rstrip(string.punctuation)).lower()
                    if(right not in combined):
                        flag2=True
                        rp=j
                    else:
                        j-=1
            #print(flag1,flag2)
            #if((flag1==False)|(flag2==False)):
            # while (((j-i)!=0)|((flag1==False)|(flag2==False))):
            if(flag1==False):
                left=(((tweetWordList[y[i]].strip('“‘"’”')).strip("'").lstrip(string.punctuation)).rstrip(string.punctuation)).lower()
                #print(left)
                if(left not in combined):
                    flag1=True
                    lp=i
                else:
                    i+=1
            if(flag2==False):
                right=(((tweetWordList[y[j]].strip('“‘"’”')).strip("'").lstrip(string.punctuation)).rstrip(string.punctuation)).lower()
                if(right not in combined):
                    flag2=True
                    rp=j
                else:
                    j-=1
            #print(lp,rp)
            if(lp==rp):
                if(lp!=-1):
                    x=[y[lp]]
            else:
                x=y[lp:(rp+1)]

        else:
            x=y
        #print(x)
        if(x):
            list1=list(map(lambda word: tweetWordList[word], x))

            phrase=" ".join(e for e in list1)
            #print(phrase)
            phrase1="".join(list1)

            #if not ((phrase[0].isdigit()) & (len(x)==1)):
            if not (phrase1.strip().isdigit()):
                NE_phrase= ne.NE_candidate(phrase.strip().strip(string.punctuation),x)
                if 0 in x:
                    NE_phrase.set_feature(ne.start_of_sentence,True)
                else:
                    NE_phrase.set_feature(ne.start_of_sentence,False)
                NE_phrase.set_feature(ne.is_quoted,quoteFlag)
            else:
                NE_phrase= ne.NE_candidate("JUST_DIGIT_ERROR",[])
        else:
            NE_phrase= ne.NE_candidate("JUST_DIGIT_ERROR",[])
        #print("====>>",NE_phrase.phraseText)
        return NE_phrase


    # In[309]:
    def ispercent(self,word):
        # print(word)

        p=re.compile(r'\b(?<!\.)(?!0+(?:\.0+)?%)(?:\d|[1-9]\d|100)(?:(?<!100)\.\d+)?%')
        l= p.match(word)
        if l:
            return True
        else:
            return False

    def isfloat(self, value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    def capCheck2(self,word):
        combined_list=[]+cachedStopWords+prep_list+chat_word_list+article_list+conjoiner
        p_num=re.compile(r'^[\W]*[0-9]')
        
        if word.startswith('@'):
            return False
        if word.startswith('#'):
            return False
        elif "<Hashtag" in word:
            return False
        #elif (((word.strip('“‘’”')).lstrip(string.punctuation)).rstrip(string.punctuation)).lower() in combined_list:
        elif (((word.strip('“‘’”')).lstrip(string.punctuation)).rstrip(string.punctuation)) in combined_list:
            # if((word=="The")|(word=="THE")):
            #     return True
            # else:
            return False
        elif word[0].isdigit():
            return True
        else:
            p=re.compile(r'^[\W]*[A-Z]')
            l= p.match(word)
            if l:
                return True
            else:
                return False

    def capCheck(self,word):
        # print(word)
        combined_list=[]+cachedStopWords+prep_list+chat_word_list+article_list+conjoiner
        p_num=re.compile(r'^[\W]*[0-9]')
        p_punct=re.compile(r'[\W]+')

        if word.startswith('@'):
            return False
        if word.startswith('#'):
            return False
        elif "<Hashtag" in word:
            return False
        # elif not (((word.strip('“‘’”')).lstrip(string.punctuation)).rstrip(string.punctuation)).lower():
        #     return True
        elif (((word.strip('“‘’”')).lstrip(string.punctuation)).rstrip(string.punctuation)) in combined_list:
            # if((word=="The")|(word=="THE")):
            #     return True
            # else:
            return True
        elif p_num.match(word):
            return True
        else:
            p=re.compile(r'^[\W]*[A-Z]')
            l= p.match(word)
            if l:
                return True
            else:
                l2= p_punct.match(word)
                if l2:
                    return True
                else:
                    if (word.strip() in string.punctuation):
                        return True
                    else:
                        return False


    # In[310]:

    def title_check(self,ne_phrase):
        title_flag=False
        words=ne_phrase.phraseText.split()
        for word in words:
            if word.lower() in cachedTitles:
                title_flag= True
                break
        ne_phrase.set_feature(ne.title,title_flag)
        return ne_phrase


    # In[311]:

    def entity_info_check(self,ne_phrase):
        flag1=False #has number
        flag3=False
        flag_ind=[] #is number
        month_ind=[]
        date_num_holder=[]
        words=ne_phrase.phraseText.split()
        for word in words:
            word=(word.strip()).rstrip(string.punctuation).lower()
            punct_flag=False
            for char in word:
                if ((char in string.punctuation)|(char in ['“','‘','’','”','…'])):
                    punct_flag=True
                    break
            #if ((not word.isalpha())& (not "'s" in word) & (not "’s" in word)):'‘“"’”
            if ((not word.isalpha())& (not punct_flag)):
                flag_ind+=[True]
                if word.isdigit():
                    date_num_holder+=['num']
                else:
                    date_num_holder+=['alpha']
            else:
                flag_ind+=[False]
                if word in month_list:
                    month_ind+=[True]
                    date_num_holder+=['month']
                elif word in day_list:
                    date_num_holder+=['day']
                elif word in prep_list:
                    date_num_holder+=['preposition']
                elif word in article_list:
                    date_num_holder+=['article']
                else:
                    #print("=>"+word)
                    date_num_holder+=['string']
        if True in flag_ind:
            flag1=True
        if True in month_ind:
            flag3=True
        ne_phrase.set_feature(ne.has_number,flag1)
        ne_phrase.set_feature(ne.date_indicator,flag3)
        ne_phrase.set_date_num_holder(date_num_holder)
        return ne_phrase


    # In[312]:

    #removing commonly used expletives, enunciated chat words and other common words (like days of the week, common expressions)
    def slang_remove(self,ne_phrase):
        phrase=(ne_phrase.phraseText.strip()).rstrip(string.punctuation).lower()
        p1= re.compile(r'([A-Za-z]+)\1\1{1,}')
        match_lst = p1.findall(phrase)
        if phrase in article_list:
            return True
        elif phrase in day_list:
            return True
        #elif phrase in month_list:
            #return True
        elif match_lst:
            return True
        else:
            return False


    # In[313]:

    def apostrope_check(self,ne_phrase):
        apostrophe="'s"
        bad_apostrophe="’s"
        ret_ne_list=[]
        phrase=(ne_phrase.phraseText.strip()).rstrip(string.punctuation).lower()
        position=ne_phrase.position
        if (apostrophe in phrase):
            if (phrase.endswith(apostrophe)):
                ne_phrase.set_feature(ne.is_apostrophed,0)
                ret_ne_list.append(ne_phrase)
            else:
                # ret_ne_list=re.split(apostrophe,ne_phrase)
                #splitting at apostrophe
                phrase_beg=phrase[:phrase.find(apostrophe)].strip()
                pos_beg=position[:len(phrase_beg.split())]
                return_ne_beg= self.build_custom_NE(phrase_beg,pos_beg,ne_phrase,ne.is_csl,ne_phrase.features[ne.is_csl])
                return_ne_beg.set_feature(ne.is_apostrophed,0)
                ret_ne_list.append(return_ne_beg)

                phrase_end=phrase[phrase.find(apostrophe)+2:].strip()
                pos_end=position[len(phrase_beg.split()):]
                return_ne_end= self.build_custom_NE(phrase_end,pos_end,ne_phrase,ne.is_csl,ne_phrase.features[ne.is_csl])
                ret_ne_list.append(return_ne_end)
                
                # ret_ne_list=[,ne_phrase[ne_phrase.find(apostrophe)+2:].strip()]
                # print(phrase,phrase_beg,phrase_end)
                # ne_phrase.set_feature(ne.is_apostrophed,phrase.find(apostrophe))
        elif (bad_apostrophe in phrase):
            if phrase.endswith(bad_apostrophe):
                ne_phrase.set_feature(ne.is_apostrophed,0)
                ret_ne_list.append(ne_phrase)
            else:
                phrase_beg=phrase[:phrase.find(bad_apostrophe)].strip()
                pos_beg=position[:len(phrase_beg.split())]
                return_ne_beg= self.build_custom_NE(phrase_beg,pos_beg,ne_phrase,ne.is_csl,ne_phrase.features[ne.is_csl])
                return_ne_beg.set_feature(ne.is_apostrophed,0)
                ret_ne_list.append(return_ne_beg)

                phrase_end=phrase[phrase.find(bad_apostrophe)+2:].strip()
                pos_end=position[len(phrase_beg.split()):]
                return_ne_end= self.build_custom_NE(phrase_end,pos_end,ne_phrase,ne.is_csl,ne_phrase.features[ne.is_csl])
                ret_ne_list.append(return_ne_end)

                # ret_ne_list=[ne_phrase[:ne_phrase.find(bad_apostrophe)].strip(),ne_phrase[ne_phrase.find(bad_apostrophe)+2:].strip()]
                # print(phrase,phrase_beg,phrase_end)
                # ne_phrase.set_feature(ne.is_apostrophed,phrase.find(bad_apostrophe))
        else:
            ne_phrase.set_feature(ne.is_apostrophed,-1)
            ret_ne_list.append(ne_phrase)
        return ret_ne_list


    # In[314]:

    def punctuation_check(self,ne_phrase):
        holder=[]
        punctuation_holder=[]
        flag_holder=[]
        phrase=(ne_phrase.phraseText.strip()).rstrip(string.punctuation).lower()
        for i in range(len(phrase)):
            if (phrase[i] in string.punctuation):
                holder+=[i]
        for i in holder:
            if ((i<(len(phrase)-1)) & (phrase[i]=="'") & (phrase[i+1]=="s")):
                flag_holder+=[False]
            elif ((i==(len(phrase)-1)) & (phrase[i]=="'")):
                flag_holder+=[False]
            else:
                flag_holder+=[True]
                punctuation_holder+=[i]
        #print(flag_holder)
        ne_phrase.set_punctuation_holder(punctuation_holder)
        if True in flag_holder:
            ne_phrase.set_feature(ne.has_intermediate_punctuation,True)
        else:
            ne_phrase.set_feature(ne.has_intermediate_punctuation,False)
        return ne_phrase


    # In[315]:

    def tense_check(self,ne_phrase):
        words=(((ne_phrase.phraseText.strip()).rstrip(string.punctuation)).lower()).split()
        verb_flag=False
        adverb_flag=False
        if (len(words)==1):
            if words[0].endswith("ing"):
                verb_flag=True
            if words[0].endswith("ly"):
                adverb_flag=True
        ne_phrase.set_feature(ne.ends_like_verb,verb_flag)
        ne_phrase.set_feature(ne.ends_like_adverb,adverb_flag)
        return ne_phrase


    # In[316]:

    def capitalization_change(self,ne_element):
        phrase=((ne_element.phraseText.lstrip(string.punctuation)).rstrip(string.punctuation)).strip()
        val=-1
        topic_indicator=False
        p1= re.compile(r'[A-Z]*\s*[A-Z]{4,}[^A-Za-z]*\s+[A-Za-z]+') #BREAKING: Toronto Raptors
        p2= re.compile(r'([A-Z]{1}[a-z]+)+[^A-Za-z]*\s+[A-Z]{4,}') #The DREAMIEST LAND
        match_lst1 = p1.findall(phrase)
        match_lst2 = p2.findall(phrase)
        if (match_lst1):
            if not phrase.isupper():
                p3=re.compile(r'[A-Z]*\s*[A-Z]{4,}[^A-Za-z]*\s+')
                val=list(p3.finditer(phrase))[-1].span()[1]
                if(":" in phrase):
                    topic_indicator=True
                ne_element.set_feature(ne.change_in_capitalization,val)
        elif (match_lst2):
            #print ("GOTIT2: "+phrase)
            p3=re.compile(r'([A-Z]{1}[a-z]+)+')
            val=list(p3.finditer(phrase))[-1].span()[1]
            ne_element.set_feature(ne.change_in_capitalization,val)
        else:
            ne_element.set_feature(ne.change_in_capitalization,val)
        ne_element.set_feature(ne.has_topic_indicator,topic_indicator)
        return ne_element




    def quoteProcess(self,unitQuoted, tweetWordList):
        candidateString=""
        retList=[]
        matches=[]
        quoteMatch=[]
        final=[]
        flag=False
        #print(tweetWordList)
        list1=list(map(lambda index: tweetWordList[index], unitQuoted))
        candidateString=" ".join(list1)
        #print("=>",candidateString)
        # candidateString=""
        # for index in range(len(unitQuoted)-1):
        #     candidateString+=tweetWordList[unitQuoted[index]]+" "
        # candidateString+=tweetWordList[unitQuoted[-1]]
        # print("=>",candidateString)
        flagOne=False
        flagTwo=False
        flagThree=False
        flagFour=False
        p= re.compile(r'[^\S]*([\'].*?[\'])[^a-zA-Z0-9\s]*[\s]*')
        p1=re.compile(r'[^\s]+([\'].*?[\'])[^\s]*')
        p2=re.compile(r'[^\s]*([\'].*?[\'])[^\s]+')
        indices= (list(p.finditer(candidateString)))
        indices1= (list(p1.finditer(candidateString)))
        indices2= (list(p2.finditer(candidateString)))
        if((len(indices)>0) & (len(indices1)==0)& (len(indices2)==0)):
            flagOne=True
        
        if(not flagOne):
            p= re.compile(r'[^\S]*([‘].*?[’])[^a-zA-Z0-9\s]*[\s]*')
            p1=re.compile(r'[^\s]+([‘].*?[’])[^\s]*')
            p2=re.compile(r'[^\s]*([‘].*?[’])[^\s]+')
            indices= (list(p.finditer(candidateString)))
            indices1= (list(p1.finditer(candidateString)))
            indices2= (list(p2.finditer(candidateString)))
            if((len(indices)>0) & (len(indices1)==0)& (len(indices2)==0)):
                flagTwo=True
        
        if((not flagOne)&(not flagTwo)):
            p= re.compile(r'[^\S]*([“].*?[”])[^a-zA-Z0-9\s]*[\s]*')
            p1=re.compile(r'[^\s]+([“].*?[”])[^\s]*')
            p2=re.compile(r'[^\s]*([“].*?[”])[^\s]+')
            indices= (list(p.finditer(candidateString)))
            indices1= (list(p1.finditer(candidateString)))
            indices2= (list(p2.finditer(candidateString)))
            if((len(indices)>0) & (len(indices1)==0)& (len(indices2)==0)):
                flagThree=True

        if((not flagOne)&(not flagTwo)&(not flagThree)):
            p= re.compile(r'[^\S]*([\"].*?[\"])[^a-zA-Z0-9\s]*[\s]*')
            p1=re.compile(r'[^\s]+([\"].*?[\"])[^\s]*')
            p2=re.compile(r'[^\s]*([\"].*?[\"])[^\s]+')
            indices= (list(p.finditer(candidateString)))
            indices1= (list(p1.finditer(candidateString)))
            indices2= (list(p2.finditer(candidateString)))
            if((len(indices)>0) & (len(indices1)==0)& (len(indices2)==0)):
                flagFour=True
        
        if (flagOne|flagTwo|flagThree|flagFour):
            flag=True
            for index in indices:
                span= list(index.span())
                #print(span[0])
                quoteMatch.append([int(span[0]),int(span[1])])
                matches+=[int(span[0]),int(span[1])]
            #print(matches)
            final+=[(candidateString[0:matches[0]],False)]
            for i in range(len(matches)-1):
                if([matches[i],matches[i+1]] in quoteMatch):
                    final+=[((candidateString[matches[i]:matches[i+1]]).strip(),True)]
                else:
                    final+=[((candidateString[matches[i]:matches[i+1]]).strip(),False)]
            final+=[(candidateString[matches[-1]:],False)]
            final=list(filter(lambda strin: strin[0]!="",final))
            final=list(map(lambda strin: (strin[0].strip(),strin[1]),final))
            #print(final)
            for unit in final:
                lst=[]
                unitsplit=list(filter(lambda unitString: unitString!='',unit[0].split()))
                for splitunit in unitsplit:
                    lst+=[tweetWordList.index(splitunit,unitQuoted[0])]
                retList+=[(lst,unit[1])]
        else:
            retList+=[(unitQuoted,False)]
        #print(retList)
        return retList

    def apostrophe_split(self,ne_phrase):

        apostrophe="'s"
        bad_apostrophe="’s"


    # In[318]:

    def trueEntity_process(self,tweet_index,tweetWordList_cappos,tweetWordList):
        
        combined=[]+cachedStopWords+prep_list+chat_word_list+article_list+day_list+conjoiner
        #returns list with position of consecutively capitalized words
        # print(tweetWordList_cappos, tweetWordList)
        output_unfiltered = self.consecutive_cap(tweet_index,tweetWordList_cappos,tweetWordList)
        #print("==>",output_unfiltered)
        # if(tweet_index==371):
        #     print("==>",output_unfiltered)

        #splitting at quoted units
        output_quoteProcessed=[]
        start_quote=[]
        end_quote=[]
        for unitQuoted in output_unfiltered:
            unitout=self.quoteProcess(unitQuoted, tweetWordList)
            # if(tweet_index==589):
            #     print("here ==>",unitout)
            for elem in unitout:
                mod_out=[]
                out=elem[0]
                flag=elem[1]
                sflag=False
                # '’”"
                #print(out,flag)
                if not (flag):
                    #for id in range(len(out)):
                    temp=[]
                    #print("::",out)
                    for index in out:
                        #print(index,tweetWordList[index])
                        word=(((tweetWordList[index].strip().strip('"“‘’”"')).lstrip(string.punctuation)).rstrip(string.punctuation)).lower()
                        #print("=>"+word)"“‘’”"
                        if (word):
                            if (word in combined):
                                if(len(out)==1):
                                    temp.append(index)
                                else:
                                    if ((word not in prep_list)&(word not in article_list)&(word not in conjoiner)):
                                        # if(tweet_index==371):
                                            # print(word)
                                        temp.append(index)
                                    else:

                                        sflag=True
                                    #else:
                                        #if ((index==0)||()):
                                        #temp.append(index)
                                        # else:
                                        #     print("here")
                            # else:
                            #     print("here")
                    #print(temp)
                    for elem in temp:
                        out.remove(elem)
                    #out[id]=temp
                    lst=[]
                    for k, g in groupby(enumerate(out), lambda elem: elem[1]-elem[0]):
                        lst=list(map(itemgetter(1), g))
                        #print("==>",lst)
                                     
                        if(lst):
                            mod_out.append((lst,sflag,flag))
                        #print('==>',mod_out)
                else:
                    mod_out=[(out,sflag,flag)]
                    #print(mod_out)
                #print(mod_out)
                if(mod_out):
                    output_quoteProcessed.extend(mod_out)
        #'cgl\print("=====>",output_quoteProcessed)
        # if(tweet_index==371):
        #     print("=====>",output_quoteProcessed)
        output= list(filter(lambda element: ((element[0]!=[0])&(element[0]!=[])), output_quoteProcessed))
        
        #print(output)

        #consecutive capitalized phrases 
        consecutive_cap_phrases1=list(map(lambda x: self.f(x[0],x[1],x[2],tweetWordList), output))

        consecutive_cap_phrases=list(filter(lambda candidate:(candidate.phraseText!="JUST_DIGIT_ERROR"),consecutive_cap_phrases1))
        # if(tweet_index==589):
        #     print('herherhe')
        #     self.printList(consecutive_cap_phrases)

        # #new apostrophe split function should be here
        # ne_List_pc=self.flatten(list(map(lambda NE_phrase: self.apostrophe_split(NE_phrase), consecutive_cap_phrases)),[])

        #implement the punctuation clause
        ne_List_pc=self.flatten(list(map(lambda NE_phrase: self.punct_clause(tweet_index,NE_phrase), consecutive_cap_phrases)),[])
        # if(tweet_index==371):
        #     # print("==>",ne_List_pc)
        #     self.printList(ne_List_pc)

        ##implement apostrophe check
        ne_List_apostropeCheck= self.flatten(list(map(lambda element: self.apostrope_check(element), ne_List_pc)),[])

        #stopword removal and start-of-sentence
        ne_List_pc_sr= list(map(lambda candidate: self.stopwordReplace(candidate), ne_List_apostropeCheck))
        #self.printList(ne_List_pc_sr)
        ne_List_pc_checked= list(filter(lambda candidate: ((candidate.phraseText!="")&(candidate.position!=[0])), ne_List_pc_sr))

        # if(tweet_index==589):
        #     print(':',self.printList(ne_List_pc_sr))
        #     print('::',self.printList(ne_List_pc_checked))


        #implement title detection
        #ne_List_titleCheck= list(map(lambda element: self.title_check(element), ne_List_pc_checked))
        
        #implement slang check and remove
        ne_List_slangCheck= list(filter(lambda element: not self.slang_remove(element), ne_List_pc_checked))
        
        #implement tense and punctuation marker with final number check
        #ne_List_punctuationCheck= list(map(lambda element: self.punctuation_check(element), ne_List_apostropeCheck))

        #not just number
        ne_List_numCheck=list(filter(lambda candidate: not ((candidate.phraseText.lstrip(string.punctuation).rstrip(string.punctuation).strip()).isdigit()|self.isfloat(candidate.phraseText.lstrip(string.punctuation).rstrip(string.punctuation).strip())|self.ispercent(candidate.phraseText.lstrip(string.punctuation).rstrip(string.punctuation).strip())), ne_List_slangCheck))
        #ne_List_tenseCheck= list(map(lambda element: self.tense_check(element), ne_List_numCheck))
        
        #tracking sudden change in capitalization pattern
        #ne_List_capPatCheck= list(map(lambda element: self.capitalization_change(element), ne_List_tenseCheck))
        
        #check on length
        ne_List_lengthCheck= list(filter(lambda element: element.length<7, ne_List_numCheck))
        
        ne_List_badWordCheck= list(filter(lambda element:((element.phraseText.strip().strip(string.punctuation).lstrip('“‘’”')).rstrip('“‘’”').lower()) not in combined, ne_List_lengthCheck))
        ne_List_allCheck= list(filter(lambda element:(len((element.phraseText.strip().strip(string.punctuation).lstrip('“‘’”')).rstrip('“‘’”'))>1),ne_List_badWordCheck))
        #ne_List_allCheck= list(filter(lambda element: (element.phraseText.lower() not in combined), ne_List_double_Check))
        #q.put(ne_List_allCheck)
        return ne_List_allCheck

#return ne_List_allCheck


# In[319]:


'''This is the main module. I am not explicitly writing it as a function as I am not sure what argument you are 
passing.However you can call this whole cell as a function and it will call the rest of the functions in my module
to extract candidates and features
'''

'''#reads input from the database file and converts to a dataframe. You can change this part accordingly and
#directly convert argument tuple to the dataframe'''

#Inputs: Collection.csv 500Sample.csv 3.2KSample.csv eric_trump.csv

#df_out.to_csv('TweetBase500.csv')
#--------------------------------------PHASE I---------------------------------------------------


# In[ ]:

#--------------------------------------PHASE II---------------------------------------------------
'''set1 = set(['Melania','Trump'])
set2 = set(['Donald','Trump'])
set3 = set(['Jared','Kushner'])

m1 = MinHash(num_perm=200)
m2 = MinHash(num_perm=200)
m3 = MinHash(num_perm=200)
for d in set1:
    m1.update(d.encode('utf8'))

for d in set2:
    m2.update(d.encode('utf8'))
for d in set3:
    m3.update(d.encode('utf8'))

# Create LSH index
lsh = MinHashLSH(threshold=0.0, num_perm=200)
lsh.insert("m2", m2)
lsh.insert("m3", m3)
result = lsh.query(m1)
print("Approximate neighbours with Jaccard similarity", result)


candidates=["donald trump","melania trump", "obama","barack obama","barack"]
listofMinhash=[]
m=MinHash(num_perm=200)
candidate0=set(candidates[0].split())
for d in candidate0:
    m.update(d.encode('utf8'))
listofMinhash.append(m)
lsh = MinHashLSH(threshold=0.0, num_perm=200)
lsh.insert("m2", m2)
for candidate in candidates[1:]:'''
    


# In[ ]:

'''
print ("Shingling articles...")

# The current shingle ID value to assign to the next new shingle we 
# encounter. When a shingle gets added to the dictionary, we'll increment this
# value.
curShingleID = 0

# Create a dictionary of the articles, mapping the article identifier (e.g., 
# "t8470") to the list of shingle IDs that appear in the document.
candidatesAsShingleSets = {};
  
candidateNames = []

t0 = time.time()

totalShingles = 0

for k in range(0, len(sorted_NE_container.keys())):
    # Read all of the words (they are all on one line) and split them by white space.
    words = list(sorted_NE_container.keys())[k].split(" ")
    
    # Retrieve the article ID, which is the first word on the line.  
    candidateID = k
    
    # Maintain a list of all document IDs.  
    candidateNames.append(candidateID)
    
    
    # 'shinglesInDoc' will hold all of the unique shingle IDs present in the current document.
    #If a shingle ID occurs multiple times in the document,
    # it will only appear once in the set (this is a property of Python sets).
    shinglesInCandidate = set()
    
    # For each word in the document...
    for index in range(0, len(words)):
        
        # Construct the shingle text by combining three words together.
        shingle = words[index]
        # Hash the shingle to a 32-bit integer.
        #crc = binascii.crc32("")
        crc = binascii.crc32(bytes(shingle, encoding="UTF-8")) & (0xffffffff)

        # Add the hash value to the list of shingles for the current document. 
        # Note that set objects will only add the value to the set if the set 
        # doesn't already contain it. 
        shinglesInCandidate.add(crc)
    
    # Store the completed list of shingles for this document in the dictionary.
    #print(str(words)+": ")
    #for i in shinglesInCandidate:
     #   print('0x%08x' %i)
    candidatesAsShingleSets[candidateID] = shinglesInCandidate
    
    # Count the number of shingles across all documents.
    totalShingles = totalShingles + (len(words))


# Report how long shingling took.
print ('\nShingling ' + str(str(len(sorted_NE_container.keys()))) + ' candidates took %.2f sec.' % (time.time() - t0))
 
print ('\nAverage shingles per doc: %.2f' % (totalShingles / len(sorted_NE_container.keys())))
'''


# In[ ]:

'''
# =============================================================================
#                 Generate MinHash Signatures
# =============================================================================
numHashes=20
numCandidates=len(sorted_NE_container.keys())
# Time this step.
t0 = time.time()

print ('Generating random hash functions...')

# Record the maximum shingle ID that we assigned.
maxShingleID = 2**32-1
nextPrime = 4294967311


# Our random hash function will take the form of:
#   h(x) = (a*x + b) % c
# Where 'x' is the input value, 'a' and 'b' are random coefficients, and 'c' is
# a prime number just greater than maxShingleID.

# Generate a list of 'k' random coefficients for the random hash functions,
# while ensuring that the same value does not appear multiple times in the 
# list.
def pickRandomCoeffs(k):
    # Create a list of 'k' random values.
    randList = []
    
    while k > 0:
        # Get a random shingle ID.
        randIndex = random.randint(0, maxShingleID) 
        # Ensure that each random number is unique.
        while randIndex in randList:
            randIndex = random.randint(0, maxShingleID) 
        # Add the random number to the list.
        randList.append(randIndex)
        k = k - 1
    return randList

# For each of the 'numHashes' hash functions, generate a different coefficient 'a' and 'b'.   
coeffA = pickRandomCoeffs(numHashes)
coeffB = pickRandomCoeffs(numHashes)

print ('\nGenerating MinHash signatures for all candidates...')

# List of documents represented as signature vectors
signatures =np.ndarray(shape=(20, numCandidates))

# Rather than generating a random permutation of all possible shingles, 
# we'll just hash the IDs of the shingles that are *actually in the document*,
# then take the lowest resulting hash code value. This corresponds to the index 
# of the first shingle that you would have encountered in the random order.

# For each document...
for candidateID in candidateNames:
    
    # Get the shingle set for this document.
    shingleIDSet = candidatesAsShingleSets[candidateID]
  
    # The resulting minhash signature for this document. 
    signature = []
  
    # For each of the random hash functions...
    for i in range(0, numHashes):
        

        # For each of the shingles actually in the document, calculate its hash code
        # using hash function 'i'. 

        # Track the lowest hash ID seen. Initialize 'minHashCode' to be greater than
        # the maximum possible value output by the hash.
        minHashCode = nextPrime + 1

        # For each shingle in the document...
        for shingleID in shingleIDSet:
            # Evaluate the hash function.
            hashCode = (coeffA[i] * shingleID + coeffB[i]) % nextPrime

            # Track the lowest hash code seen.
            if hashCode < minHashCode:
                minHashCode = hashCode

        # Add the smallest hash code value as component number 'i' of the signature.
        signature.append(minHashCode)

    # Store the MinHash signature for this document.
    #signatures.append(signature)
    signatures[:,candidateID]=signature

    # Calculate the elapsed time (in seconds)
    elapsed = (time.time() - t0)
print(list(np.shape(signatures)))
print ("\nGenerating MinHash signatures took %.2fsec" % elapsed)


#print ('\nsignatures stored in a numpy array...')
  
# Creates a N x N matrix initialized to 0.

# Time this step.
t0 = time.time()

# For each of the test documents...
for i in range(10, 11):
#for i in range(0, numCandidates):
    print(list(sorted_NE_container.keys())[i]+": ",end="")
    # Get the MinHash signature for document i.
    signature1 = signatures[i]
    
    # For each of the other test documents...
    for j in range(0, numCandidates):
        if(j!=i):
            # Get the MinHash signature for document j.
            signature2 = signatures[j]

            count = 0
            # Count the number of positions in the minhash signature which are equal.
            for k in range(0, numHashes):
                count = count + (signature1[k] == signature2[k])

            # Record the percentage of positions which matched.    
            estJSim= (count / numHashes)
            #print(estJSim)
            if (estJSim>=0.5):
                print("=>"+list(sorted_NE_container.keys())[j]+", ",end="") 
    print()
# Calculate the elapsed time (in seconds)
elapsed = (time.time() - t0)
        
print ("\nComparing MinHash signatures took %.2fsec" % elapsed)'''


# In[ ]:

'''cap_phrases="Trump:Russia,Afgha"
words=re.split('[,:]', cap_phrases)
print(words)



candidateString='"BS'
p= re.compile(r'(".*?")[^\s]*[\s]*')
indices= (list( p.finditer(candidateString) ))
matches=[]
final=[]
if(indices):
    for index in indices:
        span= list(index.span())
        #print(span[0])
        matches+=[int(span[0]),int(span[1])]
    print(matches)
    final+=[candidateString[0:matches[0]]]
    for i in range(len(matches)-1):
        final+=[(candidateString[matches[i]:matches[i+1]]).strip()]
    final+=[candidateString[matches[-1]:]]
    final=list(filter(lambda strin: strin!="",final))
    final=list(map(lambda strin: strin.strip(),final))
    print(final)'''


# tweets=pd.read_csv("deduplicated_test.csv", header=0, index_col = 0 ,encoding = 'utf-8',delimiter=';')
# tweets=tweets[:1000:]

# Phase1= SatadishaModule()
# for i in range(2):
#      Phase1= SatadishaModule()
# Phase1.extract(tweets,1)