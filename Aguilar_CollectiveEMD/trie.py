import datetime
# import Mention as me

class Trie:

    def __init__(self,text):
        #dictionary of first words
        #print("Node text: "+text)
        self.text=text
        self.path = {}
        #store the actual candidate object here--- not storing features 16
        self.feature_list = [0]*2
        self.value_valid = False

    def updateNode(self, feature):
        self.feature_list.append(feature)
        

    def __setitem__(self, candidateText,origLength, candidateFeatures, batch):
        
        head = candidateText[0]
        if head in self.path:
            node = self.path[head]
        else:
            node = Trie(head)
            self.path[head] = node
            
        if len(candidateText) > 1:
            remains = candidateText[1:]
            node.__setitem__(remains,origLength, candidateFeatures, batch)
        else:
            #INITIALIZATION/UPDATION ROUTINE
            if(node.feature_list[0]==0):
                #initial
                now = datetime.datetime.now()
                now=str(now.hour)+":"+str(now.minute)+":"+str(now.second)
                node.feature_list[1]=origLength
                node.feature_list.append(now)
                node.feature_list.append(batch)
            #common updations for either case
            node.feature_list[0]+=1
            '''for index in [0,1,2,3,4,5,6,7,9,10,11,13]:
                if (candidateFeatures[index]==True):
                    node.feature_list[index+2]+=1
            for index in [8,12]:
                if (candidateFeatures[index]!=-1):
                    node.feature_list[index+2]+=1'''
            node.value_valid = True
            #print("++",node.text)


    def setitem_forAnnotation(self, candidateText):
        
        head = candidateText[0]
        if head in self.path:
            node = self.path[head]
        else:
            node = Trie(head)
            self.path[head] = node
            
        if len(candidateText) > 1:
            remains = candidateText[1:]
            node.setitem_forAnnotation(remains)
        else:
            #INITIALIZATION/UPDATION ROUTINE
            # if(node.feature_list[0]==0):
            #     #initial
            #     now = datetime.datetime.now()
            #     now=str(now.hour)+":"+str(now.minute)+":"+str(now.second)
            #     node.feature_list[1]=origLength
            #     node.feature_list.append(now)
            #     node.feature_list.append(batch)
            # #common updations for either case
            # node.feature_list[0]+=1
            '''for index in [0,1,2,3,4,5,6,7,9,10,11,13]:
                if (candidateFeatures[index]==True):
                    node.feature_list[index+2]+=1
            for index in [8,12]:
                if (candidateFeatures[index]!=-1):
                    node.feature_list[index+2]+=1'''
            node.value_valid = True
            #print("++",node.text)
            

    def __delitem__(self, key):
        head = key[0]
        if head in self.path:
            node = self.path[head]
            if len(key) > 1:
                remains = key[1:]
                node.__delitem__(remains)
            else:
                node.value_valid = False
                node.value = None
            if len(node) == 0:
                del self.path[head]

    def __getitem__(self, candidateText):
        head = candidateText[0]
        #print(self.path.keys())
        if head in self.path:
            #print(head,"head exists")
            node = self.path[head]
        else:
            raise KeyError(candidateText)
        if len(candidateText) > 1:
            remains = candidateText[1:]
            try:
                return node.__getitem__(remains)
            except KeyError:
                raise KeyError(candidateText)
        elif node.value_valid:
            #print("Got here")
            return node.feature_list
        else:
            #print(candidateText,"exists but value not valid")
            raise KeyError(candidateText)

    def __contains__(self, candidateText):
        try:
            self.__getitem__(candidateText)
        except KeyError:
            return False
        return True

    def contains_child(self,text):
        if text in self.path.keys():
            return True
        else:
            return False

    



    def __len__(self):
        n = 1 if self.value_valid else 0
        for k in self.path.keys():
            n = n + len(self.path[k])
        return n

    def get(self, key, default=None):
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def nodeCount(self):
        n = 0
        for k in self.path.keys():
            n = n + 1 + self.path[k].nodeCount()
        return n

    def keys(self, prefix=[]):
        return self.__keys__(prefix)


    def __keys__(self, prefix=[], seen=[]):
        result = []
        if self.value_valid:
            #isStr = True
            val = ""
            for k in seen:
                '''if type(k) != str or len(k) > 2:
                    isStr = False
                    break
                else:'''
                if(val==""):
                    val=k
                else:
                    val= val+" "+k
            result.append(val)
            '''if isStr:
                result.append(val)
            else:
                result.append(prefix)'''
        #else:
            #print("++",self.text)
        if len(prefix) > 0:
            head = prefix[0]
            prefix = prefix[1:]
            if head in self.path:
                nextpaths = [head]
            else:
                nextpaths = []
        else:
            nextpaths = self.path.keys()                
        for k in nextpaths:
            nextseen = []
            nextseen.extend(seen)
            nextseen.append(k)
            result.extend(self.path[k].__keys__(prefix, nextseen))
        return result


    def updateTrie(self, text, ME_EXTR):
        #ME_EXTR.PrintDictionary()
        if(self.text!="ROOT"):
            if text=="":
                thisText=self.text
            else:
                thisText=text+" "+self.text
        else:
            thisText=text
        if(self.value_valid):
            #print(thisText,ME_EXTR.checkInDictionary(thisText))
            self.updateNode(ME_EXTR.checkInDictionary(thisText))
        else:
            self.updateNode(0)
        #print(thisText,self.feature_list)
        for k in self.path.keys():
            self.path[k].updateTrie(thisText,ME_EXTR)
        return

    def displayTrie(self,text,candidateBase):
        #ME_EXTR.PrintDictionary()
        if(self.text!="ROOT"):
            if text=="":
                thisText=self.text
            else:
                thisText=text+" "+self.text
        else:
            thisText=text
        if(self.value_valid):
            #print(thisText,ME_EXTR.checkInDictionary(thisText))
            #dict1 = {'candidate':thisText, 'freq':self.feature_list[0],'length':self.feature_list[1],'cap':self.feature_list[2],'start_of_sen':self.feature_list[3],'abbrv':self.feature_list[4],'all_cap':self.feature_list[5],'is_csl':self.feature_list[6],'title':self.feature_list[7],'has_no':self.feature_list[8],'date':self.feature_list[9],'is_apostrp':self.feature_list[10],'has_inter_punct':self.feature_list[11],'ends_verb':self.feature_list[12],'ends_adverb':self.feature_list[13],'change_in_cap':self.feature_list[14],'topic_ind':self.feature_list[15],'entry_time':self.feature_list[16],'entry_batch':self.feature_list[17],'@mention':self.feature_list[18]}
            # dict1 = {'candidate':thisText, 'freq':self.feature_list[0],'length':self.feature_list[1],'entry_time':self.feature_list[2],'entry_batch':self.feature_list[3]}
            candidateBase.append(thisText)
        
        for k in self.path.keys():
            self.path[k].displayTrie(thisText,candidateBase)
        return candidateBase
        

    def __iter__(self):
        return self.__keys__()
        #for k in self.keys():
            #print(k)
            #yield k
        #raise StopIteration

    def __add__(self, other):
        result = Trie()
        result += self
        result += other
        return result

    def __sub__(self, other):
        result = Trie()
        result += self
        result -= other
        return result

    def __iadd__(self, other):
        for k in other:
            self[k] = other[k]
        return self

    def __isub__(self, other):
        for k in other:
            del self[k]
        return self