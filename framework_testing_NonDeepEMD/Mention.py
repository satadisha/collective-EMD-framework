from collections import defaultdict
import string
import re
import copy
#import editdistance

class Mention():
    def __init__(self, plain_mention, alias, splitted_mention, easiness):
        self.plain_mention = plain_mention
        self.alias = alias
        self.splitted_mention = splitted_mention
        self.easiness = easiness
        self.AliasDict = AliasDictionary()


        
class AliasDictionary():

    def __init__(self):
        self.Dict= defaultdict(lambda: [[],[],[],0])

    def NewEntry(self,ment):
        if(ment.easiness=="easy"):
            self.Dict[ment.alias][0].append(ment)
            self.Dict[ment.alias][3]=self.Dict[ment.alias][3]+1
        if(ment.easiness=="medium"):
            self.Dict[ment.alias][1].append(ment)
            self.Dict[ment.alias][3]=self.Dict[ment.alias][3]+1
        if(ment.easiness=="difficult"):
            self.Dict[ment.alias][2].append(ment)
            self.Dict[ment.alias][3]=self.Dict[ment.alias][3]+1
            

    def PrintDictionary(self):
        for key, value in self.Dict.items():
            for easiness_list in value:

                #print(easiness_list)
                if(type(easiness_list)is list):
                    if(len(easiness_list)>0):
                        print(key,value[3])
                        if(easiness_list==value[0]):
                            print("Easy")
                        if(easiness_list==value[1]):
                            print("Medium")
                        if(easiness_list==value[2]):
                            print("Difficult")
                            
                        for easiness in easiness_list:

                            print("\t"+easiness.plain_mention)
                            print(" ")
            print("\n")
            
    def checkInAliasDict(self,candidateString):
        alias="".join(candidateString.split())
        if alias in self.Dict:
            return self.Dict[alias][3]
        else:
            return 0
        

class Mention_Extraction:
    def __init__(self):
        self.AliasDict = AliasDictionary()
        

    def ComputeAll(self,plain_mention_list):
        #print(plain_mention_list)
        if(len(plain_mention_list)>0):
            #print("Entering")

            for plain_mention in plain_mention_list:
                strip_pl_ment=plain_mention.strip( '@')



                #underscore
                if("_" in plain_mention):
                    splitted_mention=strip_pl_ment.split("_")
                    easiness="easy"

                #if all characters are is uppercase
                elif(plain_mention.isupper() or plain_mention.islower()):
                    splitted_mention=[strip_pl_ment]
                    easiness="medium"

                # number case
                elif(len(re.findall('[a-zA-Z][^A-Z]*', plain_mention)) and self.hasNumbers(plain_mention)):
                    
                    split_holder=[]
                    splitted_mention=re.findall('[a-zA-Z][^A-Z]*', plain_mention)
                    for word in splitted_mention:
                        parts = re.split('(\d.*)',word)
                        easiness="difficult"
                        parts = list(filter(None, parts))
                        for part in parts:
                            split_holder.append(part)

                    splitted_mention = copy.deepcopy(split_holder)

                #uppercase
                elif(len(re.findall('[a-zA-Z][^A-Z]*', plain_mention))):
                    splitted_mention=re.findall('[a-zA-Z][^A-Z]*', plain_mention)
                    easiness="easy"

                # some more model needs to be added.
                else:
                    continue


                alias=self.ComputeAlias(splitted_mention)

                ment=Mention(plain_mention,alias,splitted_mention,easiness)
                #print("==",ment)

                self.AliasDict.NewEntry(ment)            

    def PrintDictionary(self):
        self.AliasDict.PrintDictionary()
        
    def checkInDictionary(self,candidateString):
        return self.AliasDict.checkInAliasDict(candidateString)


    def CheckMatches(self,dict2):
        #print(type(dict2))
        alias_holder_sat=[]
        alias_holder_ment=[]

        for key, value in dict2:
            alias2=key
            
            alias2=alias2.split(" ")
            alias=''.join(alias2)
            #alias = key.translate(str.maketrans('','',string.punctuation))            
            alias=alias.lower()
        
            alias_holder_sat.append(alias)
            #print(alias)

        
        for key, value in self.AliasDict.Dict.items():
            alias_holder_ment.append(key)

        print(alias_holder_sat)
        print(alias_holder_ment)
        
        
        for alias in alias_holder_sat:
            for alias2 in alias_holder_ment:
                #print(alias2,alias)
                if(alias == alias2):
                    print(alias)
        return

    def EditDistance(self,dict2):
        alias_holder_sat=[]
        alias_holder_ment=[]

        for key, value in dict2:
            alias2=key
            
            alias2=alias2.split(" ")
            alias=''.join(alias2)
            #alias = key.translate(str.maketrans('','',string.punctuation))            
            alias=alias.lower()
        
            alias_holder_sat.append(alias)
            #print(alias)

        
        for key, value in self.AliasDict.Dict.items():
            alias_holder_ment.append(key)


        
        
        edit_distance_dict={}
        for alias in alias_holder_sat:
            for alias2 in alias_holder_ment:
                
                edit_distance_dict[alias+" "+alias2]=self.CalculateEditDistance(alias,alias2)

        for key, value in edit_distance_dict.items():
            if(value<3):
                print(key,value)

    def NewEntry(self):
        self.AliasDict.NewEntry(self)
        return
    
    def CalculateEditDistance(self,s1, s2):
        if len(s1) < len(s2):
            return self.CalculateEditDistance(s2, s1)

        # len(s1) >= len(s2)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + (10*(ord(s1[j]) in xrange(65,91)|ord(s1[j]) in xrange(97,123)|s1[j].isdigit())) # j+1 instead of j since previous_row and current_row are one character longer
                deletions = current_row[j] + (10*(s2[j].isalpha()|s2[j].isdigit()))       # than s2
                substitutions = previous_row[j] + (10*(c1 != c2))
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]



    def hasNumbers(self,inputString):
        return any(char.isdigit() for char in inputString)    

    def ComputeAlias(self,splitted_mention):
        if(type(splitted_mention) is list):
            alias=''.join(splitted_mention)
            alias = alias.translate(str.maketrans('','',string.punctuation))
            alias=alias.lower()
            alias=alias.strip("…")
            return alias