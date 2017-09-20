from __future__ import absolute_import
from six.moves import cPickle
import gzip
import random
import numpy as np

import glob, os, csv, re
from collections import Counter
from keras.preprocessing import sequence


def init_vocab(filelist="list_of_grid.txt", emb_size=300):
    

    wordcount={}
    list_of_files = [line.rstrip('\n') for line in open(filelist)]
    for file in list_of_files:
        #print(file) 
        lines = [line.rstrip('\n') for line in open(file + ".EGrid")]
        for line in lines:
            ent = line.split()[0]
            if ent not in wordcount:
                wordcount[ent] = 1
            else:
                wordcount[ent] += 1


    entities =[]
    for ent in wordcount:
        if wordcount[ent] > 2:
            entities.append(ent)

    vocabs = []
    for ent in entities:
        vocabs.append(ent + "_S")
        vocabs.append(ent + "_O")
        vocabs.append(ent + "_X")


    #print len(vocabs)
    vocabs.append("-")
    vocabs.append("0")

    np.random.seed(2017)
    E      = 0.01 * np.random.uniform( -1.0, 1.0, (len(vocabs), emb_size))
    E[len(vocabs)-1] = 0

    return vocabs, E

#loading grid with features
def load_and_numberize_egrids(filelist="list_of_grid.txt", maxlen=15000, w_size=3, E=None, vocabs=None, emb_size=300, perm_num=20):
    # loading entiry-grid data from list of pos document and list of neg document
    if vocabs is None:
        print("Please input vocab list")
        return None

    list_of_files = [line.rstrip('\n') for line in open(filelist)]

    # process postive gird, convert each file to be a sentence
    sentences_1 = []
    sentences_0 = []
    
    for file in list_of_files:
        #print(file) 

        lines = [line.rstrip('\n') for line in open(file + ".EGrid")]
        #f_lines = [line.rstrip('\n') for line in open(file + ".Feats")]

        grid_1 = "0 "* w_size

        for idx, line in enumerate(lines):
            e_trans = get_eTrans_with_Word(line) # merge the grid of positive document  
            if len(e_trans) !=0:
                #print e_trans
                grid_1 = grid_1 + e_trans + " " + "0 "* w_size
        
        p_count = 0
        for i in range(1,perm_num+1): # reading the permuted docs
            permuted_lines = [p_line.rstrip('\n') for p_line in open(file+ ".EGrid" +"-"+str(i))]    
            grid_0 = "0 "* w_size

            for idx, p_line in enumerate(permuted_lines):
                e_trans_0 = get_eTrans_with_Word(p_line)
                if len(e_trans_0) !=0:
                    grid_0 = grid_0 + e_trans_0  + " " + "0 "* w_size

            if grid_0 != grid_1: #check the duplication
                p_count = p_count + 1
                sentences_0.append(grid_0)
            #else:
            #    print(file+ ".EGrid" +"-"+str(i)) // print duplicates permuted docs with original
        
        for i in range (0, p_count): #stupid code
            sentences_1.append(grid_1)


    assert len(sentences_0) == len(sentences_1)

    vocab_idmap = {}
    for i in range(len(vocabs)):
        vocab_idmap[vocabs[i]] = i

    # Numberize the sentences
    X_1 = numberize_sentences(sentences_1, vocab_idmap)
    X_0  = numberize_sentences(sentences_0,  vocab_idmap)
    
    X_1 = adjust_index(X_1, maxlen=maxlen, window_size=w_size)
    X_0  = adjust_index(X_0,  maxlen=maxlen, window_size=w_size)

    X_1 = sequence.pad_sequences(X_1, maxlen)
    X_0 = sequence.pad_sequences(X_0, maxlen)

    return X_1, X_0



def get_eTrans_with_Word(sent):
    x = sent.split()
    
    length = len(x)
    e_occur = x.count('X') + x.count('S') + x.count('O') #counting the number of occurrence of entities
    if length > 80:
        if e_occur < 3:
            return ""
    elif length > 20:
        if e_occur < 2:
            return ""
    
    ent = x[0]
    x = x[1:]
    
    new_x = []

    for role in x:
        if role != "-":
            new_x.append(ent+"_"+role)
        else:
            new_x.append("-")

    return ' '.join(new_x)



#==================================================================================

#get entity transition from a row of Entity Grid
def get_eTrans_with_Feats(sent="",feats="",fn=None):
    x = sent.split()
    
    length = len(x)
    e_occur = x.count('X') + x.count('S') + x.count('O') #counting the number of occurrence of entities
    if length > 80:
        if e_occur < 3:
            return ""
    elif length > 20:
        if e_occur < 2:
            return ""
    
    if fn==None: #coherence model without features
        x = x[1:]
        return ' '.join(x)     

    f = feats.split()
    #print(x[0] + " -- " + f[0])
    assert f[0] == x[0] # checking working on the same entity 
    #print(x[0] + " -- " + f[0])

    x = x[1:]
    f = f[1:]
    x_f = []
    for sem_role in x:
        new_role = sem_role;
        if new_role != '-':
            for i in fn:
                if i ==0 : #adding salience
                    if e_occur == 1:
                        new_role = new_role + "F01"
                    elif e_occur == 2:
                        new_role = new_role + "F02"
                    elif e_occur == 3:
                        new_role = new_role + "F03"
                    elif e_occur >3 :
                        new_role = new_role + "F04"
                else:
                    new_role = new_role + "F" + str(i) + f[i-1] # num feat = idx + 1
  
        x_f.append(new_role)

    return ' '.join(x_f)



#get entity transition from a row of Entity Grid for insertion experiment
def get_eTrans_With_Perm(sent="",feats="",fn=None, perm=[]):
    #print(feats)
    #print(perm)

    x = sent.split()
    x = x[1:]
    length = len(x)
    e_occur = x.count('X') + x.count('S') + x.count('O') #counting the number of coocurence of the entity
    if length > 80:
        if e_occur < 3:
            return ""
    elif length > 20:
        if e_occur < 2:
            return ""
    #need to re-order the entity meaning re-order sentence in a document
    p_x = []
    for i in perm:
        p_x.append(x[i])

    if fn==None: #coherence model without features
       return ' '.join(p_x)


    f = feats.split()
    f = f[1:]

    p_x_f = []
    for sem_role in p_x: # extended coherence model without features
        new_role = sem_role;
        if new_role != '-':
            for i in fn:
                if i ==0 : #adding salience
                    if e_occur == 1:
                        new_role = new_role + "F01"
                    elif e_occur == 2:
                        new_role = new_role + "F02"
                    elif e_occur == 3:
                        new_role = new_role + "F03"
                    elif e_occur >3 :
                        new_role = new_role + "F04"
                else:
                    new_role = new_role + "F" + str(i) + f[i-1] # num feat = idx + 1
  
        p_x_f.append(new_role)

    return ' '.join(p_x_f)


def load_POS_EGrid(filename="", w_size=3, maxlen=1000, vocab_list=None , fn=None ):
    lines = [line.rstrip('\n') for line in open(filename + ".EGrid")]
    f_lines = [line.rstrip('\n') for line in open(filename + ".Feats")]

    grid_1 = "0 "* w_size
    for idx, line in enumerate(lines):
        # merge the grid of positive document 
        e_trans = get_eTrans_with_Feats(sent=line, feats=f_lines[idx], fn=fn)
        if len(e_trans) !=0:
            grid_1 = grid_1 + e_trans + " " + "0 "* w_size
            #print(e_trans)
    #print(grid_1)
    vocab_idmap = {}
    for i in range(len(vocab_list)):
        vocab_idmap[vocab_list[i]] = i

    X_1 = numberize_sentences([grid_1], vocab_idmap)
    X_1 = adjust_index(X_1, maxlen=maxlen, window_size=w_size)
    X_1 =  sequence.pad_sequences(X_1, maxlen)

    return X_1

def load_NEG_EGrid(filename="", w_size=3, maxlen=1000,  vocab_list=None, fn=None, perm=None):
    #print(perm)
    if perm != None:
        lines = [line.rstrip('\n') for line in open(filename + ".EGrid")]
        f_lines = [line.rstrip('\n') for line in open(filename + ".Feats")]


        grid_0 = "0 "* w_size
        for idx, line in enumerate(lines):
            #print(line)
            e_trans_0 = get_eTrans_With_Perm(sent=line,feats=f_lines[idx],fn=fn, perm=perm) # need to include features
            if len(e_trans_0) !=0:
                grid_0 = grid_0 + e_trans_0  + " " + "0 "* w_size
            #print(e_trans_0)
        
        #print(grid_0)
        vocab_idmap = {}
        for i in range(len(vocab_list)):
            vocab_idmap[vocab_list[i]] = i

        X_0 = numberize_sentences([grid_0], vocab_idmap)
        X_0 = adjust_index(X_0, maxlen=maxlen, window_size=w_size)
        X_0 = sequence.pad_sequences(X_0, maxlen)

        return X_0

    else:
        print("no permuted list")
        return None

def load_all(filelist="list_of_grid.txt",fn=None):

    list_of_files = [line.rstrip('\n') for line in open(filelist)]
    print("Using features: " + str(fn))
    vocab = Counter()

    for file in list_of_files:
#        print(file)
        lines = [line.rstrip('\n') for line in open(file + ".EGrid")]
        f_lines = [line.rstrip('\n') for line in open(file + ".Feats")]

        for idx, line in enumerate(lines):
            # merge the grid of positive document 
            e_trans = get_eTrans_with_Feats(sent=line,feats=f_lines[idx],fn=fn)
            # need to update the dictionary here
            if len(e_trans) !=0:
                for wrd in e_trans.split():
                    vocab[wrd] += 1

    vocab = dict (vocab)
    vocab_list = sorted (vocab.keys())
    vocab_list.append('0')
    print "Total vocabulary size in the whole dataset: " + str (len(vocab))        

    return vocab_list

#loading grid with features
def load_and_numberize_Egrid_with_Feats(filelist="list_of_grid.txt", perm_num = 20, maxlen=15000, window_size=3, E=None, vocab_list=None, emb_size=300, fn=None):
    # loading entiry-grid data from list of pos document and list of neg document
    if vocab_list is None:
        print("Please input vocab list")
        return None

    list_of_files = [line.rstrip('\n') for line in open(filelist)]
    
    # process postive gird, convert each file to be a sentence
    sentences_1 = []
    sentences_0 = []
    
    for file in list_of_files:
        #print(file) 

        lines = [line.rstrip('\n') for line in open(file + ".EGrid")]
        f_lines = [line.rstrip('\n') for line in open(file + ".Feats")]

        grid_1 = "0 "* window_size

        for idx, line in enumerate(lines):
            e_trans = get_eTrans_with_Feats(sent=line,feats=f_lines[idx],fn=fn) # merge the grid of positive document 
            if len(e_trans) !=0:
                grid_1 = grid_1 + e_trans + " " + "0 "* window_size
        #print(grid_1)
                
        p_count = 0
        for i in range(1,perm_num+1): # reading the permuted docs
            permuted_lines = [p_line.rstrip('\n') for p_line in open(file+ ".EGrid" +"-"+str(i))]    
            grid_0 = "0 "* window_size

            for idx, p_line in enumerate(permuted_lines):
                e_trans_0 = get_eTrans_with_Feats(sent=p_line, feats=f_lines[idx],fn=fn)
                if len(e_trans_0) !=0:
                    grid_0 = grid_0 + e_trans_0  + " " + "0 "* window_size

            if grid_0 != grid_1: #check the duplication
                p_count = p_count + 1
                sentences_0.append(grid_0)
            #else:
            #    print(file+ ".EGrid" +"-"+str(i)) // print duplicates permuted docs with original
        
        for i in range (0, p_count): #stupid code
            sentences_1.append(grid_1)


    assert len(sentences_0) == len(sentences_1)

    vocab_idmap = {}
    for i in range(len(vocab_list)):
        vocab_idmap[vocab_list[i]] = i

    # Numberize the sentences
    X_1 = numberize_sentences(sentences_1, vocab_idmap)
    X_0  = numberize_sentences(sentences_0,  vocab_idmap)
    
    X_1 = adjust_index(X_1, maxlen=maxlen, window_size=window_size)
    X_0  = adjust_index(X_0,  maxlen=maxlen, window_size=window_size)

    X_1 = sequence.pad_sequences(X_1, maxlen)
    X_0 = sequence.pad_sequences(X_0, maxlen)

    if E is None:
        E      = 0.01 * np.random.uniform( -1.0, 1.0, (len(vocab_list), emb_size))
        E[len(vocab_list)-1] = 0

    return X_1, X_0, E 

#===================================================
#loading data for summary experiment task 
def load_summary_data(filelist="list_of_paris", perm_num = 20, maxlen=15000, window_size=3, E=None, vocab_list=None, emb_size=300, fn=None):
    # loading entiry-grid data from list of pos document and list of neg document
    if vocab_list is None:
        print("Please input vocab list")
        return None

    list_of_pairs = [line.rstrip('\n') for line in open(filelist)]
    
    # process postive gird, convert each file to be a sentence
    sentences_1 = []
    sentences_0 = []
    
    maxLEN_ = 0

    for pair in list_of_pairs:
        #print(pair) 
        #loading pos document
        lines = [line.rstrip('\n') for line in open("./final_data/summary/" +  pair.split()[0] + ".parsed.ner.EGrid")]
        f_lines = [line.rstrip('\n') for line in open("./final_data/summary/" +  pair.split()[0] + ".parsed.ner.Feats")]
        grid_1 = "0 "* window_size
        for idx, line in enumerate(lines):
            e_trans = get_eTrans_with_Feats(sent=line,feats=f_lines[idx],fn=fn) # merge the grid of positive document 
            if len(e_trans) !=0:
                grid_1 = grid_1 + e_trans + " " + "0 "* window_size
        

        #loading neg document
        neg_lines = [line.rstrip('\n') for line in open("./final_data/summary/" + pair.split()[1] + ".parsed.ner.EGrid")]
        neg_f_lines = [line.rstrip('\n') for line in open("./final_data/summary/"  +pair.split()[1] + ".parsed.ner.Feats")]
        grid_0 = "0 "* window_size
        for idx, p_line in enumerate(neg_lines):
                #print(p_line)
                e_trans_0 = get_eTrans_with_Feats(sent=p_line, feats=neg_f_lines[idx],fn=fn)
                if len(e_trans_0) !=0:
                    grid_0 = grid_0 + e_trans_0  + " " + "0 "* window_size

        #find max length for the copus
        if len(grid_1) > maxLEN_:
            maxLEN_ = len(grid_1)

        if len(grid_0) > maxLEN_:
            maxLEN_ = len(grid_0)


        sentences_1.append(grid_1)        
        sentences_0.append(grid_0)
        

    assert len(sentences_0) == len(sentences_1)

    vocab_idmap = {}
    for i in range(len(vocab_list)):
        vocab_idmap[vocab_list[i]] = i

    # Numberize the sentences
    X_1 = numberize_sentences(sentences_1, vocab_idmap)
    X_0  = numberize_sentences(sentences_0,  vocab_idmap)
    
    X_1 = adjust_index(X_1, maxlen=maxlen, window_size=window_size)
    X_0  = adjust_index(X_0,  maxlen=maxlen, window_size=window_size)

    X_1 = sequence.pad_sequences(X_1, maxlen)
    X_0 = sequence.pad_sequences(X_0, maxlen)

    if E is None:
        E      = 0.01 * np.random.uniform( -1.0, 1.0, (len(vocab_list), emb_size))
        E[len(vocab_list)-1] = 0

    return X_1, X_0, E, maxLEN_
#===================================================


#loading the grid with normal CNN
def load_and_numberize_Egrid(filelist="list_of_grid.txt", perm_num = 3, maxlen=None, window_size=3, ignore=0):
    # loading entiry-grid data from list of pos document and list of neg document
    list_of_files = [line.rstrip('\n') for line in open(filelist)]
    
    # process postive gird, convert each file to be a sentence
    sentences_1 = []
    sentences_0 = []

    e_count_list = []

    for file in list_of_files:
        #print(file)
        lines = [line.rstrip('\n') for line in open(file)]

        grid_1 = "0 "* window_size
        e_count = 0
        for line in lines:
            # merge the grid of positive document 
            e_trans = get_eTrans(sent=line)
            if len(e_trans) !=0:
                grid_1 = grid_1 + e_trans + " " + "0 "* window_size
                e_count = e_count + 1
        	
        p_count = 0
        for i in range(1,perm_num+1): # reading the permuted docs
            permuted_lines = [p_line.rstrip('\n') for p_line in open(file+"-"+str(i))]    
            grid_0 = "0 "* window_size
            for p_line in permuted_lines:
                e_trans_0 = get_eTrans(sent=p_line)
                if len(e_trans_0) !=0:
                    grid_0 = grid_0 + e_trans_0  + " " + "0 "* window_size

            if grid_0 != grid_1:
                p_count = p_count + 1
                sentences_0.append(grid_0)
        
        for i in range (0, p_count): #stupid code
            sentences_1.append(grid_1)
            
       #update new number of entity
        e_count_list.append(e_count)

    assert len(sentences_0) == len(sentences_1)
#    with open('e_count_list_after.txt','w') as f:
#        f.write('\n'.join([str(n) for n in e_count_list])+'\n')

    # numberize_data
    vocab_list = ['0','S','O','X','-']
    vocab_idmap = {}
    for i in range(len(vocab_list)):
        vocab_idmap[vocab_list[i]] = i

     # Numberize the sentences
    X_1 = numberize_sentences(sentences_1, vocab_idmap)
    X_0  = numberize_sentences(sentences_0,  vocab_idmap)
    
    X_1 = adjust_index(X_1, maxlen=maxlen, ignore=ignore, window_size=window_size)
    X_0  = adjust_index(X_0,  maxlen=maxlen, ignore=ignore, window_size=window_size)

    return X_1, X_0


def load_embeddings(emb_size=300):
    # maybe we have to load a fixed embeddeings for each S,O,X,- the representation of 0 is zeros vector
    np.random.seed(2016)
    E      = 0.01 * np.random.uniform( -1.0, 1.0, (5, emb_size))
    E[0] = 0
    return E   
 

def numberize_sentences(sentences, vocab_idmap):  

    sentences_id=[]  

    for sid, sent in enumerate (sentences):
        tmp_list = []
        #print(sid)
        for wrd in sent.split():
            if wrd in vocab_idmap:
                wrd_id = vocab_idmap[wrd]  
            else:
                wrd_id = 0
            tmp_list.append(wrd_id)

        sentences_id.append(tmp_list)

    return sentences_id  

def adjust_index(X, maxlen=None, window_size=3):

    if maxlen: # exclude tweets that are larger than maxlen
        new_X = []
        for x in X:

            if len(x) > maxlen:
                #print("************* Maxlen of whole dataset: " + str(len(x)) )
            	tmp = x[0:maxlen]
            	tmp[maxlen-window_size:maxlen] = ['0'] * window_size
            	new_X.append(tmp)
            else:
                new_X.append(x)

        X = new_X

    return X





