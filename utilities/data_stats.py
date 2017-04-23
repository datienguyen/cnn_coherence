from __future__ import absolute_import
from six.moves import cPickle
import gzip
import random
import numpy as np

import glob, os, csv, re
from collections import Counter

def find_len(sent=""):
    x = sent.split()
    return len(x) -1

def sent_stats(sent=""):
    x = sent.split()
    count = x.count('X') + x.count('S') + x.count('O') #counting the number of entities
    word = x[0].strip()
    x = x[1:]
    length = len(x)
    return word, ' '.join(x), count 

def update_dict(y_dict, w):
    if y_dict.has_key(w):
        y_dict[w] = y_dict[w] + 1
    else:
        y_dict[w] = 1
    return y_dict

def do_stats(filelist="list_of_grid.txt"):
    # loading entiry-grid data from list of pos document and list of neg document
    list_of_files = [line.rstrip('\n') for line in open(filelist)]
    
    e_num_list = []
    sent_num_list = []
    
    p_all = 0

    dict_1 = {}
    dict_2 = {}
    dict_3 = {}
    dict_4 = {}
    dict_x = {}
            
    count_1 = 0
    count_2 = 0
    count_3 = 0
    count_4 = 0
    count_x = 0

    for file in list_of_files:
        #print(file)
        lines = [line.rstrip('\n') for line in open(file)]
        
        e_num = len(lines)
        sent_num = find_len(sent=lines[1])

        e_num_list.append(e_num)
        sent_num_list.append(sent_num)

        e_1 = ""
        
        for line in lines:
            # do something here
            word, e_trans_1, count = sent_stats(sent=line)
            e_1 = e_1 + e_trans_1 + " " 
            if count ==1:
                count_1 = count_1 + 1
                dict_1 = update_dict(dict_1,word)
            elif count ==2:
                count_2 = count_2 + 1
                dict_2 = update_dict(dict_2,word)
            elif count ==3:
                count_3 = count_3 + 1
                dict_3 = update_dict(dict_3,word)
            elif count == 4:
                count_4 = count_4 + 1
                dict_4 = update_dict(dict_4,word)
            else:
                count_x = count_x + 1
                dict_x = update_dict(dict_x,word)

        	
        p_count = 0
        for i in range(1,21): # reading the permuted docs
            permuted_lines = [p_line.rstrip('\n') for p_line in open(file+"-"+str(i))]    
            e_0 = "" 
            for p_line in permuted_lines:
                word, e_trans_0, count = sent_stats(sent = p_line)
                e_0 = e_0 + e_trans_0 + " "
            
            if e_0 != e_1:
                p_count = p_count + 1
            else:
                print (file + "-" + str(i))
        
        p_all = p_all + p_count   
        
    
    with open(filelist + '.sent_num_list','w') as f:
         f.write('\n'.join([str(n) for n in sent_num_list])+'\n')

    with open(filelist + '.e_num_list','w') as f:
        f.write('\n'.join([str(n) for n in e_num_list])+'\n')
        

    f.close()

    print(count_1)
    print(count_2)
    print(count_3)
    print(count_4)
    print(count_x)

    print('----------------')
    print(p_all)
    
    return p_all  



p_all = do_stats(filelist="./final_data/list.train_dev.path")



