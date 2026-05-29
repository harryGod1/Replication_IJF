from sklearn.metrics import roc_curve, auc
import matplotlib as mpl  
import matplotlib.pyplot as plt
import numpy as nplanced
import sklearn
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer 
import string
import math
from sklearn.metrics import roc_curve, auc
import matplotlib as mpl  
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer 
import string
import math
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import tensorflow.compat.v1 as tf
import numpy as np
import sys
from sklearn.metrics import roc_auc_score
import random
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
import os
import time
import datetime
import signal
import math
import matplotlib.pyplot as plt
import statistics
from sklearn.metrics import roc_curve, auc
import matplotlib as mpl  
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer 
import string
import math
import csv
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#The following example code automates the processing of model predictions, eliminating the need for readers to manually feed quarterly data into the model for test output generation. 
#Readers may adjust the number of test samples as needed
ym = 2004
q = 1
overall_PRS = 0
overall_AUC = 0
overall_BS = 0
n_vintage = 1 # number of sub datasets users want to test

#automatically run sub dataests(2004 to 2015) to get the final evaluations in each quarter.
for prs in range(n_vintage):
    from sklearn import preprocessing
    TRAING_TIME = 15
    SHUFFLE = True
    LOAD_LITTLE_DATA = False
    show_survival_curve = False

    class SparseData():

        def shuffle(self):
            if SHUFFLE:
                np.random.shuffle(self.index)
                print('shuffle!!!!!')
            return self.data[self.index], self.seqlen[self.index], self.labels[self.index], self.data2[self.index]

        def __init__(self, INPUT_FILE, win, all,discount,fixed_batch_size):
            self.data = []
            self.data2 = []
            self.data3 = [] #for statistics
            self.data4 = [] #for statistics
            self.labels = []
            self.labels2 = []
            self.deli = []
            self.deli2 = [] #for statistics
            self.seqlen = []
            self.load_data = []
            self.seq = []
            self.final_data = []
            current_deli = []
            last_seq = ''
            new_seq = ''
            last_default = 0
            new_default = 0
            is_default = 0
            max_seqlen = 0
            is_break = 0
            len_count = 0

            self.fixed_batch_size = fixed_batch_size
            self.vt = preprocessing.LabelEncoder()
            if(prs<48):
                self.vt.fit([200401,200402,200403,200404,200501,200502,200503,200504,200601,200602,200603,200604,200701,200702,200703,200704,200801,
                       200802,200803,200804,200901,200902,200903,200904,201001,201002,201003,201004,201101,201102,201103,201104,201201,201202,201203,
                       201204,201301,201302,201303,201304])
            else:
                self.vt.fit([201601,201602,201603,201604,201701,201702,201703,201704,201801,
                       201802,201803,201804,201901,201902,201903,201904,202001,202002,202003,202004,202101,202102,202103,202104,202201,202202,202203,
                       202204,202301,202302,202303,202304,202401,202402,202403])
            fi = open(INPUT_FILE, 'r')
            COUNT = 0
            max_d = -1
            self.finish_epoch = False

            for line in fi:
                self.load_data.append(line)
            last_seq = self.load_data[0].split(' ')[11]  

            for i in range(len(self.load_data)):
                #if COUNT > 10000 and LOAD_LITTLE_DATA:
                    #break\n",
                len_count = len_count+1
                COUNT += 1
                s = self.load_data[i].split(' ')
                slen = len(s)
                if(s[3] != 999):# added at 2024/12/22, for processing th missing value of DTI
                    #delinquency,lag 3 months
                    if(int(s[slen-5])<=2):
                        self.deli.append(0)
                        self.deli2.append(0)
                    else:
                        s_p3 = self.load_data[i-3].split(' ')
                        self.deli.append(s_p3[slen-1])
                        self.deli2.append(s_p3[slen-1])
                    current_deli.append(int(s[slen-1]))
                    self.final_data.append(s)


                    new_seq = s[11] 

                    if(int(s[slen-2]) == 0):
                        self.labels2.append([1., 0.])

                    else:
                        self.labels2.append([0., 1.])

                    if(int(s[slen-2]) == 1):
                        is_default = 1
                        #print('is_default')

                    if(len(last_seq) ==  0):
                        last_seq = s[11] 


                    if(str(new_seq) == str(last_seq) and int(s[slen-5]) > max_seqlen):
                        max_seqlen = int(s[slen-5])



                    if(str(new_seq) == str(last_seq) and len_count > max_seqlen + 1):
                        #print(load_data[i-1])
                        #print(load_data[i])
                        #print(self.labels2[len(self.labels2)-1])
                        del self.labels2[len(self.labels2)-1]
                        del self.deli[len(self.deli)-1]
                        del current_deli[len(current_deli)-1]
                        #print('last one: ',self.final_data[len(self.final_data)-2])
                        #print('delete current: ',self.final_data[len(self.final_data)-1])
                        del self.final_data[len(self.final_data)-1]
                        #print(load_data[i-1])
                        #print(self.labels2[len(self.labels2)-1])
                        #print('Exception3!')
                    elif(str(new_seq) == str(last_seq) and is_default == 0 and int(s[slen-5]) < max_seqlen ):
                        del self.labels2[len(self.labels2)-1]
                        del self.deli[len(self.deli)-1]
                        del current_deli[len(current_deli)-1]
                        del self.final_data[len(self.final_data)-1]
                        #print('Exception1!')
                    elif(str(new_seq) == str(last_seq) and is_default == 1 and int(s[slen-2]) == 0):
                        #print(load_data[i-1])
                        #print(load_data[i])
                        #print(self.labels2[len(self.labels2)-1])
                        del self.labels2[len(self.labels2)-1]
                        del self.deli[len(self.deli)-1]
                        del current_deli[len(current_deli)-1]
                        #print('last one: ',self.final_data[len(self.final_data)-2])
                        #print('delete current: ',self.final_data[len(self.final_data)-1])
                        del self.final_data[len(self.final_data)-1]
                        #print(load_data[i-1])
                        #print(self.labels2[len(self.labels2)-1])
                        #print('Exception2!')
                    elif(str(new_seq) != str(last_seq)):
                        s = self.load_data[i-1].split(' ')
                        slen = len(s)
                        t_indices = []
                        t_indices2 = []
                        #max_d = max(td,max_d)
                        t_indices.append(float(s[0]))
                        t_indices.append(float(s[3]))
                        t_indices.append(float(s[4]))
                        t_indices.append(float(s[5]))
                        t_indices.append(float(s[6]))
                        t_indices.append(float(s[10]))

                        #t_indices.append(vt.transform([int(s[13])]))

                        t_indices2.append(s[2])
                        t_indices2.append(s[8])
                        t_indices2.append(s[9])
                        t_indices2.append(self.vt.transform([int(s[13])])[0])
                        #t_indices2.append(self.vt.transform([201304])[0])

                        if((max_seqlen+1 != (len(current_deli)-1)) or (max_seqlen == 0)):
                            is_break = 1
                            #print('max_seqlen+1:',max_seqlen+1)
                            #print('current_deli:',len(current_deli)-1)
                            #print('current_seq:',self.final_data[len(self.final_data)-2])

                            #important!!!, cause to wrong training: need to delete the from the last index, not the current index
                            del self.deli[len(self.deli)-len(current_deli):len(self.deli)-1]
                            del self.labels2[len(self.labels2)-len(current_deli):len(self.labels2)-1]
                            del self.final_data[len(self.final_data)-len(current_deli):len(self.final_data)-1]

                        if(is_break != 1):
                            self.seq.append(last_seq)
                            self.data2.append(t_indices)
                            self.data.append(t_indices2) 
                            self.data4.append(t_indices2)

                            #default = 1
                            if(is_default == 1):
                                self.labels.append([0., 1.])
                            #default = 0
                            else:
                                self.labels.append([1., 0.])


                            #e.g., s[slen-3]=7,means at 7th quarter, but totally 8 quarters
                            #self.seqlen.append(int(s[slen-4])+1)

                            #exclude the exception of default = 0 after default = 1 within one account


                            #case for solving initial state problem
                            self.seqlen.append(max_seqlen+41)

                        is_default = 0
                        is_break = 0
                        max_seqlen = 0
                        len_count = 0

                        current_deli = []
                        #cleaned but need to return this current data(current data belongs to the new seq)
                        s = self.load_data[i].split(' ')
                        slen = len(s)
                        current_deli.append(int(s[slen-1])) 

                    last_seq = new_seq

            if(max_seqlen != 0):
                s = self.load_data[len(self.load_data)-1].split(' ')
                slen = len(s)
                t_indices = []
                t_indices2 = []
                #max_d = max(td,max_d)
                t_indices.append(float(s[0]))
                t_indices.append(float(s[3]))
                t_indices.append(float(s[4]))
                t_indices.append(float(s[5]))
                t_indices.append(float(s[6]))
                t_indices.append(float(s[10]))

                #t_indices.append(vt.transform([int(s[13])]))

                t_indices2.append(s[2])
                t_indices2.append(s[8])
                t_indices2.append(s[9])
                t_indices2.append(self.vt.transform([int(s[13])])[0])
                #t_indices2.append(self.vt.transform([201304])[0])

                if(max_seqlen+1 != len(current_deli)):
                    is_break = 1
                    del self.deli[len(self.deli)-len(current_deli):len(self.deli)]
                    del self.labels2[len(self.labels2)-len(current_deli):len(self.labels2)]
                    del self.seq[len(self.seq)-len(current_deli):len(self.seq)]


                if(is_break != 1):
                    self.seq.append(last_seq)
                    self.data2.append(t_indices)
                    self.data.append(t_indices2) 
                    self.data4.append(t_indices2)

                    #default = 1
                    if(is_default == 1):
                        self.labels.append([0., 1.])
                    #default = 0
                    else:
                        self.labels.append([1., 0.])


                    #e.g., s[slen-3]=7,means at 7th quarter, but totally 8 quarters
                    #self.seqlen.append(int(s[slen-4])+1)

                    #exclude the exception of default = 0 after default = 1 within one account


                    #case for solving initial state problem
                    self.seqlen.append(max_seqlen+41)

                last_seq = new_seq
                is_break = 0
                is_default = 0
                max_seqlen = 0
                current_deli = []
                #print('max_seqlen')

            #data:3 catagoty variables; data2: 6 other variables



            #need modify self.data2
            for i in range(len(self.data)):
                #FTHF:N=1,Y=2,9=3
                #OS:P=1,I=2,S=3,9=4
                if(self.data[i][0]=='P'):
                    self.data[i][0]=0
                if(self.data[i][0]=='I'):
                    self.data[i][0]=1
                if(self.data[i][0]=='S'):
                    self.data[i][0]=2   
                #if(x[i][2]=='9'):
                    #x[i][2]=4 
                #Channel:R=1,B=2,C=3,T=4,9=5
                #PT:PU=1,SF=2,CO=3,MH&CP&9=4
                if(self.data[i][1]=='PU'):
                    self.data[i][1]=0
                if(self.data[i][1]=='SF'):
                    self.data[i][1]=1
                if(self.data[i][1]=='CO'):
                    self.data[i][1]=2   
                if(self.data[i][1]=='MH'):
                    self.data[i][1]=1 
                if(self.data[i][1]=='CP'):
                    self.data[i][1]=1 
                #if(x[i][8]=='9'):
                    #x[i][8]=4 
                #LP:P=1,C=2,N=3,R=4,9=5
                if(self.data[i][2]=='P'):
                    self.data[i][2]=0
                if(self.data[i][2]=='C'):
                    self.data[i][2]=1
                if(self.data[i][2]=='N'):
                    self.data[i][2]=2   
                #if(x[i][9]=='R'):
                    #x[i][9]=4 
                #if(x[i][9]=='9'):
                    #x[i][9]=5 
                #NB:1=1,2=2,9=3
                # for NB,the process is different
                if(int(self.data2[i][5])==1):
                    self.data2[i][5]=0.0 
                if(int(self.data2[i][5])==2):
                    self.data2[i][5]=1.0 
                if(int(self.data2[i][5])==99):
                    self.data2[i][5]=0.5  

            #self.max_d = max_d
            fi.close()
            self.size = len(self.data)
            self.data = np.array(self.data)
            self.data2 = np.array(self.data2)
            self.labels = np.array(self.labels)
            self.labels2 = np.array(self.labels2)
            self.deli = np.array(self.deli)
            self.seq = np.array(self.seq)
            self.seqlen = np.array(self.seqlen)

            #for statistics
            self.data3 = self.data2


            #standard transformation
            self.data2 = scaler.transform(self.data2)

            #print('data2',self.data2[0])
            #print('data',self.data[0])


            #print("data size ", self.size, "\n")
            self.index = list(range(0, self.size))
            #self.data, self.seqlen, self.labels = self.shuffle()
            self.batch_id = 0
            self.batch_all_id = 0
            self.batch_all_deli_id = 0
            self.batch_all_seq_id = 0
            self.batch_all_count = 0




        def next(self, batch_size):
            #if self.batch_id + batch_size > len(self.data):
                #self.data, self.seqlen, self.labels, self.data2 = self.shuffle()
                #self.finish_epoch = True

                #b = []
                #for i in range(len(self.data)):
                    ##for j in range(len(self.seqlen)-1):
                    #for j in range(self.seqlen[i]-41):
                        #b.append([1., 0.])
                    #b.append(self.labels[i])
                #self.labels2 = b        

                #self.batch_id = 0
                #self.batch_all_id = 0
                #self.batch_all_count = 0




            if self.batch_id + batch_size > len(self.data):
                self.batch_id = 0
                self.batch_all_id = 0
                self.batch_all_deli_id = 0
                self.batch_all_seq_id = 0
                self.batch_all_count = 0

            batch_data = self.data[self.batch_id:self.batch_id + batch_size]
            batch_data2 = self.data2[self.batch_id:self.batch_id + batch_size]
            batch_labels = self.labels[self.batch_id:self.batch_id + batch_size]
            batch_seqlen = self.seqlen[self.batch_id:self.batch_id + batch_size]
            batch_seq = self.seq[self.batch_id:self.batch_id + batch_size]

            batch_data = batch_data.tolist()
            batch_data2 = batch_data2.tolist()
            batch_labels = batch_labels.tolist()
            batch_seqlen = batch_seqlen.tolist()
            batch_seq = batch_seq.tolist()

            if(batch_size < self.fixed_batch_size):
                a = []
                b = []
                for i in range(self.fixed_batch_size - batch_size):
                    a = []
                    b = []
                    a.append(-100)
                    a.append(-100)
                    a.append(-100)
                    a.append(-100)
                    a.append(-100)
                    a.append(-100)

                    b.append(-100)
                    b.append(-100)
                    b.append(-100)
                    if(prs<48):
                        b.append(self.vt.transform([200401])[0])
                    else:
                        b.append(self.vt.transform([201601])[0])
                    batch_data2.append(a)
                    batch_data.append(b)
                    batch_labels.append([1.,0.])
                    batch_seqlen.append(1+40)


            all_count = 0
            #for i in range(512):
                #all_count = all_count + self.seqlen[i]

            #for i in range(512,512+32-1):
                #print(self.labels2[all_count:all_count+self.seqlen[i]])
                #all_count = all_count + self.seqlen[i]
                #print('=======================================')



            for i in range(batch_size):
                all_count = all_count + batch_seqlen[i] - 40



            batch_labels2 = self.labels2[self.batch_all_id:self.batch_all_id + all_count]
            #batch_labels2 = np.array(batch_labels2)
            batch_labels2 = batch_labels2.tolist()

            if(batch_size < self.fixed_batch_size):
                for i in range(self.fixed_batch_size - batch_size):
                    batch_labels2.append([1.,0.])

            batch_deli = self.deli[self.batch_all_deli_id:self.batch_all_deli_id + all_count]
            batch_deli = batch_deli.tolist()

            if(batch_size < self.fixed_batch_size):
                for i in range(self.fixed_batch_size - batch_size):
                    batch_deli.append(0)



            self.batch_id = self.batch_id + batch_size
            self.batch_all_id = self.batch_all_id + all_count
            self.batch_all_deli_id = self.batch_all_deli_id + all_count

            return np.array(batch_data), np.array(batch_data2),np.array(batch_labels), np.array(batch_seqlen), np.array(batch_labels2),np.array(batch_deli),np.array(batch_seq)

        def next_by_id(self,batch_id,batch_all_id,batch_size):
            batch_data = self.data[self.batch_id:self.batch_id + batch_size]
            batch_data2 = self.data2[self.batch_id:self.batch_id + batch_size]
            batch_labels = self.labels[self.batch_id:self.batch_id + batch_size]
            batch_seqlen = self.seqlen[self.batch_id:self.batch_id + batch_size]

            all_count = 0
            #for i in range(512):
                #all_count = all_count + self.seqlen[i]

            #for i in range(512,512+32-1):
                #print(self.labels2[all_count:all_count+self.seqlen[i]])
                #all_count = all_count + self.seqlen[i]
                #print('=======================================')

            for i in range(batch_size):
                all_count = all_count + batch_seqlen[i]

            batch_labels2 = self.labels2[self.batch_all_id:self.batch_all_id + all_count]

            self.batch_id = self.batch_id + batch_size
            self.batch_all_id = self.batch_all_id + all_count
            return np.array(batch_data), np.array(batch_data2),np.array(batch_labels), np.array(batch_seqlen), np.array(batch_labels2)

    class biSparseData():
        def __init__(self, INPUT_FILE, discount):
            random.seed(time.time())
            self.winData = SparseData(INPUT_FILE, True, False,discount)
            self.size = self.winData.size

        def next(self, batch):
            #win = int(random.random() * 100) % 11 == 1# todoe 1/10 get windata
            win = True
            if win:
                a, b, c, d, e, f, g= self.winData.next(batch)
                return a, b, c, d, e, f, g, True
            else:
                a, b, c, d, e = self.loseData.next(batch)
                return a, b, c, d, e, f, g, False

    class BASE_RNN():

        train_data = None
        def init_matrix(self, shape):
            return tf.random_normal(shape, stddev=0.1)

        def __init__(self,  EMB_DIM = 16,
                            FEATURE_SIZE = 3,
                            FEATURE_SIZE2 = 7,
                            BATCH_SIZE = 128,
                            MAX_DEN = 1580000,
                            MAX_SEQ_LEN = 350,
                            TRAING_STEPS = 100000,
                            STATE_SIZE = 64,
                            LR = 0.001,
                            GRAD_CLIP = 5.0,
                            L2_NORM = 0.01,
                            INPUT_FILE = "2997",
                            ALPHA = 1.0,
                            BETA = 0.2,
                            ADD_TIME_FEATURE=False,
                            MIDDLE_FEATURE_SIZE = 30,
                            LOG_FILE_NAME=None,
                            FIND_PARAMETER = False,
                            SAVE_LOG=True,
                            OPEN_TEST=True,
                            ONLY_TRAIN_ANLP=False,
                            LOG_PREFIX="",
                            TEST_FREQUENT=False,
                            ANLP_LR = 0.001,
                            DNN_MODEL = False,
                            QRNN_MODEL = False,
                            GLOAL_STEP = 0,
                            COV_SIZE = 1,
                            DOUBLE_QRNN = False,
                            SHOW_SURVIVAL_CURSE = False,
                            ANLP_ROUND_ROBIN_RATE = 0.2,
                            DISCOUNT = 1,
                            SURVIVAL_RATE = [],
                            TRUE_LABEL2 = [],
                            PREDICTED_LABEL2 = [],
                            WEIGHT = [],
                            KEEP_PRO = 0.5,
                            NUM_LAYERS = 2,
                            W_attention = []
    ):
            self.train_survival_rate = []
            self.TRUE_LABEL2 = []
            self.WEIGHT = []
            self.PREDICTED_LABEL2 = []
            self.KEEP_PRO = KEEP_PRO
            self.NUM_LAYERS = NUM_LAYERS
            self.DISCOUNT = DISCOUNT
            self.DOUBLE_QRNN = DOUBLE_QRNN
            self.ANLP_ROUND_ROBIN_RATE = ANLP_ROUND_ROBIN_RATE
            self.QRNN_MODEL = QRNN_MODEL
            self.global_step = GLOAL_STEP
            self.DNN_MODEL = DNN_MODEL
            self.ANLP_LR = ANLP_LR
            self.TEST_FREQUENT = TEST_FREQUENT
            self.ONLY_TRAIN_ANLP = ONLY_TRAIN_ANLP
            self.FIND_PARAMETER = FIND_PARAMETER
            self.add_time_feature = ADD_TIME_FEATURE
            self.MIDDLE_FEATURE_SIZE = MIDDLE_FEATURE_SIZE
            tf.reset_default_graph()
            self.TRAING_STEPS = TRAING_STEPS
            self.BATCH_SIZE = BATCH_SIZE
            self.STATE_SIZE = STATE_SIZE
            self.EMB_DIM = EMB_DIM
            self.FEATURE_SIZE = FEATURE_SIZE
            self.FEATURE_SIZE2 = FEATURE_SIZE2
            self.MAX_DEN = MAX_DEN
            self.MAX_SEQ_LEN = int(MAX_SEQ_LEN / self.DISCOUNT + 10)
            #print(type(MAX_SEQ_LEN))
            #print(type(self.MAX_SEQ_LEN))
            self.LR = LR
            self.GRAD_CLIP = GRAD_CLIP
            self.L2_NORM = L2_NORM
            self.ALPHA = ALPHA
            self.BETA = BETA
            self.INPUT_FILE = INPUT_FILE
            self.SAVE_LOG = SAVE_LOG
            self.TRAIN_FILE = "./data/" + INPUT_FILE + "/train_all_dtsm_x_forecast_Deli_random_debug0.txt"
            self.TEST_FILE = "./data/" + INPUT_FILE + "/test_all_dtsm_x_forecast_Deli_random_debug0.txt"
            self.OPEN_TEST = OPEN_TEST
            self.COV_SIZE = COV_SIZE
            self.SHOW_SURVIVAL_CURVE = False
            self.W_attention = W_attention


            para = None
            if LOG_FILE_NAME != None:
                para = LOG_FILE_NAME
            else:
                para = LOG_PREFIX + str(self.EMB_DIM) + "_" + \
                    str(BATCH_SIZE) + "_" + \
                    str(self.STATE_SIZE) + "_" + \
                    "{:.6f}".format(self.LR) + "_" + \
                    "{:.6f}".format(self.L2_NORM) + "_" + \
                    INPUT_FILE + "_" + \
                    "{:.2f}".format(self.ALPHA) + "_" \
                    "{:.2f}".format(self.BETA) + "_" + str(ADD_TIME_FEATURE) + \
                        "_" + str(self.QRNN_MODEL) + "_" + str(self.COV_SIZE) + "_" + str(DISCOUNT)

            #print(para, '\n')
            self.filename = para
            self.train_log_txt_filename = "./" + para + '.train.log.txt'
            if os.path.exists(self.train_log_txt_filename):
                self.exist = True
            else:
                if self.SAVE_LOG:
                    self.exist = False
                    self.train_log_txt = open(self.train_log_txt_filename, 'w')
                    self.train_log_txt.close()

        def get_survival_data(self, model, sess):
            alltestdata = SparseData(self.TEST_FILE, True, True)
            ret = []
            while alltestdata.finish_epoch == False:
                test_batch_x, test_batch_x2, test_batch_y, test_batch_len = alltestdata.next(self.BATCH_SIZE)
                bid_loss, bid_test_prob, preds = sess.run(
                    [self.cost, self.predict, self.preds],
                    feed_dict={self.tf_x: test_batch_x,
                                self.tf_x2: test_batch_x2,
                                self.tf_y: test_batch_y,
                                self.tf_bid_len: test_batch_len,
                                })
                ret.append(preds)
            return ret

        def load_data(self):
            self.train_data = biSparseData(self.TRAIN_FILE, self.DISCOUNT)
            self.test_data_win = SparseData(self.TEST_FILE, True, False, self.DISCOUNT)
            #self.test_data_lose = SparseData(self.TEST_FILE, False, False, self.DISCOUNT)

        def is_exist(self):
            if self.SAVE_LOG == False:
                return False
            return self.exist

        def create_graph(self):
            BATCH_SIZE = self.BATCH_SIZE

            tf.disable_eager_execution()

            self.tf_x = tf.placeholder(tf.float32, [None, self.FEATURE_SIZE], name="tf_x")
            self.tf_x_deli = tf.placeholder(tf.float32, [None], name="tf_x_deli")
            self.tf_x2 = tf.placeholder(tf.float32, [None, self.FEATURE_SIZE2], name="tf_x2")
            self.tf_seq = tf.placeholder(tf.string, [None], name="tf_string")
            self.tf_y = tf.placeholder(tf.float32, [None,2], name="tf_y")
            #self.tf_BATCH_ALL_SIZE = tf.placeholder(tf.int32, name="tf_BATCH_ALL_SIZE")

            #self.tf_BATCH_ALL_SIZE2 = tf.to_int32(self.tf_BATCH_ALL_SIZE)
            self.tf_y2 = tf.placeholder(tf.float32, [None, 2], name="tf_y2")
            self.tf_bid_len = tf.placeholder(tf.int32, [None], name="tf_len")

            #self.batch_id = tf.placeholder(tf.int32, name="tf_batch_id")
            #self.tf_market_price = tf.placeholder(tf.int32, [BATCH_SIZE], name="tf_market_price")
            self.tf_control_parameter = tf.placeholder(tf.float32, [2], name="tf_control_parameter")
            alpha = self.tf_control_parameter[0]
            beta = self.tf_control_parameter[1]
            #self.tf_rnn_len = self.tf_bid_len + 2
            self.tf_rnn_len = self.tf_bid_len
            #embeddings = tf.Variable(self.init_matrix([self.MAX_DEN, self.EMB_DIM]))
            #print(embeddings)
            #x_all = tf.concat([tf.cast(self.tf_x,dtype=tf.float32),self.tf_x2],1)
            #x_emds = tf.nn.embedding_lookup(embeddings, x_all)
            #with tf.Session() as sess:
            #    sess.run(tf.global_variables_initializer())
            #    print(sess.run(x_emds))
            #input = tf.reshape(x_emds, [BATCH_SIZE, self.FEATURE_SIZE * self.EMB_DIM])

            #wb_mean, wb_var = tf.nn.moments(self.tf_x2, [0,1])
            #scale = tf.Variable(tf.ones([6]))
            #offset = tf.Variable(tf.zeros([6]))
            #variance_epsilon = 0.001
            #input_x = tf.nn.batch_normalization(self.tf_x2[:,0:6], wb_mean, wb_var, offset, scale, variance_epsilon)
            #input_x = tf.concat([input_x, tf.reshape(tf.cast(self.tf_x2[:,6], dtype=tf.float32),[BATCH_SIZE,1])],1)
            input_x = self.tf_x2
            #input_x = None
            #input_x2 = None
            if self.add_time_feature:
                #kind of dropout?
                #middle_layer = tf.layers.dense(input, self.MIDDLE_FEATURE_SIZE, tf.nn.relu)  # hidden layer
                #self.middle_layer = middle_layer
                def add_time(x):

                    y = tf.reshape(tf.tile(x[0:6], [self.MAX_SEQ_LEN]), [self.MAX_SEQ_LEN, self.FEATURE_SIZE2])
                    t = tf.range(self.MAX_SEQ_LEN)
                    t = tf.one_hot(t,depth = self.MAX_SEQ_LEN)
                    z = tf.concat([y, tf.cast(t, dtype=tf.float32)], 1) 
                    #t2 = tf.tile([x[6]],[self.MAX_SEQ_LEN])
                    #t2 = tf.one_hot(tf.cast(t2, dtype=tf.int32),depth = 40)
                    #z = tf.concat([z, tf.cast(t2, dtype=tf.float32)], 1)

                    return z
                self.t2 = input_x[0][5]
                self.t2 = tf.add(self.t2,0,name="t2")
                input_x = tf.map_fn(add_time, input_x)
                self.vintage = self.tf_x[0][3]


                def add_variable(x):
                    #print(x)
                    #t = tf.one_hot(x,3)

                    #t = tf.reshape(t,[9])



                    t1 = tf.tile([x[0]],[self.MAX_SEQ_LEN])
                    t1 = tf.one_hot(tf.cast(t1, dtype=tf.int32),depth = 3)
                    t2 = tf.tile([x[1]],[self.MAX_SEQ_LEN])
                    t2 = tf.one_hot(tf.cast(t2, dtype=tf.int32),depth = 3)
                    t3 = tf.tile([x[2]],[self.MAX_SEQ_LEN])
                    t3 = tf.one_hot(tf.cast(t3, dtype=tf.int32),depth = 3)
                    t4 = tf.tile([x[3]],[self.MAX_SEQ_LEN])
                    t4 = tf.one_hot(tf.cast(t4, dtype=tf.int32),depth = 40)

                    t1 = tf.concat([t1, t2], 1)
                    t1 = tf.concat([t1, t3], 1)
                    t1 = tf.concat([t1, t4], 1)

                    #m = tf.reshape(tf.tile(t, [self.MAX_SEQ_LEN]), [self.MAX_SEQ_LEN, 9])
                    #y = tf.reshape(tf.tile(x, [self.MAX_SEQ_LEN]), [self.MAX_SEQ_LEN, self.FEATURE_SIZE])



                    return t1



                input_x2 = tf.map_fn(add_variable, self.tf_x)

                self.input_x2 = input_x2[0]

                input_x = tf.concat([input_x, tf.cast(input_x2, dtype=tf.float32)], 2)

                #add delinquency variable

                id = 0
                for i in range(self.BATCH_SIZE):
                    if(i==0):
                        #append the first x washout value accotding to the washout month
                        washout_one_hot_deli = tf.one_hot(tf.tile([tf.cast(0, dtype=tf.int32)],[40]),depth = 4)
                        one_hot_deli = tf.concat([washout_one_hot_deli,tf.one_hot(tf.cast(self.tf_x_deli[0:self.tf_bid_len[0]-40], dtype=tf.int32),depth = 4)],0)
                        left_deli = tf.one_hot(tf.tile([tf.cast(0,dtype=tf.int32)], [self.MAX_SEQ_LEN-self.tf_bid_len[0] + 40-40]),depth = 4)
                        one_hot_deli = tf.concat([one_hot_deli,tf.cast(left_deli,dtype=tf.float32)],0)
                        id = self.tf_bid_len[0] - 40
                    else:
                        #append the first x washout value accotding to the washout month
                        washout_one_hot_deli = tf.one_hot(tf.tile([tf.cast(0, dtype=tf.int32)],[40]),depth = 4)
                        one_hot_deli = tf.concat([one_hot_deli,washout_one_hot_deli],0)
                        one_hot_deli = tf.concat([one_hot_deli,tf.one_hot(tf.cast(self.tf_x_deli[id:id+self.tf_bid_len[i]-40], dtype=tf.int32),depth = 4)],0)
                        left_deli = tf.one_hot(tf.tile([tf.cast(0,dtype=tf.int32)], [self.MAX_SEQ_LEN-self.tf_bid_len[i]+40-40]),depth = 4)
                        one_hot_deli = tf.concat([one_hot_deli,tf.cast(left_deli,dtype=tf.float32)],0)
                        id = id + self.tf_bid_len[i] - 40


                one_hot_deli = tf.reshape(one_hot_deli, [self.BATCH_SIZE,self.MAX_SEQ_LEN, 4])
                input_x = tf.concat([input_x, tf.cast(one_hot_deli, dtype=tf.float32)], 2)

                self.input_x = input_x




                #count = 1

                #y = tf.reshape(tf.tile([1.,0.], [self.tf_bid_len[0]-1]), [self.tf_bid_len[0]-1, 2])
                #y = tf.RaggedTensor.from_tensor(y)
                #y = tf.cast(y,dtype=tf.float32)
                #z = tf.concat([y, [self.tf_y[0]]], 0)

                #for i in range(BATCH_SIZE-1):

                    #a = tf.reshape(tf.tile([1.,0.], [self.tf_bid_len[count]-1]), [self.tf_bid_len[count]-1, 2])
                    #a = tf.RaggedTensor.from_tensor(a)
                    #a = tf.cast(a,dtype=tf.float32)
                    #b = tf.concat([a,[self.tf_y[count]]],0)
                    #z = tf.concat([z, b], 0)
                    #z = tf.cast(z,dtype=tf.float32)
                    #count = count + 1

                    #self.tf_y = z

            preds = None

            if self.DNN_MODEL:
                outlist = []
                for i in range(0, self.BATCH_SIZE):
                    sigleout = tf.layers.dense(input_x[i], 1, tf.nn.sigmoid)
                    outlist.append(sigleout)
                preds = tf.reshape(tf.stack(outlist, axis=0), [self.BATCH_SIZE, self.MAX_SEQ_LEN], name="preds")
            else:
                # input_x = tf.reshape(tf.tile(input, [1, self.MAX_SEQ_LEN]), [BATCH_SIZE, self.MAX_SEQ_LEN, self.FEATURE_SIZE * self.EMB_DIM])
                rnn_cell = None
                rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.STATE_SIZE)
                #add dropout + MultiRNN

                #keep_prob = [0.5,0.5]
                #cells = [tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=n) for n in keep_prob]
                #rnn_dropout = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=0.5)

                #multi_cells = [rnn_dropout for _ in range(2)]

                #multi_cells = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=n),output_keep_prob=self.KEEP_PRO) for n in [32,32]]
                #stacked_lstm = tf.nn.rnn_cell.MultiRNNCell(multi_cells)


                outputs, (h_c, h_n) = tf.nn.dynamic_rnn(
                    rnn_cell,                   # cell you have chosen
                    input_x,                    # input
                    initial_state=None,         # the initial hidden state
                    dtype=tf.float32,           # must given if set initial_state = None
                    time_major=False,           # False: (batch, time step, input); True: (time step, batch, input)
                    sequence_length=self.tf_rnn_len
                )

                self.row_output = outputs

                #switch or switch off the attention mechanism
                e = tf.matmul(outputs,tf.transpose(outputs,[0,2,1])) #(batch * time step * hidden size) * (batch * hidden size * time step) ---> batch * time step * time step
                self.attention = tf.nn.softmax(e,dim=-1) #batch * time step * time step
                self.attention_score_seq = self.attention

                #test: whether attention actually works,make the same score for each time steps
                #e = tf.random.uniform([self.BATCH_SIZE,self.MAX_SEQ_LEN,self.MAX_SEQ_LEN],0.,5.)
                #self.attention = tf.nn.softmax(e,dim=-1)
                outputs = tf.matmul(self.attention,outputs) # (batch * time step * time step) * (batch * time step * hidden size) ---> batch * time step * hidden size
                self.attention_weighted_output = outputs
                #outputs = tf.nn.dropout(outputs,0.5)

                #1
                new_output = tf.reshape(outputs, [self.MAX_SEQ_LEN * self.BATCH_SIZE, self.STATE_SIZE])
                #print(self.BATCH_SIZE)

                with tf.variable_scope('softmax'):
                    W = tf.get_variable('W', [self.STATE_SIZE, 1])
                    b = tf.get_variable('b', [1], initializer=tf.constant_initializer(0))

                #
                #print(W)
                self.weight = W
                self.weight = tf.add(self.weight,0,name="weight")
                logits = tf.matmul(new_output, W) + b





                #1
                preds = tf.transpose(tf.nn.sigmoid(logits, name="preds"), name="preds")[0]




            #
            self.preds = preds
            survival_rate = preds

            #
            batch_rnn_survival_rate = tf.reshape(survival_rate, [self.BATCH_SIZE, self.MAX_SEQ_LEN])

            #

            self.map_parameter = tf.concat([batch_rnn_survival_rate,
                                        tf.cast(tf.reshape(self.tf_bid_len, [self.BATCH_SIZE, 1]), tf.float32)],
                                        1,name="map_parameter")



            #
            #map_parameter = tf.concat([map_parameter,
            #                           tf.cast(tf.reshape(self.tf_market_price, [BATCH_SIZE, 1]), tf.float32)],
            #                          1)

            #def reduce_mul(x):
                #bid_len = tf.cast(x[self.MAX_SEQ_LEN], dtype=tf.int32)

                #survival_rate_last_one = x[0:bid_len]

                #ret = survival_rate_last_one
                #return ret

            self.cross_entropy2 = 0
            self.cross_entropy3 = 0
            id = 0


            #
            default = []
            good = []
            bad = []
            count_predict = 0
            for i in range(self.BATCH_SIZE):  
                if(i == 0):
                    id = 0
                if(id == 0):
                    survival_rate = self.map_parameter[i][40:self.tf_bid_len[i]]
                    dead_rate = tf.subtract(tf.constant(1.0, dtype=tf.float32), survival_rate)
                    predict = tf.transpose(tf.stack([survival_rate, dead_rate])) 
                    self.cross_entropy2 = self.cross_entropy2 -tf.reduce_sum(self.tf_y2[0:self.tf_bid_len[i]-40]*tf.log(tf.clip_by_value(predict,1e-10,1.0)))
                    predicted_label = predict
                    true_label = self.tf_y2[id:id+self.tf_bid_len[i]-40]
                    id = self.tf_bid_len[i]-40

                    count_predict = count_predict + 1
                    #self.id = id
                    #true_label.append(self.tf_y2[id:id+self.tf_bid_len[i]])


                    #predicted_label = tf.concat([input_x, tf.cast(input_x2, dtype=tf.float32)], 0)
                else:
                    survival_rate = self.map_parameter[i][40:self.tf_bid_len[i]]
                    dead_rate = tf.subtract(tf.constant(1.0, dtype=tf.float32), survival_rate)
                    predict = tf.transpose(tf.stack([survival_rate, dead_rate])) 

                    self.cross_entropy2 = self.cross_entropy2 -tf.reduce_sum(self.tf_y2[id:id+self.tf_bid_len[i]-40]*tf.log(tf.clip_by_value(predict,1e-10,1.0)))
                    predicted_label = tf.concat([predicted_label, tf.cast(predict, dtype=tf.float32)], 0)
                    true_label = tf.concat([true_label, tf.cast(self.tf_y2[id:id+self.tf_bid_len[i]-40], dtype=tf.float32)], 0)
                    count_predict = count_predict + 1
                    id = id + self.tf_bid_len[i]-40
                    #self.id = tf.concat([self.id, tf.cast(self.tf_bid_len[i], dtype=tf.int32)], 1)
                self.count_predict = count_predict

            #


            #print("create graph")

            self.true_label = true_label

            self.predicted_label = predicted_label

            self.true_label = tf.add(self.true_label,0,name="true_label")
            self.predicted_label = tf.add(self.predicted_label,0,name="predicted_label")

            self.survival_rate = survival_rate

            self.mp_para = self.map_parameter
            #rate_result = tf.map_fn(reduce_mul, elems=map_parameter ,name=\"rate_result\")
            #print(rate_result)
            #rate_result = tf.reshape(rate_result,[1,-1])
            #print(rate_result)
            #self.rate_result = rate_result
            #log_minus = tf.log(tf.add(tf.transpose(rate_result)[2] - tf.transpose(rate_result)[1], 1e-20))#todo debug

            #self.anlp_node = -tf.reduce_sum(log_minus) / self.BATCH_SIZE #todo load name
            #self.anlp_node = tf.add(self.anlp_node, 0, name=\"anlp_node\")
            #self.final_survival_rate = tf.transpose(rate_result)
            #self.final_survival_rate = rate_result
            #self.final_dead_rate = tf.subtract(tf.constant(1.0, dtype=tf.float32), self.final_survival_rate)

            #self.predict = tf.transpose(tf.stack([self.final_survival_rate, self.final_dead_rate]), name=\"predict\")
            #cross_entropy = -tf.reduce_sum(self.tf_y2*tf.log(tf.clip_by_value(self.predict,1e-10,1.0)))


            #
            all_count = 0
            for i in range(self.BATCH_SIZE):
                all_count = all_count + self.tf_bid_len[i] - 40
            self.cross_entropy2 = self.cross_entropy2/tf.cast(all_count,dtype=tf.float32)
            self.cross_entropy2 = tf.add(self.cross_entropy2,0,name="cross_entropy2")

            #
            tvars = tf.trainable_variables()
            #print(tvars)
            #self.cross_entropy2 = tf.add_n([ tf.nn.l2_loss(v) for v in tvars ]) * self.L2_NORM / self.BATCH_SIZE

            #self.cross_entropy2 = -tf.reduce_sum(self.tf_y2*tf.log(tf.clip_by_value(self.predict,1e-10,1.0)))/ tf.cast(all_count,dtype=tf.float32)

            #tvars = tf.trainable_variables()

            lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in tvars ]) * self.L2_NORM / self.BATCH_SIZE 

            #cost = tf.add(cross_entropy, lossL2, name = "cost")  / self.BATCH_SIZE      
            #cost = tf.add(self.cross_entropy2, lossL2, name = "cost")
            #self.cost = tf.add(cost, 0, name="cost")

            #self.cost = tf.add(self.cross_entropy2, lossL2, name="cost")

            self.cost = tf.add(self.cross_entropy2, 0, name="cost")
            optimizer = tf.train.AdamOptimizer(learning_rate=self.LR, beta2=0.99)#.minimize(cost)
            #optimizer_anlp = tf.train.AdamOptimizer(learning_rate=self.ANLP_LR, beta2=0.99)#.minimize(cost)

            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                                                self.GRAD_CLIP,
                                                )
            self.train_op = optimizer.apply_gradients(zip(grads, tvars), name="train_op")
            tf.add_to_collection('train_op', self.train_op)

            #anlp_grads, _ = tf.clip_by_global_norm(tf.gradients(self.anlp_node, tvars),
            #                                  self.GRAD_CLIP,
            #                                  )
            #self.anlp_train_op = optimizer_anlp.apply_gradients(zip(anlp_grads, tvars), name="anlp_train_op")
            #tf.add_to_collection('anlp_train_op', self.anlp_train_op)

            #self.com_cost = tf.add(alpha * self.cost, beta * self.anlp_node)  we don't have this anlp
            #self.com_cost = self.cross_entropy2
            self.com_cost = self.cost
            com_grads, _ = tf.clip_by_global_norm(tf.gradients(self.com_cost, tvars),
                                                self.GRAD_CLIP,
                                                )

            self.com_train_op = optimizer.apply_gradients(zip(com_grads, tvars), name="train_op")
            tf.add_to_collection('com_train_op', self.com_train_op)


            #correct_pred = tf.equal(tf.argmax(self.predict, 1), tf.argmax(self.tf_y2, 1))
            #self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")
            #

        def train_test(self,sess):
            self.load_data()
            init = tf.global_variables_initializer()
            self.sess = sess
            sess.run(init)
            saver = tf.train.Saver(max_to_keep=100)
            self.saver = saver
            TRAIN_LOG_STEP = int((self.train_data.size * 0.1) / self.BATCH_SIZE)
            train_auc_arr = []
            train_anlp_arr = []
            train_loss_arr = []
            train_auc_label = []
            train_auc_prob = []
            train_survival = []
            train_cross_entropy = []
            train_good_label = []
            train_bad_label = []
            total_train_duration = 0
            total_test_duration = 0
            TEST_COUNT = 0
            max_auc = -1
            min_anlp = 200
            enough_test = 0
            last_loss = [9999.0, 9999.0]

            start_time = time.time()
            for step in range(1, self.TRAING_STEPS + 1):
                self.global_step = step
                batch_x, batch_x2, batch_y, batch_len, batch_y2,batch_x_deli, batch_seq, win = self.train_data.next(self.BATCH_SIZE)

                #
                #print('training steps:',step)
                #batch_y = tf.ragged.RaggedTensorValue(batch_y,np.array([self.BATCH_SIZE]))

                if self.ONLY_TRAIN_ANLP:
                    if win: #if win
                        _, train_anlp, train_loss, train_cross_entropy2 = sess.run([self.com_train_op, self.anlp_node, self.cost, self.cross_entropy2],
                                                                        feed_dict={self.tf_x: batch_x,
                                                                                    self.tf_x2: batch_x2,  
                                                                                    self.tf_y: batch_y,
                                                                                    self.tf_y2: batch_y2,

                                                                                    self.tf_bid_len: batch_len,

                                                                                    #self.tf_market_price: test_batch_market_price
                                                                                    self.tf_control_parameter:[self.ALPHA, self.BETA]
                                                            })
                        #train_anlp_arr.append(train_anlp)
                        train_loss_arr.append(train_loss)

                        train_auc_label.append(batch_y.T[0])
                        #train_auc_prob.append(np.array(train_outputs).T[0])
                        train_cross_entropy.append(train_cross_entropy2)

                    else:
                        train_loss, train_cross_entropy2 = sess.run([self.cost,  self.cross_entropy2], feed_dict={self.tf_x: batch_x,
                                                            self.tf_y: batch_y,
                                                            self.tf_x2: batch_x2,
                                                            self.tf_y2: batch_y2,                                                                                

                                                            self.tf_bid_len: batch_len,

                                                            #self.tf_market_price: test_batch_market_price
                                                            self.tf_control_parameter:[self.ALPHA, self.BETA]
                                                            })
                        #print train_outputs
                        train_loss_arr.append(train_loss)
                        train_auc_label.append(batch_y.T[0])
                        #train_auc_prob.append(np.array(train_outputs).T[0])
                        train_cross_entropy.append(train_cross_entropy2)
                else:
                    if win: #if win

                        #
                        _,  train_loss, train_cross_entropy2, survival_rate,  train_predicted_label,weight,train_row_output,train_vintage, train_input_x2,train_tf_x= sess.run([self.com_train_op, self.cost, self.cross_entropy2, self.survival_rate,self.predicted_label,self.weight,self.row_output,self.vintage,self.input_x2,self.tf_x],
                                                                        feed_dict={self.tf_x: batch_x,
                                                                                    self.tf_x2: batch_x2,
                                                                                    self.tf_y: batch_y,
                                                                                    self.tf_y2: batch_y2,
                                                                                    self.tf_x_deli: batch_x_deli,
                                                                                    self.tf_seq: batch_seq,
                                                                                    self.tf_bid_len: batch_len,

                                                                                    #self.tf_market_price: test_batch_market_price
                                                                                    self.tf_control_parameter:[self.ALPHA, self.BETA]
                                                            })
                        #train_anlp_arr.append(train_anlp)

                        train_loss_arr.append(train_loss)
                        train_auc_label.append(batch_y.T[0])
                        #train_auc_prob.append(np.array(train_outputs).T[0])
                        train_cross_entropy.append(train_cross_entropy2)
                        self.train_survival_rate = survival_rate
                        # draw the survival curve
                        #x_axis = []
                        #print('train_survival_rate',survival_rate)
                        #for i in range(0,len(survival_rate)):

                            #x_axis.append((i+1))

                        #plt.plot(x_axis, survival_rate,'b')
                        #plt.ylabel('Survival Rate')
                        #plt.xlabel('Time')
                        #plt.show()

                        #print(train_vintage)    

                        #print(train_t2)
                        #print('##############################')
                        #print(train_tf_x[0])
                        #print(train_input_x2[0])

                        #for i in range(32):
                            #print('train_true_label',len(train_true_label[i]))
                            #print('train_predicted_label',len(train_predicted_label[i]))

                        #print(len(train_outputs))
                        #print(train_true,train_default)
                        #print(weight)  
                        #print(train_row_output)
                        #print('=========')
                        #print(train_logits)
                        #print('next steps')
                    else:
                        _, train_loss, train_cross_entropy2 = sess.run([self.train_op, self.cost, self.cross_entropy2], feed_dict={self.tf_x: batch_x,
                                                            self.tf_y: batch_y,
                                                            self.tf_x2: batch_x2,
                                                            self.tf_y2: batch_y2,

                                                            self.tf_bid_len: batch_len,

                                                            #self.tf_market_price: test_batch_market_price
                                                            self.tf_control_parameter:[self.ALPHA, self.BETA]
                                                            })
                        #print train_outputs
                        train_loss_arr.append(train_loss)
                        train_auc_label.append(batch_y.T[0])
                        #train_auc_prob.append(np.array(train_outputs).T[0])
                        train_cross_entropy.append(train_cross_entropy2)

                if step % 100 == 0:
                    #mean_anlp = np.array(train_anlp_arr[-99:]).mean()
                    mean_loss = np.array(train_loss_arr[-99:]).mean()
                    mean_cross_entropy = np.array(train_cross_entropy[-99:]).mean()
                    mean_auc = 0.0001
                    #comment auc
                    #if not self.ONLY_TRAIN_ANLP:
                        #try:
                            #mean_auc = roc_auc_score(np.reshape(train_auc_label, [1, -1])[0], np.reshape(train_auc_prob, [1, -1])[0])
                            #print('AUC score:',mean_auc)
                        #except Exception:
                            #print (\"AUC ERROE\")
                            #continue
                    #delete the mean_auc for a moment
                    log = self.getStatStr("TRAIN", self.global_step, mean_loss,mean_cross_entropy)
                    #print(log)
                    self.force_write(log)
                    train_loss_arr = []
                    train_anlp_arr = []
                    train_auc_label = []
                    train_auc_prob = []
                    train_cross_entropy = []
                    if self.TEST_FREQUENT:
                        self.run_test(sess)
                        #self.save_model()

                if self.global_step < 300:
                    self.SHOW_SURVIVAL_CURVE = True
                    if self.global_step % 100 == 0:
                        self.run_test(sess)

                        #self.save_model()
                elif self.global_step < 1000:
                    self.SHOW_SURVIVAL_CURVE = True
                    if self.global_step % 300 == 0:
                        self.run_test(sess)
                        self.save_model()
                elif self.global_step < 4000:
                    #self.SHOW_SURVIVAL_CURVE = True
                    if self.global_step % 200 == 0:
                        self.run_test(sess)
                        self.save_model()
                elif self.global_step < 6000:
                    #self.SHOW_SURVIVAL_CURVE = True
                    if self.global_step % 500 == 0:
                        self.run_test(sess)
                        self.save_model()        
                elif self.global_step < 8000:
                    #self.SHOW_SURVIVAL_CURVE = True
                    if self.global_step % 200 == 0:
                        self.run_test(sess)
                        self.save_model()
                elif self.global_step < 15000:
                    #self.SHOW_SURVIVAL_CURVE = True
                    if self.global_step % 1000 == 0:
                        self.run_test(sess)
                        self.save_model()
                elif self.global_step <= 21000:
                    self.SHOW_SURVIVAL_CURVE = True
                    if self.global_step % 1000 == 0:
                        self.run_test(sess)
                        self.save_model()
                #elif self.global_step <= 40000:
                    #if self.global_step % 5000 == 0:
                        #self.run_test(sess)
                        #self.save_model()
                #elif self.global_step <= 100000:
                    #if self.global_step % 10000 == 0:
                        #self.run_test(sess)
                        #self.save_model()
                else:
                    break

        def run_model(self):
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:
                self.train_test(sess)

        def save_model(self):
            print("model name: ", self.filename, " ", self.global_step, "\n")
            self.saver.save(self.sess, "../saved_model/model_forecast_washout20_noL2_512_8_exclude_exception/" + self.filename, global_step=self.global_step)

        def getStatStr(self, category ,step, mean_loss,mean_cross_entropy):
            statistics_log = str(self.INPUT_FILE) + "\t" + category + "\t" + str(step) + "\t" \
                                "{:.6f}".format(mean_loss) + "\t" + \
                                "{:.6f}".format(mean_cross_entropy) + "\t" + \
                                str(self.EMB_DIM) + "\t" + str(self.BATCH_SIZE) + "\t" + \
                                str(self.STATE_SIZE) + "\t" + \
                                "{:.6f}".format(self.LR) + "\t" + \
                                "{:.6}".format(self.L2_NORM) + "\t" +\
                                str(self.ALPHA) + '\t' + \
                                str(self.BETA) + "\n"
            return statistics_log

        def load(self, meta, ckpt, step):
            tf.reset_default_graph()
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True

            #

            saver = tf.train.import_meta_graph(meta)


            #
            #self.load_data()
            self.global_step = step
            #with tf.Session(config=config) as sess:
            sess = tf.Session(config=config)
            saver.restore(sess, ckpt)
            graph = tf.get_default_graph()
            self.tf_x = graph.get_tensor_by_name("tf_x:0")
            self.tf_x_deli = graph.get_tensor_by_name("tf_x_deli:0")
            self.tf_x2 = graph.get_tensor_by_name("tf_x2:0")
            self.tf_y = graph.get_tensor_by_name("tf_y:0")
            self.tf_y2 = graph.get_tensor_by_name("tf_y2:0")
            self.tf_seq = graph.get_tensor_by_name("tf_string:0")
            self.tf_bid_len = graph.get_tensor_by_name("tf_len:0")
            #self.W_attention = graph.get_tensor_by_name("trainable_attention_mul_weight:0")
            #self.tf_market_price = graph.get_tensor_by_name("tf_market_price:0")
            #self.accuracy = graph.get_tensor_by_name("accuracy:0")
            self.cost = graph.get_tensor_by_name("cost:0")
            self.cross_entropy2 = graph.get_tensor_by_name("cross_entropy2:0")
            self.true_label = graph.get_tensor_by_name("true_label:0")
            self.predicted_label = graph.get_tensor_by_name("predicted_label:0")
            #self.tf_training = graph.get_tensor_by_name("tf_training:0")
            #self.predict = graph.get_tensor_by_name("predict:0")
            #self.anlp_node = graph.get_tensor_by_name("anlp_node:0")
            #self.train_op = tf.get_collection('train_op')[0]
            self.t2 = graph.get_tensor_by_name("t2:0")
            self.map_parameter = graph.get_tensor_by_name("map_parameter:0")

            #self.anlp_train_op = graph.get_collection("anlp_train_op")[0]
            #self.train _op = graph.get_tensor_by_name("train_op:0")
            self.preds = graph.get_tensor_by_name("preds:0")
            #self.com_train_op = tf.get_collection("com_train_op")[0]
            #self.tf_control_parameter = graph.get_tensor_by_name("tf_control_parameter:0")
            # self.train_log_txt.write(statistics_log)
            return sess

        def run_test(self, sess):
            auc_arr = []
            loss_arr = []
            anlp_arr = []
            auc_prob = []
            auc_label = []
            true_label = []
            predicted_label = []
            cross_entropy = []
            self.TRUE_LABEL2 = []
            self.PREDICTED_LABEL2 = []
            log_good = 0
            log_bad = 0
            count_good = 0
            count_bad = 0

            #print self.test_data_win.size + self.test_data_lose.size, \"total size\"
            total_time = 0
            count = 0    
            #

            for i in range(0, int(self.test_data_win.size / self.BATCH_SIZE)):


                #
                test_batch_x, test_batch_x2, test_batch_y, test_batch_len, test_batch_y2, test_batch_x_deli, test_batch_seq = self.test_data_win.next(
                    self.BATCH_SIZE)
                #test_batch_y = tf.ragged.RaggedTensorValue(test_batch_y,np.array([self.BATCH_SIZE]))
                start_time = time.time()
                bid_loss, bid_test_cross_entropy2, bid_true_label, bid_predicted_label,bid_weight,survival_rate,t2,test_vintage= sess.run(
                    [self.cost,  self.cross_entropy2, self.true_label,self.predicted_label,self.weight,self.survival_rate,self.t2,self.vintage],
                    feed_dict={self.tf_x: test_batch_x,
                                self.tf_x2: test_batch_x2,
                                self.tf_y: test_batch_y,
                                self.tf_y2: test_batch_y2,
                                self.tf_seq: test_batch_seq, 
                                self.tf_x_deli: test_batch_x_deli,
                                self.tf_bid_len: test_batch_len

                                #self.tf_market_price: test_batch_market_price
                                })


                #
                if(count == 0):
                    #fig = plt.figure()
                    #plt.grid(False)
                    # draw the survival curve
                    #print('vintage:',test_vintage)
                    #x_axis = []
                    #print('test_survival_rate',survival_rate)
                    #for i in range(0,len(survival_rate)):

                        #x_axis.append((i+1))

                    #plt.plot(x_axis, survival_rate,'b')
                    #plt.ylabel('Survival Rate')
                    #plt.xlabel('Time')
                    #plt.show()
                    count = count + 1


                #for i in range(len(bid_true_label)):
                    #for j in range(len(bid_true_label[i])):

                        #self.TRUE_LABEL2.append(bid_true_label[i][j])
                        #self.PREDICTED_LABEL2.append(bid_predicted_label[i][j])


                self.WEIGHT.append(bid_weight)

                #for i in range(len(bid_true_label)):
                        #self.TRUE_LABEL2.append(bid_true_label[i])
                        #self.PREDICTED_LABEL2.append(bid_predicted_label[i])


                #print('test+++++++++++++++++++++++++++++++')
                #print(self.TRUE_LABEL2,self.PREDICTED_LABEL2)
                total_time += time.time() - start_time
                #auc_prob.append(np.array(bid_test_prob).T[0])
                auc_label.append(test_batch_y.T[0])
                cross_entropy.append(bid_test_cross_entropy2)

                #print(prediction)
                #anlp_arr.append(anlp)
                loss_arr.append(bid_loss)
            #if len(auc_prob) > 0:
                #try:
                    #auc = roc_auc_score(np.reshape(np.array(auc_label), [1, -1])[0],
                                    #np.reshape(np.array(auc_prob), [1, -1])[0])
                #except Exception:
                    #print(\"AUC ERROR\")
                    #return

                #auc_arr.append(auc)

            mean_loss = np.array(loss_arr).mean()
            mean_cross_entropy = np.array(cross_entropy).mean()



            #mean_anlp = np.array(anlp_arr).mean()
            #mean_auc = np.array(auc_arr).mean()
            #delete mean_auc for a moment
            log = self.getStatStr("TEST_DATA", self.global_step, mean_loss, mean_cross_entropy)
            #print(log)

            #comment the lose part for a moment\
            #for i in range(0, int(self.test_data_lose.size / self.BATCH_SIZE)):
                #test_batch_x, test_batch_y, test_batch_len = self.test_data_lose.next(
                    #self.BATCH_SIZE)
                #bid_loss, bid_test_prob = sess.run(
                                        #[self.cost, self.predict],
                                        #feed_dict={self.tf_x: test_batch_x,
                                                    #self.tf_y: test_batch_y,
                                                    #self.tf_bid_len: test_batch_len,
                                                    #self.tf_market_price: test_batch_market_price
                                                    #})
                #auc_prob.append(np.array(bid_test_prob).T[0])
                #auc_label.append(test_batch_y.T[0])
                #anlp_arr.append(anlp)
                #loss_arr.append(bid_loss)
            #delete mean_auc for a moment
            #if len(auc_prob) > 0:
                #try:
                    #auc = roc_auc_score(np.reshape(np.array(auc_label), [1, -1])[0],
                                    #np.reshape(np.array(auc_prob), [1, -1])[0])
                #except Exception:
                    #print(\"AUC ERROR\")
                    #return

                #auc_arr.append(auc)
            #mean_loss = np.array(loss_arr).mean()
            #mean_auc = np.array(auc_arr).mean()
            #mean_anlp = np.array(anlp_arr).mean()
            #delete mean_auc for a moment
            #log = self.getStatStr("TEST", self.global_step, mean_loss)
            self.force_write(log)
            #print(log)
            return mean_loss,mean_cross_entropy

        def force_write(self, log):
            if not self.SAVE_LOG:
                return
            self.train_log_txt = open(self.train_log_txt_filename, 'a')
            self.train_log_txt.write(log)
            self.train_log_txt.close()

    #3lstm standardscaler
    if(prs<48):
        path = os.getcwd()
        new_path = path.replace("\\","/")
        f2 = open(new_path + "/Replication_IJF/data/2259/train_all_dtsm_x_forecast_Deli_random_debug0.txt")
    else:
        path = os.getcwd()
        new_path = path.replace("\\","/")
        f2 = open(new_path + "/Replication_IJF/data/2259/train16to21_all_dtsm_x_forecast_Deli_random_debug0.txt")

    #f2 = open("./test4_data.txt")
    #f2 = open("./test3.txt")
    #train_data = f1.readline()

    #test_data = f2.readline()
    test_data = []
    for line in f2:
        test_data.append(line)



    test_credit = []
    test_DTI = []
    test_UPB = []
    test_LTV = []
    test_IR = []
    test_Age = []
    test_SEQ = []
    test_vintage = []
    test_calendar = []
    test_Def = []
    test_FTHF = []
    test_OS = []
    test_Channel = []
    test_PT = []
    test_LP = []
    test_NB = []
    test_Deli = []
    lag_deli = []
    current_deli = []
    final_data = []

    x = []
    y = []
    b2 = []

    last_seq = ''
    new_seq = ''
    last_default = 0
    new_default = 0
    is_default = 0
    max_seqlen = 0

    for i in range(len(test_data)):
        line_data = test_data[i].split(' ')
        #if(int(line_data[0])!=9999 ):
        #if(int(line_data[6])!=0):
        test_credit.append(int(line_data[0]));
        test_FTHF.append(str(line_data[1]))
        test_OS.append(str(line_data[2]))
        test_DTI.append(int(line_data[3]));
        test_UPB.append(int(line_data[4]));
        test_LTV.append(int(line_data[5]));
        test_IR.append(float(line_data[6]));
        test_Channel.append(str(line_data[7]));
        test_PT.append(str(line_data[8]));
        test_LP.append(str(line_data[9]));
        test_NB.append(int(line_data[10]));
        test_SEQ.append(str(line_data[11]))
        test_Age.append(int(line_data[12]));
        test_vintage.append(int(line_data[13]));
        test_calendar.append((int(line_data[14])));
        test_Def.append(int(line_data[15]));
        test_Deli.append(int(line_data[16]));



    f2.close()

    #test_credit = np.array(test_credit)
    #test_DTI = np.array(test_DTI)
    #test_UPB = np.array(test_UPB)
    #test_LTV = np.array(test_LTV)
    #test_IR = np.array(test_IR)
    #test_SEQ = np.array(test_SEQ)
    #test_Def = np.array(test_Def)

    for i in range(len(test_credit)):
        line_data = test_data[i].split(' ')
        new_seq = test_SEQ[i]
        b2.append(test_credit[i])
        #b2.append(test_FTHF[i])
        #b2.append(test_OS[i])
        b2.append(test_DTI[i])
        b2.append(test_UPB[i])
        b2.append(test_LTV[i])
        b2.append(test_IR[i])
        #b2.append(test_Channel[i])
        #b2.append(test_PT[i])
        #b2.append(test_LP[i])
        b2.append(test_NB[i])
        #b2.append(test_Age[i])
        #b2.append(test_vintage[i])
        #b2.append(test_calendar[i])
        #if(int(test_Age[i])<=2):
            #b2.append(0)
        #else:
            #s_p3 = test_data[i-3].split(' ')
            #b2.append(int(s_p3[16]))
        current_deli.append(int(line_data[16]))
        final_data.append(line_data)
        #b2.append(test_Deli[i])
        y.append(test_Def[i])
        x.append(b2)


        if(test_Age[i] > max_seqlen):
            max_seqlen = test_Age[i]

        if(test_Def[i] == 1):
            is_default = 1
        if(len(last_seq) ==  0):
            last_seq = test_SEQ[0] 
        if(str(new_seq) == str(last_seq) and is_default == 0 and int(test_Age[i]) < max_seqlen ):
            del x[len(x)-1]
            del y[len(y)-1]
            del current_deli[len(current_deli)-1]
            #print('last one: ',final_data[len(final_data)-2])
            #print('delete current: ',final_data[len(final_data)-1])
            del final_data[len(final_data)-1]
            #print('Exception1!')
        if(str(new_seq) == str(last_seq) and is_default == 1 and test_Def[i] == 0):
            del x[len(x)-1]
            del y[len(y)-1]
            del current_deli[len(current_deli)-1]
            #print('last one: ',final_data[len(final_data)-2])
            #print('delete current: ',final_data[len(final_data)-1])
            del final_data[len(final_data)-1]
            #print('after delete: ',final_data[len(final_data)-1])
            #print('after delete x: ',x[len(x)-1])
            #print('after delete y:',y[len(y)-1])
            #print('Exception2!')
        if(str(new_seq) != str(last_seq)):
            if(max_seqlen+1 != len(current_deli)-1):

                #print('max_seqlen+1:',max_seqlen+1)
                #print('current_deli:',len(current_deli)-1)
                #print('current_seq:',final_data[len(final_data)-2])
                #print('current_x:',x[len(x)-2])
                #print('current_y:',y[len(y)-2])

                #important!!!, cause to wrong training: need to delete the from the last index, not the current index
                del x[len(x)-len(current_deli):len(x)-1]
                del y[len(y)-len(current_deli):len(y)-1]
                del final_data[len(final_data)-len(current_deli):len(final_data)-1]
                #print('after_delete_seq:',final_data[len(final_data)-2])
                #print('after_delete_x:',x[len(x)-2])
                #print('after_delete_y:',y[len(y)-2])



            is_default = 0
            max_seqlen = 0 
            current_deli = []
            #cleaned but need to return this current data(current data belongs to the new seq)
            s = test_data[i].split(' ')
            slen = len(s)
            current_deli.append(int(s[slen-1])) 

        last_seq = new_seq        
        b2=[]

    if(max_seqlen != 0):
        if(max_seqlen+1 != len(current_deli)-1):

            #print('max_seqlen+1:',max_seqlen+1)
            #print('current_deli:',len(current_deli)-1)
            #print('current_seq:',final_data[len(final_data)-2])
            #print('current_x:',x[len(x)-2])
            #print('current_y:',y[len(y)-2])

            #important!!!, cause to wrong training: need to delete the from the last index, not the current index
            del x[len(x)-len(current_deli):len(x)-1]
            del y[len(y)-len(current_deli):len(y)-1]
            del final_data[len(final_data)-len(current_deli):len(final_data)-1]
            #print('after_delete_seq:',final_data[len(final_data)-2])
            #print('after_delete_x:',x[len(x)-2])
            #print('after_delete_y:',y[len(y)-2])


    #print(x2.shape)
    #print(y2.shape)
    for i in range(len(x)):
        #NB:1=1,2=2,9=3
        if(x[i][5]==1):
            x[i][5]=0 
        if(x[i][5]==2):
            x[i][5]=1 
        if(x[i][5]==99):
            x[i][5]=0.5 
    #print(x[0][5])
    #print(len(x),len(x[0]))
    scaler = preprocessing.StandardScaler().fit(x)

    state_size = 16
    batch_size = 512

    #default parameter
    FEATURE_SIZE = 4 # dataset input fields count
    FEATURE_SIZE2 = 6 # dataset input fields count
    MAX_DEN = 580000 # max input data demension
    EMB_DIM = 31
    BATCH_SIZE = batch_size
    TRUE_BATCH_SIZE = batch_size
    MAX_SEQ_LEN = 300
    TRAING_STEPS = 21000
    STATE_SIZE = state_size
    GRAD_CLIP = 5.0
    L2_NORM = 0.001
    KEEP_PRO = 0.5
    NUM_LAYERS = 2
    ADD_TIME = True
    ALPHA = 1.2 # coefficient for cross entropy
    BETA = 0.2 # coefficient for anlp
    input_file="2259" #toy dataset

    #if len(sys.argv) < 2:
    #    print("Please input learning rate. ex. 0.0001")
    #    sys.exit(0)


    LR = float(0.0001)
    LR_ANLP = LR
    RUNNING_MODEL = BASE_RNN(EMB_DIM=EMB_DIM,
                             FEATURE_SIZE=FEATURE_SIZE,
                             FEATURE_SIZE2=FEATURE_SIZE2,
                             BATCH_SIZE=BATCH_SIZE,
                             #TRUE_BATCH_SIZE=TRUE_BATCH_SIZE,
                             MAX_DEN=MAX_DEN,
                             MAX_SEQ_LEN=MAX_SEQ_LEN,
                             TRAING_STEPS=TRAING_STEPS,
                             STATE_SIZE=STATE_SIZE,
                             LR=LR,
                             GRAD_CLIP=GRAD_CLIP,
                             L2_NORM=L2_NORM,
                             INPUT_FILE=input_file,
                             ALPHA=ALPHA,
                             BETA=BETA,
                             ADD_TIME_FEATURE=ADD_TIME,
                             FIND_PARAMETER=False,
                             ANLP_LR=LR,
                             DNN_MODEL=False,
                             DISCOUNT=1,
                             ONLY_TRAIN_ANLP=False,
                             SURVIVAL_RATE = [],
                             TRUE_LABEL2 = [],
                             PREDICTED_LABEL2 = [],
                             WEIGHT = [],
                             KEEP_PRO = KEEP_PRO,
                             NUM_LAYERS = NUM_LAYERS,
                             LOG_PREFIX="drsa")
    RUNNING_MODEL.create_graph()




    #T1 = time.perf_counter()
    import seaborn as sns
    from scipy import stats
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import auc


    #lstm washout 3lstm attention deli testing
    if(int(ym%100)<10):
        path = os.getcwd()
        new_path = path.replace("\\","/")
        TEST_FILE = new_path + "/Replication_IJF/data/2259/" + "test" + "0" + str(int(ym%100)) + "_" + str(q) + "_unbalanced_dtsm_x_Deli.txt"
    else:
        path = os.getcwd()
        new_path = path.replace("\\","/")
        TEST_FILE = new_path + "/Replication_IJF/data/2259/" + "test" + str(int(ym%100)) + "_" + str(q) + "_unbalanced_dtsm_x_Deli.txt"
    RUNNING_MODEL.test_data_win = SparseData(TEST_FILE, True, False,1, 512)
    if(prs<48):
        path = os.getcwd()
        new_path = path.replace("\\","/")
        meta = new_path + "/Replication_IJF/saved_model/model04to12_gpu_forecast_washout40_deli_random_3Weighted LSTM_attention seq selfdot_noL2_512_16" + "/drsa32_512_16_0.000100_0.100000_2259_1.20_0.20_True_False_1_1.meta"
        ckpt = new_path + "/Replication_IJF/saved_model/model04to12_gpu_forecast_washout40_deli_random_3Weighted LSTM_attention seq selfdot_noL2_512_16" + "/drsa32_512_16_0.000100_0.100000_2259_1.20_0.20_True_False_1_1"
    else:
        path = os.getcwd()
        new_path = path.replace("\\","/")
        meta = new_path + "/Replication_IJF/saved_model/model16to21_gpu_forecast_washout40_deli_random_3Weighted LSTM_attention seq selfdot_noL2_512_16" + "/drsa32_512_16_0.000100_0.100000_2259_1.20_0.20_True_False_1_1.meta"
        ckpt = new_path + "/Replication_IJF/saved_model/model16to21_gpu_forecast_washout40_deli_random_3Weighted LSTM_attention seq selfdot_noL2_512_16" + "/drsa32_512_16_0.000100_0.100000_2259_1.20_0.20_True_False_1_1"
    step = 100000


    sess = RUNNING_MODEL.load(meta,ckpt,step)
    T1 = time.perf_counter()
    #RUNNING_MODEL.run_test(sess)
    auc_arr = []
    loss_arr = []
    anlp_arr = []
    auc_prob = []
    auc_label = []
    cross_entropy = []
    RUNNING_MODEL.TRUE_LABEL2 = []
    RUNNING_MODEL.PREDICTED_LABEL2 = []

    #print self.test_data_win.size + self.test_data_lose.size, \"total size\"
    total_time = 0
    log_good = 0
    log_bad = 0
    true_label = []
    predicted_label = []
    seqlen = []
    count_good = 0
    count_bad = 0
    count = 0
    #for i in range(RUNNING_MODEL.BATCH_SIZE):
    #    survival_rate = RUNNING_MODEL.map_parameter[i][0:RUNNING_MODEL.tf_bid_len[i]]

    #    dead_rate = tf.subtract(tf.constant(1.0, dtype=tf.float32), survival_rate)
    #    predict = tf.transpose(tf.stack([survival_rate, dead_rate])) 

        #true_label.append(RUNNING_MODEL.tf_y2[id:id+RUNNING_MODEL.tf_bid_len[i]])
        #id = id + RUNNING_MODEL.tf_bid_len[i]
        #true_label.append(self.tf_y2[id:id+self.tf_bid_len[i]])
        #predicted_label.append(predict)
        #for i in range(self.tf_bid_len[i].shape):
            #default.append(dead_rate[i])
       
    for i in range(0, int(RUNNING_MODEL.test_data_win.size / RUNNING_MODEL.BATCH_SIZE)):

        test_batch_x, test_batch_x2, test_batch_y, test_batch_len, test_batch_y2, test_batch_x_deli, test_batch_seq = RUNNING_MODEL.test_data_win.next(
            RUNNING_MODEL.BATCH_SIZE)
        #test_batch_y = tf.ragged.RaggedTensorValue(test_batch_y,np.array([self.BATCH_SIZE]))
        #start_time = time.time()
        count_predict = 0

        #customized loss function according to the number of the remaining test data
        for j in range(RUNNING_MODEL.BATCH_SIZE):  
            if(j == 0):
                if(j == 11):
                    #print(j)
                    length = RUNNING_MODEL.tf_bid_len[j]
                    survival_rate0 = RUNNING_MODEL.map_parameter[j][40:RUNNING_MODEL.tf_bid_len[j]]
                survival_rate = RUNNING_MODEL.map_parameter[j][40:RUNNING_MODEL.tf_bid_len[j]]
                dead_rate = tf.subtract(tf.constant(1.0, dtype=tf.float32), survival_rate)
                predict = tf.transpose(tf.stack([survival_rate, dead_rate])) 
                id = RUNNING_MODEL.tf_bid_len[j]-40
                cross_entropy2 = -tf.reduce_sum(RUNNING_MODEL.tf_y2[0:id]*tf.log(tf.clip_by_value(predict,1e-10,1.0)))
                if(count_predict == 0):
                    predicted_label2 = predict
                    true_label2 = RUNNING_MODEL.tf_y2[0:id]
                    seqlen2 = tf.stack([tf.cast(RUNNING_MODEL.tf_bid_len[0]-40,dtype=tf.float32),0.0])
                else:
                    predicted_label2 = tf.concat([predicted_label2, tf.cast(predict, dtype=tf.float32)], 0)
                    true_label2 = tf.concat([true_label2, tf.cast(RUNNING_MODEL.tf_y2[0:id], dtype=tf.float32)], 0)
                    seqlen2 = tf.concat([seqlen2, tf.stack([tf.cast(RUNNING_MODEL.tf_bid_len[0]-40,dtype=tf.float32),0.0])], 0)
                count_predict = count_predict + 1
                #id = id + RUNNING_MODEL.tf_bid_len[i]
            else:
                survival_rate = RUNNING_MODEL.map_parameter[j][40:RUNNING_MODEL.tf_bid_len[j]]
                dead_rate = tf.subtract(tf.constant(1.0, dtype=tf.float32), survival_rate)
                predict = tf.transpose(tf.stack([survival_rate, dead_rate])) 
                cross_entropy2 = cross_entropy2 -tf.reduce_sum(RUNNING_MODEL.tf_y2[id:id+RUNNING_MODEL.tf_bid_len[j]-40]*tf.log(tf.clip_by_value(predict,1e-10,1.0)))

                if(count_predict == 0):
                    predicted_label2 = predict
                    true_label2 = RUNNING_MODEL.tf_y2[id:id+RUNNING_MODEL.tf_bid_len[j]-40]
                    seqlen2 = tf.stack([tf.cast(RUNNING_MODEL.tf_bid_len[j]-40,dtype=tf.float32),0.0])
                else:
                    predicted_label2 = tf.concat([predicted_label2, tf.cast(predict, dtype=tf.float32)], 0)
                    true_label2 = tf.concat([true_label2, tf.cast(RUNNING_MODEL.tf_y2[id:id+RUNNING_MODEL.tf_bid_len[j]-40], dtype=tf.float32)], 0)
                    seqlen2 = tf.concat([seqlen2, tf.stack([tf.cast(RUNNING_MODEL.tf_bid_len[j]-40,dtype=tf.float32),0.0])], 0)
                count_predict = count_predict + 1
                id = id + RUNNING_MODEL.tf_bid_len[j] - 40
            #true_label.append(self.tf_y2[id:id+self.tf_bid_len[i]])


            #predicted_label = tf.concat([input_x, tf.cast(input_x2, dtype=tf.float32)], 0)



            #predicted_label.append(predict)
            #for i in range(self.tf_bid_len[i].shape):
                #default.append(dead_rate[i])

        all_count = 0
        for m in range(RUNNING_MODEL.BATCH_SIZE):
            all_count = all_count + RUNNING_MODEL.tf_bid_len[m] - 40
        cross_entropy2 = cross_entropy2/tf.cast(all_count,dtype=tf.float32)


        bid_loss,test_survival_rate,test_t2,test_true_label,test_predicted_label,test_seqlen,test_count= sess.run(
            [cross_entropy2,survival_rate,RUNNING_MODEL.t2,true_label2,predicted_label2,seqlen2,all_count],
            feed_dict={RUNNING_MODEL.tf_x: test_batch_x,
                        RUNNING_MODEL.tf_x2: test_batch_x2,
                        RUNNING_MODEL.tf_y: test_batch_y,
                        RUNNING_MODEL.tf_y2: test_batch_y2,
                        RUNNING_MODEL.tf_x_deli: test_batch_x_deli,
                        #RUNNING_MODEL.tf_seq: test_batch_seq,
                        RUNNING_MODEL.tf_bid_len: test_batch_len
                        #RUNNING_MODEL.tf_training: False    
                        #self.tf_market_price: test_batch_market_price
                        })

        count = count + test_count
        loss_arr.append(bid_loss*test_count)
        true_label.append(test_true_label)
        predicted_label.append(test_predicted_label)
        seqlen.append(test_seqlen)
        #if(RUNNING_MODEL.SHOW_SURVIVAL_CURVE == True):

        

    #usually the number of the last test batch is not equal to the batch size
    remaining = RUNNING_MODEL.test_data_win.size - int(RUNNING_MODEL.test_data_win.size / RUNNING_MODEL.BATCH_SIZE)*RUNNING_MODEL.BATCH_SIZE

    if(remaining != 0):
        test_batch_x, test_batch_x2, test_batch_y, test_batch_len, test_batch_y2, test_batch_x_deli, test_batch_seq = RUNNING_MODEL.test_data_win.next(
            remaining)
        #test_batch_y = tf.ragged.RaggedTensorValue(test_batch_y,np.array([self.BATCH_SIZE]))
        #start_time = time.time()

    for i in range(remaining):
        survival_rate = RUNNING_MODEL.map_parameter[i][40:RUNNING_MODEL.tf_bid_len[i]]

    count_predict = 0

    #customized loss function according to the number of the remaining test data
    for i in range(remaining):  
        if(i == 0):

            survival_rate = RUNNING_MODEL.map_parameter[i][40:RUNNING_MODEL.tf_bid_len[i]]
            dead_rate = tf.subtract(tf.constant(1.0, dtype=tf.float32), survival_rate)
            predict = tf.transpose(tf.stack([survival_rate, dead_rate])) 
            id = RUNNING_MODEL.tf_bid_len[i]-40
            cross_entropy2 = -tf.reduce_sum(RUNNING_MODEL.tf_y2[0:id]*tf.log(tf.clip_by_value(predict,1e-10,1.0)))



            if(count_predict == 0):
                predicted_label2 = predict
                true_label2 = RUNNING_MODEL.tf_y2[0:id]
                seqlen2 = tf.stack([tf.cast(RUNNING_MODEL.tf_bid_len[0]-40,dtype=tf.float32),0.0])
            else:
                predicted_label2 = tf.concat([predicted_label2, tf.cast(predict, dtype=tf.float32)], 0)
                true_label2 = tf.concat([true_label2, tf.cast(RUNNING_MODEL.tf_y2[0:id], dtype=tf.float32)], 0)
                seqlen2 = tf.concat([seqlen2, tf.stack([tf.cast(RUNNING_MODEL.tf_bid_len[0]-40,dtype=tf.float32),0.0])], 0)
            count_predict = count_predict + 1
            #id = id + RUNNING_MODEL.tf_bid_len[i]
        else:
            if(i == 12):
                length = RUNNING_MODEL.tf_bid_len[i]
                survival_rate0 = RUNNING_MODEL.map_parameter[i][40:RUNNING_MODEL.tf_bid_len[i]]
            survival_rate = RUNNING_MODEL.map_parameter[i][40:RUNNING_MODEL.tf_bid_len[i]]
            dead_rate = tf.subtract(tf.constant(1.0, dtype=tf.float32), survival_rate)
            predict = tf.transpose(tf.stack([survival_rate, dead_rate])) 
            cross_entropy2 = cross_entropy2 -tf.reduce_sum(RUNNING_MODEL.tf_y2[id:id+RUNNING_MODEL.tf_bid_len[i]-40]*tf.log(tf.clip_by_value(predict,1e-10,1.0)))

            if(count_predict == 0):
                predicted_label2 = predict
                true_label2 = RUNNING_MODEL.tf_y2[id:id+RUNNING_MODEL.tf_bid_len[i]-40]
                seqlen2 = tf.stack([tf.cast(RUNNING_MODEL.tf_bid_len[i]-40,dtype=tf.float32),0.0])
            else:
                predicted_label2 = tf.concat([predicted_label2, tf.cast(predict, dtype=tf.float32)], 0)
                true_label2 = tf.concat([true_label2, tf.cast(RUNNING_MODEL.tf_y2[id:id+RUNNING_MODEL.tf_bid_len[i]-40], dtype=tf.float32)], 0)
                seqlen2 = tf.concat([seqlen2, tf.stack([tf.cast(RUNNING_MODEL.tf_bid_len[i]-40,dtype=tf.float32),0.0])], 0)
            count_predict = count_predict + 1
            id = id + RUNNING_MODEL.tf_bid_len[i] - 40
        #true_label.append(self.tf_y2[id:id+self.tf_bid_len[i]])


        #predicted_label = tf.concat([input_x, tf.cast(input_x2, dtype=tf.float32)], 0)



        #predicted_label.append(predict)
        #for i in range(self.tf_bid_len[i].shape):
            #default.append(dead_rate[i])

    all_count = RUNNING_MODEL.tf_bid_len[0] - 40
    for i in range(1,remaining):
        all_count = all_count + RUNNING_MODEL.tf_bid_len[i] - 40
    cross_entropy2 = cross_entropy2/tf.cast(all_count,dtype=tf.float32)



    bid_loss,test_survival_rate,test_t2,test_true_label,test_predicted_label,test_seqlen,test_count= sess.run(
        [cross_entropy2,survival_rate,RUNNING_MODEL.t2,true_label2,predicted_label2,seqlen2,all_count],
        feed_dict={RUNNING_MODEL.tf_x: test_batch_x,
                    RUNNING_MODEL.tf_x2: test_batch_x2,
                    RUNNING_MODEL.tf_y: test_batch_y,
                    RUNNING_MODEL.tf_y2: test_batch_y2,
                    RUNNING_MODEL.tf_x_deli: test_batch_x_deli,
                    #RUNNING_MODEL.tf_seq: test_batch_seq,
                    RUNNING_MODEL.tf_bid_len: test_batch_len
                    #RUNNING_MODEL.tf_training: False
                    #self.tf_market_price: test_batch_market_price
                    })
    #print('lentgh:',test_length)


    count = count + test_count
    loss_arr.append(bid_loss*test_count)
    true_label.append(test_true_label)
    seqlen.append(test_seqlen)
    predicted_label.append(test_predicted_label)
    #if(RUNNING_MODEL.SHOW_SURVIVAL_CURVE == True):

    #print('test_trainable_attention_mul_weight',test_trainable_attention_mul_weight)

    #print(attention_weighted_output):
    #print('first attention_weighted_output:',test_attention_weighted_output[0])

   
    mean_loss = 0
    for i in range(len(loss_arr)):
        mean_loss = mean_loss + loss_arr[i]
    mean_loss = mean_loss/count    

    #for log_likelihoo_ratio test
    llr_test_3lstm_attention_deli = -mean_loss*count 

    #mean_loss = np.array(loss_arr).mean()

    #print('length of true label:',len(true_label))

    a = []
    b = []
    c = []
    for i in range(len(true_label)):
        for j in range(len(true_label[i])):

            a.append(true_label[i][j])
            b.append(predicted_label[i][j])

    for i in range(len(seqlen)):
        for j in range(len(seqlen[i])):
            c.append(seqlen[i][j])

    true_label = a
    predicted_label = b
    seqlen = c


    true_label = np.array(true_label)
    predicted_label = np.array(predicted_label)
    seqlen = np.array(seqlen)


    default = 0
    for i in range(len(predicted_label)):
        default = default + predicted_label[i][1]

    label_true = []
    for i in range(len(true_label)):
        label_true.append(true_label[i][1])

    prediction = []
    for i in range(len(predicted_label)):
        prediction.append(predicted_label[i][1])

    unbalanced_prediction = []
    for i in range(len(predicted_label)):
        unbalanced_prediction.append(0.1*predicted_label[i][1]/(0.1*predicted_label[i][1] - predicted_label[i][1] + 1))

    unbalanced_default = 0
    for i in range(len(predicted_label)):
        unbalanced_default = unbalanced_default + unbalanced_prediction[i]

    unbalanced_log_likeli_3lstm_attention_deli = 0
    for i in range(len(unbalanced_prediction)):
        if(true_label[i][0] == 1.0):
            unbalanced_log_likeli_3lstm_attention_deli = unbalanced_log_likeli_3lstm_attention_deli + math.log(1-unbalanced_prediction[i])
        else:
            unbalanced_log_likeli_3lstm_attention_deli = unbalanced_log_likeli_3lstm_attention_deli + math.log(unbalanced_prediction[i])


    #AUC    
    auc_score = roc_auc_score(label_true,prediction)


    auc_score = roc_auc_score(label_true,unbalanced_prediction)

    general_auc = auc_score


    bad = []
    mean_bad = 0
    mean_good = 0
    predicted_bad = []
    for i in range(len(true_label)):

        if(true_label[i][0] == 1.0):

            count_good = count_good + 1

        if(true_label[i][0] == 0.0):
            bad.append(unbalanced_prediction[i]) 
            count_bad = count_bad + 1

    all_true = 0
    for i in range(len(true_label)):
        if(true_label[i][0] == 1.0):
            all_true = all_true + 0
        if(true_label[i][0] == 0.0):
            all_true = all_true + 1



    dr = count_bad/(count_bad + count_good)

    baseline_llr = -dr*math.log(dr)-(1-dr)*math.log(1-dr)




    #for log_likelihoo_ratio test
    llr_test_unbalanced_3lstm_attention_deli_lag3_12 = unbalanced_log_likeli_3lstm_attention_deli 

    unbalanced_prediction_3lstm_attention_deli_lag3_12 = unbalanced_prediction

    seqlen2 = []
    for i in range(len(seqlen)):
        if(seqlen[i]!=0):
            seqlen2.append(int(seqlen[i]))



    #time dependent AUC
    is_default = 0
    #set the time windows:
    time_window = [24,36,60]

    duration = []

    AUC = []

    #get the conditional default status based on the current time window
    conditional_labels = []

    predict = []
    prediction = []

    brier_score = 0

    id = 0
    for m in range(len(time_window)): 
        id = 0
        prediction = []
        conditional_labels = []
        is_default = 0
        for i in range(len(seqlen2)):
            #if max_age <= time_window
            if(seqlen2[i] <= time_window[m]):
                #1
                predict = unbalanced_prediction_3lstm_attention_deli_lag3_12[id:id+int(seqlen2[i])]
                h0 = predict[0]
                survival = 1-h0
                for j in range(1,len(predict)):
                    survival = survival*(1-predict[j])
                #print('survival1:',survival)
                prediction.append(1-survival)
                #prediction.append(np.sum(predict))
                #print(x2[id+seqlen[i]-1])
                if(true_label[id+int(seqlen2[i])-1][0] == 1.0):
                    conditional_labels.append(0)
                else:
                    conditional_labels.append(1)
                    is_default = 1

            #max_age > time_window
            else:
                predict = unbalanced_prediction_3lstm_attention_deli_lag3_12[id:int(id+time_window[m])]
                h0 = predict[0]
                survival = 1-h0
                for j in range(1,len(predict)):
                    survival = survival*(1-predict[j])
                #print('survival2:',survival)
                prediction.append(1-survival)
                #prediction.append(np.sum(predict))

                conditional_labels.append(0)
            id = id + int(seqlen2[i])
        #AUC
        if(is_default != 0):
            auc_score = roc_auc_score(conditional_labels,prediction)
            AUC.append(auc_score)


    #print(conditional_labels)
    AUC = np.array(AUC)    

    AUC24 = AUC




    brier_score = 0
    for i in range(len(unbalanced_prediction_3lstm_attention_deli_lag3_12)):
        brier_score = brier_score + (unbalanced_prediction_3lstm_attention_deli_lag3_12[i]-int(true_label[i][1]))**2
    #print('3lstm_attention_deli brier score: ',brier_score/len(unbalanced_prediction_3lstm_attention_deli_lag3_12))
    
    #fTo make the PRS indicator clear
    PRS = (1-(-unbalanced_log_likeli_3lstm_attention_deli)/len(unbalanced_prediction)/baseline_llr)
    
    print(PRS,AUC24.mean(),brier_score/len(unbalanced_prediction_3lstm_attention_deli_lag3_12))
    overall_PRS += PRS
    overall_AUC += AUC24.mean()
    overall_BS += brier_score/len(unbalanced_prediction_3lstm_attention_deli_lag3_12)
    
    
    if((q+1)<=4):
        q = q + 1
    else:
        q = 1
        ym += 1
        
#print("overall: ", str(float(overall_PRS)/n_vintage)+"(Higer is better)",str(float(overall_AUC)/n_vintage)+"(Higer is better)",str(overall_BS)+"(Lower is better)")

path = os.getcwd()
new_path = path.replace("\\","/")

csvFile = open(new_path + "/Replication_IJF/data/2259/summary.csv", "r",encoding='gb18030', errors='ignore')
reader = csv.reader(csvFile)


result = []
for item in reader:
    data = []
    # 忽略第一行
    #if reader.line_num == 1:
        #continue
    for i in range(13):
        data.append(item[i])
    result.append(data)

csvFile.close()
df = pd.DataFrame(result)
print("Summary:")
print(df.to_string(index=False))
