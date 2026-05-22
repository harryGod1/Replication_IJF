#The experimental results can be reproduced by running the corresponding trainning code (located in the training directory) to perform model training under different washout step configurations.
#Model weights are saved in the saved_model folder within the saved directory.           
#The code presented here enables readers to train and validate the model independently.
#The following code provides an example of validating the output results. Readers may adapt it as needed.
########################################################################################################
exposure = False

import os
import tensorflow.compat.v1 as tf
from sklearn.metrics import roc_curve, auc
import matplotlib as mpl  
import matplotlib.pyplot as plt
import numpy as nplanced
import sklearn
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
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
import csv
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

ym = 2004
q = 1
n_vintage = 1 # number of sub datasets users want to test
print("Loading may take a while, please wait.")
print("This needs to be run repeatedly, outputting the results for each quarter between 2004 and 2024.")
print("A higher Pseudo-R-Square score indicates better model performance. Model performance can be compared by comparing the scores of each model in each round. The final results are summarized in a graph, as shown in Figure 11 of the paper.")

for prs in range(n_vintage):
    if(prs == 0):
        pseudo_dtsm = []
        pseudo_gam = []
        pseudo_cox = []
        pseudo_weibull = []
        pseudo_deephit = []
        pseudo_3lstm = []
    if(prs == 48):
        pseudo_dtsm = []
        pseudo_gam = []
        pseudo_cox = []
        pseudo_weibull = []
        pseudo_deephit = []
        pseudo_3lstm = []
    print("==================================================")
    #print number of rounds of the comparative results
    print(prs+1)

    print('Linear DTSM ...')

    if(prs<48):
        #DTSM training forecast
        path = os.getcwd()
        new_path = path.replace("\\","/")
        f2 = open(new_path + "/Replication_IJF/data/2259/train_all_dtsm_x_forecast_Deli_random_debug0.txt")

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
            b2.append(test_OS[i])
            b2.append(test_DTI[i])
            b2.append(test_UPB[i])
            b2.append(test_LTV[i])
            b2.append(test_IR[i])
            #b2.append(test_Channel[i])
            b2.append(test_PT[i])
            b2.append(test_LP[i])
            b2.append(test_NB[i])
            b2.append(test_Age[i])
            b2.append(test_vintage[i])
            #b2.append(test_calendar[i])
            if(int(test_Age[i])<=2):
                b2.append(0)
            else:
                s_p3 = test_data[i-3].split(' ')
                b2.append(int(s_p3[16]))
            current_deli.append(int(line_data[16]))
            final_data.append(line_data)
            #b2.append(test_Deli[i])
            y.append(test_Def[i])
            x.append(b2)


            if(str(new_seq) == str(last_seq) and test_Age[i] > max_seqlen):
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
                if((max_seqlen+1 != (len(current_deli)-1)) or (max_seqlen == 0)):

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
            if(max_seqlen+1 != len(current_deli)):
                #print('last one')
                #print('max_seqlen+1:',max_seqlen+1)
                #print('current_deli:',len(current_deli)-1)
                #print('current_seq:',final_data[len(final_data)-1])
                #print('current_x:',x[len(x)-1])
                #print('current_y:',y[len(y)-1])

                #important!!!, cause to wrong training: need to delete the from the last index, not the current index
                del x[len(x)-len(current_deli):len(x)]
                del y[len(y)-len(current_deli):len(y)]
                del final_data[len(final_data)-len(current_deli):len(final_data)]
                #print('after_delete_seq:',final_data[len(final_data)-1])
                #print('after_delete_x:',x[len(x)-1])
                #print('after_delete_y:',y[len(y)-1])


        #print(x2.shape)
        #print(y2.shape)



        #print(len(x),len(x[0]))

        #优化版本，据集每一笔最后都多了一笔月份为0地数据 会导致default数据被误认为non default（因为last seq ！= new seq部分的判定原因，导致default 1 之后多出了个default0得标签，会误将default0
        #标签取代原本default应该是1得标签）e.g.,F113Q3250129，F112Q4005541
        #！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        #Delinquency版本中去除大部分max_seqlen和deli两者数据集不相等的账户（和上面版本的区别！）
        #！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        #最终版本Linear DTSM代码 -- 导入数据集部分
        #DTSM testing
        #f1 = open("./train_data.txt")
        path = os.getcwd()
        new_path = path.replace("\\","/")
        f2 = open(new_path + "/Replication_IJF/data/2259/test_all_dtsm_x_forecast_Deli_random_debug0.txt")
        #f2 = open("./test4_data.txt")
        #f2 = open("./test3.txt")
        #train_data = f1.readline()
        #test_data = f2.readline()

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



        x2 = []
        y2 = []
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
            b2.append(test_OS[i])
            b2.append(test_DTI[i])
            b2.append(test_UPB[i])
            b2.append(test_LTV[i])
            b2.append(test_IR[i])
            #b2.append(test_Channel[i])
            b2.append(test_PT[i])
            b2.append(test_LP[i])
            b2.append(test_NB[i])
            b2.append(test_Age[i])
            b2.append(test_vintage[i])
            #b2.append(test_calendar[i])
            #b2.append(test_SEQ[i]) just for test if the result excluded the exception
            if(int(test_Age[i])<=2):
                b2.append(0)
            else:
                s_p3 = test_data[i-3].split(' ')
                b2.append(int(s_p3[16]))
            current_deli.append(int(line_data[16]))
            final_data.append(line_data)
            y2.append(test_Def[i])
            x2.append(b2)


            if(str(new_seq) == str(last_seq) and test_Age[i] > max_seqlen):
                max_seqlen = test_Age[i]

            if(test_Def[i] == 1):
                is_default = 1
            if(len(last_seq) ==  0):
                last_seq = test_SEQ[0] 
            if(str(new_seq) == str(last_seq) and is_default == 0 and int(test_Age[i]) < max_seqlen ):
                del x2[len(x2)-1]
                del y2[len(y2)-1]
                del current_deli[len(current_deli)-1]
                #print('last one: ',final_data[len(final_data)-2])
                #print('delete current: ',final_data[len(final_data)-1])
                del final_data[len(final_data)-1]
                #print('Exception1!')
            if(str(new_seq) == str(last_seq) and is_default == 1 and test_Def[i] == 0):
                del x2[len(x2)-1]
                del y2[len(y2)-1]
                del current_deli[len(current_deli)-1]
                #print('last one: ',final_data[len(final_data)-2])
                #print('delete current: ',final_data[len(final_data)-1])
                del final_data[len(final_data)-1]
                #print('after delete: ',final_data[len(final_data)-1])
                #print('after delete x: ',x2[len(x2)-1])
                #print('after delete y:',y2[len(y2)-1])
                #print('Exception2!')
            if(str(new_seq) != str(last_seq)):
                if((max_seqlen+1 != (len(current_deli)-1)) or (max_seqlen == 0)):

                    #print('max_seqlen+1:',max_seqlen+1)
                    #print('current_deli:',len(current_deli)-1)
                    #print('current_seq:',final_data[len(final_data)-2])
                    #print('current_x:',x2[len(x2)-2])
                    #print('current_y:',y2[len(y2)-2])

                    #important!!!, cause to wrong training: need to delete the from the last index, not the current index
                    del x2[len(x2)-len(current_deli):len(x2)-1]
                    del y2[len(y2)-len(current_deli):len(y2)-1]
                    del final_data[len(final_data)-len(current_deli):len(final_data)-1]
                    #print('after_delete_seq:',final_data[len(final_data)-2])
                    #print('after_delete_x:',x2[len(x2)-2])
                    #print('after_delete_y:',y2[len(y2)-2])



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
            if(max_seqlen+1 != len(current_deli)):

                #print('max_seqlen+1:',max_seqlen+1)
                #print('current_deli:',len(current_deli)-1)
                #print('current_seq:',final_data[len(final_data)-1])
                #print('current_x:',x2[len(x2)-1])
                #print('current_y:',y2[len(y2)-1])

                #important!!!, cause to wrong training: need to delete the from the last index, not the current index
                del x2[len(x2)-len(current_deli):len(x2)]
                del y2[len(y2)-len(current_deli):len(y2)]
                del final_data[len(final_data)-len(current_deli):len(final_data)]
                #print('after_delete_seq:',final_data[len(final_data)-1])
                #print('after_delete_x:',x2[len(x2)-1])
                #print('after_delete_y:',y2[len(y2)-1])

        #print(x2.shape)
        #print(y2.shape)

        #print(len(x2),len(x2[0]))

        #最终版本Linear DTSM代码 -- 给类别变量编码的部分
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #For delinquency version
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        for i in range(len(x)):
            #FTHF:N=1,Y=2,9=3
            #OS:P=1,I=2,S=3,9=4
            if(x[i][1]=='P'):
                x[i][1]=1
            if(x[i][1]=='I'):
                x[i][1]=2
            if(x[i][1]=='S'):
                x[i][1]=3   
            #if(x[i][2]=='9'):
                #x[i][2]=4 
            #Channel:R=1,B=2,C=3,T=4,9=5
            #PT:PU=1,SF=2,CO=3,MH&CP&9=4
            if(x[i][6]=='PU'):
                x[i][6]=1
            if(x[i][6]=='SF'):
                x[i][6]=2
            if(x[i][6]=='CO'):
                x[i][6]=3   
            if(x[i][6]=='MH'):
                x[i][6]=2 
            if(x[i][6]=='CP'):
                x[i][6]=2 
            #if(x[i][8]=='9'):
                #x[i][8]=4 
            #LP:P=1,C=2,N=3,R=4,9=5
            if(x[i][7]=='P'):
                x[i][7]=1
            if(x[i][7]=='C'):
                x[i][7]=2
            if(x[i][7]=='N'):
                x[i][7]=3   
            #if(x[i][9]=='R'):
                #x[i][9]=4 
            #if(x[i][9]=='9'):
                #x[i][9]=5 
            #NB:1=1,2=2,9=3
            if(x[i][8]==1):
                x[i][8]=0 
            if(x[i][8]==2):
                x[i][8]=1 
            if(x[i][8]==99):
                x[i][8]=0.5   
            #Delinquency:0=0,1=1,2=2,3=3
            if(x[i][11]==0):
                x[i][11]=0 
            if(x[i][11]==1):
                x[i][11]=1 
            if(x[i][11]==2):
                x[i][11]=2 
            if(x[i][11]==3):
                x[i][11]=3 

        #print(x[0])

        #最终版本Linear DTSM代码 -- 给类别变量编码的部分
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #For delinquency version
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        for i in range(len(x2)):
            #FTHF:N=1,Y=2,9=3
            #OS:P=1,I=2,S=3,9=4
            if(x2[i][1]=='P'):
                x2[i][1]=1
            if(x2[i][1]=='I'):
                x2[i][1]=2
            if(x2[i][1]=='S'):
                x2[i][1]=3   
            #if(x[i][1]=='9'):
                #x[i][1]=4 
            #Channel:R=1,B=2,C=3,T=4,9=5
            #PT:PU=1,SF=2,CO=3,MH&CP&9=4
            if(x2[i][6]=='PU'):
                x2[i][6]=1
            if(x2[i][6]=='SF'):
                x2[i][6]=2
            if(x2[i][6]=='CO'):
                x2[i][6]=3   
            if(x2[i][6]=='MH'):
                x2[i][6]=2 
            if(x2[i][6]=='CP'):
                x2[i][6]=2 
            #if(x[i][8]=='9'):
                #x[i][8]=4 
            #LP:P=1,C=2,N=3,R=4,9=5
            if(x2[i][7]=='P'):
                x2[i][7]=1
            if(x2[i][7]=='C'):
                x2[i][7]=2
            if(x2[i][7]=='N'):
                x2[i][7]=3   
            #if(x[i][9]=='R'):
                #x[i][9]=4 
            #if(x[i][9]=='9'):
                #x[i][9]=5 
            #NB:1=1,2=2,9=3
            if(x2[i][8]==1):
                x2[i][8]=0 
            if(x2[i][8]==2):
                x2[i][8]=1 
            if(x2[i][8]==99):
                x2[i][8]=0.5   
            #Delinquency:0=0,1=1,2=2,3=3
            if(x2[i][11]==0):
                x2[i][11]=0 
            if(x2[i][11]==1):
                x2[i][11]=1 
            if(x2[i][11]==2):
                x2[i][11]=2 
            if(x2[i][11]==3):
                x2[i][11]=3 
    else:
        #DTSM training forecast
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
            b2.append(test_OS[i])
            b2.append(test_DTI[i])
            b2.append(test_UPB[i])
            b2.append(test_LTV[i])
            b2.append(test_IR[i])
            #b2.append(test_Channel[i])
            b2.append(test_PT[i])
            b2.append(test_LP[i])
            b2.append(test_NB[i])
            b2.append(test_Age[i])
            b2.append(test_vintage[i])
            #b2.append(test_calendar[i])
            if(int(test_Age[i])<=2):
                b2.append(0)
            else:
                s_p3 = test_data[i-3].split(' ')
                b2.append(int(s_p3[16]))
            current_deli.append(int(line_data[16]))
            final_data.append(line_data)
            #b2.append(test_Deli[i])
            y.append(test_Def[i])
            x.append(b2)


            if(str(new_seq) == str(last_seq) and test_Age[i] > max_seqlen):
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
                if((max_seqlen+1 != (len(current_deli)-1)) or (max_seqlen == 0)):

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
            if(max_seqlen+1 != len(current_deli)):
                #print('last one')
                #print('max_seqlen+1:',max_seqlen+1)
                #print('current_deli:',len(current_deli)-1)
                #print('current_seq:',final_data[len(final_data)-1])
                #print('current_x:',x[len(x)-1])
                #print('current_y:',y[len(y)-1])

                #important!!!, cause to wrong training: need to delete the from the last index, not the current index
                del x[len(x)-len(current_deli):len(x)]
                del y[len(y)-len(current_deli):len(y)]
                del final_data[len(final_data)-len(current_deli):len(final_data)]
                #print('after_delete_seq:',final_data[len(final_data)-1])
                #print('after_delete_x:',x[len(x)-1])
                #print('after_delete_y:',y[len(y)-1])


        #print(x2.shape)
        #print(y2.shape)



        #print(len(x),len(x[0]))

        #优化版本，据集每一笔最后都多了一笔月份为0地数据 会导致default数据被误认为non default（因为last seq ！= new seq部分的判定原因，导致default 1 之后多出了个default0得标签，会误将default0
        #标签取代原本default应该是1得标签）e.g.,F113Q3250129，F112Q4005541
        #！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        #Delinquency版本中去除大部分max_seqlen和deli两者数据集不相等的账户（和上面版本的区别！）
        #！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        #最终版本Linear DTSM代码 -- 导入数据集部分
        #DTSM testing
        #f1 = open("./train_data.txt")
        path = os.getcwd()
        new_path = path.replace("\\","/")
        f2 = open(new_path + "/Replication_IJF/data/2259/test16to21_all_dtsm_x_forecast_Deli_random_debug0.txt")
        #f2 = open("./test4_data.txt")
        #f2 = open("./test3.txt")
        #train_data = f1.readline()
        #test_data = f2.readline()

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



        x2 = []
        y2 = []
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
            b2.append(test_OS[i])
            b2.append(test_DTI[i])
            b2.append(test_UPB[i])
            b2.append(test_LTV[i])
            b2.append(test_IR[i])
            #b2.append(test_Channel[i])
            b2.append(test_PT[i])
            b2.append(test_LP[i])
            b2.append(test_NB[i])
            b2.append(test_Age[i])
            b2.append(test_vintage[i])
            #b2.append(test_calendar[i])
            #b2.append(test_SEQ[i]) just for test if the result excluded the exception
            if(int(test_Age[i])<=2):
                b2.append(0)
            else:
                s_p3 = test_data[i-3].split(' ')
                b2.append(int(s_p3[16]))
            current_deli.append(int(line_data[16]))
            final_data.append(line_data)
            y2.append(test_Def[i])
            x2.append(b2)


            if(str(new_seq) == str(last_seq) and test_Age[i] > max_seqlen):
                max_seqlen = test_Age[i]

            if(test_Def[i] == 1):
                is_default = 1
            if(len(last_seq) ==  0):
                last_seq = test_SEQ[0] 
            if(str(new_seq) == str(last_seq) and is_default == 0 and int(test_Age[i]) < max_seqlen ):
                del x2[len(x2)-1]
                del y2[len(y2)-1]
                del current_deli[len(current_deli)-1]
                #print('last one: ',final_data[len(final_data)-2])
                #print('delete current: ',final_data[len(final_data)-1])
                del final_data[len(final_data)-1]
                #print('Exception1!')
            if(str(new_seq) == str(last_seq) and is_default == 1 and test_Def[i] == 0):
                del x2[len(x2)-1]
                del y2[len(y2)-1]
                del current_deli[len(current_deli)-1]
                #print('last one: ',final_data[len(final_data)-2])
                #print('delete current: ',final_data[len(final_data)-1])
                del final_data[len(final_data)-1]
                #print('after delete: ',final_data[len(final_data)-1])
                #print('after delete x: ',x2[len(x2)-1])
                #print('after delete y:',y2[len(y2)-1])
                #print('Exception2!')
            if(str(new_seq) != str(last_seq)):
                if((max_seqlen+1 != (len(current_deli)-1)) or (max_seqlen == 0)):

                    #print('max_seqlen+1:',max_seqlen+1)
                    #print('current_deli:',len(current_deli)-1)
                    #print('current_seq:',final_data[len(final_data)-2])
                    #print('current_x:',x2[len(x2)-2])
                    #print('current_y:',y2[len(y2)-2])

                    #important!!!, cause to wrong training: need to delete the from the last index, not the current index
                    del x2[len(x2)-len(current_deli):len(x2)-1]
                    del y2[len(y2)-len(current_deli):len(y2)-1]
                    del final_data[len(final_data)-len(current_deli):len(final_data)-1]
                    #print('after_delete_seq:',final_data[len(final_data)-2])
                    #print('after_delete_x:',x2[len(x2)-2])
                    #print('after_delete_y:',y2[len(y2)-2])



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
            if(max_seqlen+1 != len(current_deli)):

                #print('max_seqlen+1:',max_seqlen+1)
                #print('current_deli:',len(current_deli)-1)
                #print('current_seq:',final_data[len(final_data)-1])
                #print('current_x:',x2[len(x2)-1])
                #print('current_y:',y2[len(y2)-1])

                #important!!!, cause to wrong training: need to delete the from the last index, not the current index
                del x2[len(x2)-len(current_deli):len(x2)]
                del y2[len(y2)-len(current_deli):len(y2)]
                del final_data[len(final_data)-len(current_deli):len(final_data)]
                #print('after_delete_seq:',final_data[len(final_data)-1])
                #print('after_delete_x:',x2[len(x2)-1])
                #print('after_delete_y:',y2[len(y2)-1])

        #print(x2.shape)
        #print(y2.shape)

        #print(len(x2),len(x2[0]))

        #最终版本Linear DTSM代码 -- 给类别变量编码的部分
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #For delinquency version
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        for i in range(len(x)):
            #FTHF:N=1,Y=2,9=3
            #OS:P=1,I=2,S=3,9=4
            if(x[i][1]=='P'):
                x[i][1]=1
            if(x[i][1]=='I'):
                x[i][1]=2
            if(x[i][1]=='S'):
                x[i][1]=3   
            #if(x[i][2]=='9'):
                #x[i][2]=4 
            #Channel:R=1,B=2,C=3,T=4,9=5
            #PT:PU=1,SF=2,CO=3,MH&CP&9=4
            if(x[i][6]=='PU'):
                x[i][6]=1
            if(x[i][6]=='SF'):
                x[i][6]=2
            if(x[i][6]=='CO'):
                x[i][6]=3   
            if(x[i][6]=='MH'):
                x[i][6]=2 
            if(x[i][6]=='CP'):
                x[i][6]=2 
            #if(x[i][8]=='9'):
                #x[i][8]=4 
            #LP:P=1,C=2,N=3,R=4,9=5
            if(x[i][7]=='P'):
                x[i][7]=1
            if(x[i][7]=='C'):
                x[i][7]=2
            if(x[i][7]=='N'):
                x[i][7]=3   
            #if(x[i][9]=='R'):
                #x[i][9]=4 
            #if(x[i][9]=='9'):
                #x[i][9]=5 
            #NB:1=1,2=2,9=3
            if(x[i][8]==1):
                x[i][8]=0 
            if(x[i][8]==2):
                x[i][8]=1 
            if(x[i][8]==99):
                x[i][8]=0.5   
            #Delinquency:0=0,1=1,2=2,3=3
            if(x[i][11]==0):
                x[i][11]=0 
            if(x[i][11]==1):
                x[i][11]=1 
            if(x[i][11]==2):
                x[i][11]=2 
            if(x[i][11]==3):
                x[i][11]=3 

        #print(x[0])

        #最终版本Linear DTSM代码 -- 给类别变量编码的部分
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #For delinquency version
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        for i in range(len(x2)):
            #FTHF:N=1,Y=2,9=3
            #OS:P=1,I=2,S=3,9=4
            if(x2[i][1]=='P'):
                x2[i][1]=1
            if(x2[i][1]=='I'):
                x2[i][1]=2
            if(x2[i][1]=='S'):
                x2[i][1]=3   
            #if(x[i][1]=='9'):
                #x[i][1]=4 
            #Channel:R=1,B=2,C=3,T=4,9=5
            #PT:PU=1,SF=2,CO=3,MH&CP&9=4
            if(x2[i][6]=='PU'):
                x2[i][6]=1
            if(x2[i][6]=='SF'):
                x2[i][6]=2
            if(x2[i][6]=='CO'):
                x2[i][6]=3   
            if(x2[i][6]=='MH'):
                x2[i][6]=2 
            if(x2[i][6]=='CP'):
                x2[i][6]=2 
            #if(x[i][8]=='9'):
                #x[i][8]=4 
            #LP:P=1,C=2,N=3,R=4,9=5
            if(x2[i][7]=='P'):
                x2[i][7]=1
            if(x2[i][7]=='C'):
                x2[i][7]=2
            if(x2[i][7]=='N'):
                x2[i][7]=3   
            #if(x[i][9]=='R'):
                #x[i][9]=4 
            #if(x[i][9]=='9'):
                #x[i][9]=5 
            #NB:1=1,2=2,9=3
            if(x2[i][8]==1):
                x2[i][8]=0 
            if(x2[i][8]==2):
                x2[i][8]=1 
            if(x2[i][8]==99):
                x2[i][8]=0.5   
            #Delinquency:0=0,1=1,2=2,3=3
            if(x2[i][11]==0):
                x2[i][11]=0 
            if(x2[i][11]==1):
                x2[i][11]=1 
            if(x2[i][11]==2):
                x2[i][11]=2 
            if(x2[i][11]==3):
                x2[i][11]=3 

    #print(x2[2])

    #最终版本Linear DTSM代码 -- 训练模型的部分
    #Linear DTSM final
    from sklearn import preprocessing

    #print(x[0],x[1])
    x_s=x

    scaler = preprocessing.StandardScaler().fit(x_s)
    x = scaler.transform(x)
    #print(x[0],x[1])

    #用训练集的标准差进行标准化还是用测试集自己的标准差进行标准化？
    #x_s2=x2
    #scaler = preprocessing.StandardScaler().fit(x_s2)
    x2 = scaler.transform(x2)
    #print(x[0])
    #print(x2[0])
    #x_auc = scaler.transform(x_auc)
    #x2_auc = scaler.transform(x2_auc)

    #for i in range(len(x)):
    #    if(x[i][5]==-1.2333916623473458):
    #        count=count+1
    #print(count)
    #scaler2 = preprocessing.StandardScaler().fit(x2)
    #x2 = scaler2.transform(x2)

    ct = ColumnTransformer(
        [('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'), [1,6,7,9,10,11])],   # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
        remainder='passthrough'                                         # Leave the rest of the columns untouched
    )
    ct.fit(x)
    #x = ct.fit_transform(x)

    #x2 = ct.fit_transform(x2)

    x=ct.transform(x)
    x2=ct.transform(x2)


    #print(x[0])
    #print(x2[0])

    #x2 = []
    #print(x2)
    #print(x2[0][0,0])
    #for i in range(x2.shape[0]):
        #if(x2[i][0,0]):
            #print(x2[i])
            #x3.append(x2[i])


    #x_auc = ct.fit_transform(x_auc)

    #x2_auc = ct.fit_transform(x2_auc)


    model = LogisticRegression(solver='newton-cg',penalty='l2',C=0.1).fit(x, y)

    s = model.predict_proba(x2)
    #s2 = []
    #for i in range(len(x3)):
        #s2.append(model.predict_proba(x3[i])[0][1])

    #print(s2)

    #score = 0

    #for i in range(len(s2)):
        #score=score+math.log(s2[i]/(1-s2[i]))


    #print(score/len(s2))

    log_likely = 0
    for i in range(len(s)):
        if(y2[i] == 0):
            log_likely = log_likely + math.log(s[i][0])
        else:
            log_likely = log_likely + math.log(s[i][1])
    #print('log_likelyhood: ',(-log_likely)/len(s))

    #DTSM testing each vintage 
    vintage = 201204

    x2 = []
    y2 = []
    b2 = []


    if(int(ym%100)<10):
        path = os.getcwd()
        new_path = path.replace("\\","/")
        f2 = open(new_path + "/Replication_IJF/data/2259/" + "test" + "0" + str(int(ym%100)) + "_" + str(q) + "_unbalanced_dtsm_x_Deli.txt")
        #TEST_FILE = "./data/2259/" + "test" + "0" + str(int(ym%100)) + "_" + str(q) + "_unbalanced_dtsm_x_Deli.txt"
    else:
        path = os.getcwd()
        new_path = path.replace("\\","/")
        f2 = open(new_path + "/Replication_IJF/data/2259/" + "test" + str(int(ym%100)) + "_" + str(q) + "_unbalanced_dtsm_x_Deli.txt")
        #TEST_FILE = "./data/2259/" + "test" + str(int(ym%100)) + "_" + str(q) + "_unbalanced_dtsm_x_Deli.txt"
    #f2 = open(TEST_FILE)
    vintage = 201204
    #f2 = open("./test4_data.txt")
    #f2 = open("./test3.txt")
    #train_data = f1.readline()
    #test_data = f2.readline()

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
    seqlen = []



    x2 = []
    y2 = []
    b2 = []

    last_seq = ''
    new_seq = ''
    last_default = 0
    new_default = 0
    is_default = 0
    max_seqlen = 0
    count_len = 0

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
        b2.append(test_OS[i])
        b2.append(test_DTI[i])
        b2.append(test_UPB[i])
        b2.append(test_LTV[i])
        b2.append(test_IR[i])
        #b2.append(test_Channel[i])
        b2.append(test_PT[i])
        b2.append(test_LP[i])
        b2.append(test_NB[i])
        b2.append(test_Age[i])
        b2.append(test_vintage[i])
        #b2.append(test_calendar[i])
        #b2.append(test_SEQ[i]) just for test if the result excluded the exception
        if(int(test_Age[i])<=2):
            b2.append(0)
        else:
            s_p3 = test_data[i-3].split(' ')
            b2.append(int(s_p3[16]))
        current_deli.append(int(line_data[16]))
        final_data.append(line_data)
        y2.append(test_Def[i])
        x2.append(b2)



        if(str(new_seq) == str(last_seq) and test_Age[i] > max_seqlen):
            max_seqlen = test_Age[i]

        if(test_Def[i] == 1):
            is_default = 1
        if(len(last_seq) ==  0):
            last_seq = test_SEQ[0] 
        if(str(new_seq) == str(last_seq) and is_default == 0 and int(test_Age[i]) < max_seqlen ):
            del x2[len(x2)-1]
            del y2[len(y2)-1]
            del current_deli[len(current_deli)-1]
            #print('last one: ',final_data[len(final_data)-2])
            #print('delete current: ',final_data[len(final_data)-1])
            del final_data[len(final_data)-1]

            #print('Exception1!')
        if(str(new_seq) == str(last_seq) and is_default == 1 and test_Def[i] == 0):
            del x2[len(x2)-1]
            del y2[len(y2)-1]
            del current_deli[len(current_deli)-1]
            #print('last one: ',final_data[len(final_data)-2])
            #print('delete current: ',final_data[len(final_data)-1])

            #print('after delete: ',final_data[len(final_data)-1])
            #print('after delete x: ',x2[len(x2)-1])
            #print('after delete y:',y2[len(y2)-1])
            #print('Exception2!')
        if(str(new_seq) != str(last_seq)):
            if((max_seqlen+1 != (len(current_deli)-1)) or (max_seqlen == 0)):

                #print('max_seqlen+1:',max_seqlen+1)
                #print('current_deli:',len(current_deli)-1)
                #print('current_seq:',final_data[len(final_data)-2])
                #print('current_x:',x2[len(x2)-2])
                #print('current_y:',y2[len(y2)-2])

                #important!!!, cause to wrong training: need to delete the from the last index, not the current index
                del x2[len(x2)-len(current_deli):len(x2)-1]
                del y2[len(y2)-len(current_deli):len(y2)-1]
                del final_data[len(final_data)-len(current_deli):len(final_data)-1]

                #print('after_delete_seq:',final_data[len(final_data)-2])
                #print('after_delete_x:',x2[len(x2)-2])
                #print('after_delete_y:',y2[len(y2)-2])



            is_default = 0
            max_seqlen = 0 
            current_deli = []
            count_len = 1
            #cleaned but need to return this current data(current data belongs to the new seq)
            s = test_data[i].split(' ')
            slen = len(s)
            current_deli.append(int(s[slen-1])) 


        last_seq = new_seq        
        b2=[]

    if(max_seqlen != 0):
        if(max_seqlen+1 != len(current_deli)):

            #print('max_seqlen+1:',max_seqlen+1)
            #print('current_deli:',len(current_deli)-1)
            #print('current_seq:',final_data[len(final_data)-1])
            #print('current_x:',x2[len(x2)-1])
            #print('current_y:',y2[len(y2)-1])

            #important!!!, cause to wrong training: need to delete the from the last index, not the current index
            del x2[len(x2)-len(current_deli):len(x2)]
            del y2[len(y2)-len(current_deli):len(y2)]
            del final_data[len(final_data)-len(current_deli):len(final_data)]
            #print('after_delete_seq:',final_data[len(final_data)-1])
            #print('after_delete_x:',x2[len(x2)-1])
            #print('after_delete_y:',y2[len(y2)-1])

    count_len = 0
    last_age = 0
    is_nonzero = 0
    for i in range(1,len(x2)):
        if(x2[i][9] >  x2[i-1][9]):
            count_len += 1
        else:
            seqlen.append(count_len+1)
            count_len = 0
            if(x2[i][9] !=0):
                #print(x2[i][9])
                is_nonzero = 1
    if(is_nonzero == 0):
        print('ok')

    if(count_len != 0 ):
        seqlen.append(count_len+1)

    #print(x2.shape)
    #print(y2.shape)

    #print(len(x2),len(x2[0]))

    import math
    good = 0
    bad = 0

    #print(len(y2))

    for i in range(len(y2)):
        if(y2[i]==1):
            bad = bad+1
        else:
            good = good + 1

    #print("good: ",good)
    #print("bad: ",bad)
    #print('Real DR:',(bad/(bad+good))*100)
    #print("baseline: ", -(bad/(bad+good))*math.log(bad/(bad+good))-(1-bad/(bad+good))*math.log(1-bad/(bad+good)))

    for i in range(len(x2)):
        #FTHF:N=1,Y=2,9=3
        #OS:P=1,I=2,S=3,9=4
        if(x2[i][1]=='P'):
            x2[i][1]=1
        if(x2[i][1]=='I'):
            x2[i][1]=2
        if(x2[i][1]=='S'):
            x2[i][1]=3   
        #if(x[i][1]=='9'):
            #x[i][1]=4 
        #Channel:R=1,B=2,C=3,T=4,9=5
        #PT:PU=1,SF=2,CO=3,MH&CP&9=4
        if(x2[i][6]=='PU'):
            x2[i][6]=1
        if(x2[i][6]=='SF'):
            x2[i][6]=2
        if(x2[i][6]=='CO'):
            x2[i][6]=3   
        if(x2[i][6]=='MH'):
            x2[i][6]=2 
        if(x2[i][6]=='CP'):
            x2[i][6]=2 
        #if(x[i][8]=='9'):
            #x[i][8]=4 
        #LP:P=1,C=2,N=3,R=4,9=5
        if(x2[i][7]=='P'):
            x2[i][7]=1
        if(x2[i][7]=='C'):
            x2[i][7]=2
        if(x2[i][7]=='N'):
            x2[i][7]=3   
        #if(x[i][9]=='R'):
            #x[i][9]=4 
        #if(x[i][9]=='9'):
            #x[i][9]=5 
        #NB:1=1,2=2,9=3
        if(x2[i][8]==1):
            x2[i][8]=0 
        if(x2[i][8]==2):
            x2[i][8]=1 
        if(x2[i][8]==99):
            x2[i][8]=0.5   

    #最终版本Linear DTSM代码 -- 测试每个季度模型的部分
    from sklearn import preprocessing
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import auc

    x_s=x

    #scaler = preprocessing.StandardScaler().fit(x_s)
    #x = scaler.transform(x)

    x2 = scaler.transform(x2)

    #x_auc = scaler.transform(x_auc)
    #x2_auc = scaler.transform(x2_auc)

    #for i in range(len(x)):
    #    if(x[i][5]==-1.2333916623473458):
    #        count=count+1
    #print(count)
    #scaler2 = preprocessing.StandardScaler().fit(x2)
    #x2 = scaler2.transform(x2)


    x2=ct.transform(x2)


    #print(x[0])
    #print(x2[0])

    #x2 = []
    #print(x2)
    #print(x2[0][0,0])
    #for i in range(x2.shape[0]):
        #if(x2[i][0,0]):
            #print(x2[i])
            #x3.append(x2[i])


    #x_auc = ct.fit_transform(x_auc)

    #x2_auc = ct.fit_transform(x2_auc)


    #model = LogisticRegression(solver='newton-cg',penalty='l2').fit(x, y)

    s = model.predict_proba(x2)
    prediction = []
    for i in range(len(s)):
        prediction.append(s[i][1])
    #print(s[0][1])
    #auc_score = roc_auc_score(y2,prediction)
    #print('AUC: ',auc_score)

    default = 0
    for i in range(len(s)):
        default = default + (1-s[i][0])  


    unbalanced_prediction = []
    for i in range(len(s)):
        unbalanced_prediction.append(0.1*s[i][1]/(0.1*s[i][1] - s[i][1] + 1))

    unbalanced_default = 0
    for i in range(len(unbalanced_prediction)):
        unbalanced_default = unbalanced_default + unbalanced_prediction[i]

    #print('DR: ', default/len(s)*100)
    #print('Unbalanced DR predicted by DTSM: ', unbalanced_default/len(s)*100)

    unbalanced_log_likeli_dtsm = 0
    for i in range(len(unbalanced_prediction)):
        if(y2[i] == 0):
            unbalanced_log_likeli_dtsm = unbalanced_log_likeli_dtsm + math.log(1-unbalanced_prediction[i])
        else:
            unbalanced_log_likeli_dtsm = unbalanced_log_likeli_dtsm + math.log(unbalanced_prediction[i])

    log_likely_dtsm = 0
    for i in range(len(s)):
        if(y2[i] == 0):
            log_likely_dtsm  = log_likely_dtsm  + math.log(s[i][0])
        else:
            log_likely_dtsm  = log_likely_dtsm  + math.log(s[i][1])
    #print('log_likelyhood_dtsm : ',(-log_likely_dtsm )/len(s))

    #print('Unbalanced log_likelyhood_dtsm : ',(-unbalanced_log_likeli_dtsm )/len(s))

    auc_score = roc_auc_score(y2,unbalanced_prediction)
    #print('DTSM AUC: ',auc_score)

    general_auc = auc_score

    #llr ratior test
    llr_test_unbalanced_dtsm = unbalanced_log_likeli_dtsm

    dr = bad/(bad + good)

    baseline_llr = -dr*math.log(dr)-(1-dr)*math.log(1-dr)

    pseudo_dtsm.append(1-((-unbalanced_log_likeli_dtsm)/len(unbalanced_prediction))/baseline_llr)

    #for exposure
    exposure_pd = unbalanced_prediction
    #for time-dependent AUC
    time_auc = unbalanced_prediction
    #for delongtest
    prediction_dtsm = unbalanced_prediction

    #time dependent AUC

    #print(len(time_auc))

    count = 0 
    for i in range(len(seqlen)):
        count = count + seqlen[i]
    #print(count)

    #print('time window 24,36,60==============================')
    #set the time windows:
    time_window = [24,36,60]
    is_default = 0
    duration = []

    AUC = []

    #get the conditional default status based on the current time window
    conditional_labels = []

    predict = []
    prediction = []

    s = np.array(s)

    id = 0
    for m in range(len(time_window)): 
        id = 0
        prediction = [] 
        conditional_labels = []
        is_default = 0
        for i in range(len(seqlen)):
            #if max_age <= time_window
            if(seqlen[i] <= time_window[m]):
                #1
                predict = time_auc[id:id+int(seqlen[i])]
                #prediction.append(predict.mean())
                h0 = predict[0]
                survival = 1-h0
                for j in range(1,len(predict)):
                    survival = survival*(1-predict[j])
                #print('survival1:',survival)
                prediction.append(1-survival)
                #print(x2[id+seqlen[i]-1])
                if(y2[id+int(seqlen[i])-1] == 0):
                    conditional_labels.append(0)
                else:
                    conditional_labels.append(1)
                    is_default = 1

            #max_age > time_window
            else:
                predict = time_auc[id:int(id+time_window[m])]
                h0 = predict[0]
                survival = 1-h0
                for j in range(1,len(predict)):
                    survival = survival*(1-predict[j])
                #print('survival2:',survival)
                prediction.append(1-survival)

                #prediction.append(predict.mean())
                conditional_labels.append(0)
            id = id + int(seqlen[i])
        #AUC
        if(is_default == 1):
            auc_score = roc_auc_score(conditional_labels,prediction)
            AUC.append(auc_score)
            

    #print(conditional_labels)
    AUC = np.array(AUC)    
    
    if(AUC.mean()<0.1):
        AUC = AUC.mean() * 10
        
    AUC24 = AUC

    

    brier_score = 0
    for i in range(len(time_auc)):
        brier_score = brier_score + (time_auc[i]-y2[i])**2
    #print('DTSM Brier Score: ',brier_score/len(time_auc))

    print('GAM...')

    from pygam import LogisticGAM,s,f,l
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
    from sklearn import preprocessing
    import time


    #computational time
    T1 = time.perf_counter()

    if(prs<48):
        path = os.getcwd()
        new_path = path.replace("\\","/")
        f2 = open(new_path + "/Replication_IJF/data/2259/train_all_dtsm_x_forecast_Deli_random_debug0.txt")
    else:
        path = os.getcwd()
        new_path = path.replace("\\","/")
        f2 = open(new_path + "/Replication_IJF/data/2259/train16to21_all_dtsm_x_forecast_Deli_random_debug0.txt")

    vt = preprocessing.LabelEncoder()
    if(prs<48):
        vt.fit([200401,200402,200403,200404,200501,200502,200503,200504,200601,200602,200603,200604,200701,200702,200703,200704,200801,
           200802,200803,200804,200901,200902,200903,200904,201001,201002,201003,201004,201101,201102,201103,201104,201201,201202,201203,
           201204,201301,201302,201303,201304,201401,201402,201403,201404,201501,201502,201503,201504])
    else:
        vt.fit([201601,201602,201603,201604,201701,201702,201703,201704,201801,
           201802,201803,201804,201901,201902,201903,201904,202001,202002,202003,202004,202101,202102,202103,202104,202201,202202,202203,
           202204,202301,202302,202303,202304,202401,202402,202403])


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
    test_HPI = []
    test_UNRATE = []
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
        test_vintage.append(vt.transform([int(line_data[13])])[0]-1);
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
        b2.append(test_OS[i])
        b2.append(test_DTI[i])
        b2.append(test_UPB[i])
        b2.append(test_LTV[i])
        b2.append(test_IR[i])
        #b2.append(test_Channel[i])
        b2.append(test_PT[i])
        b2.append(test_LP[i])
        b2.append(test_NB[i])
        #b2.append(test_HPI[i])
        #b2.append(test_UNRATE[i])
        b2.append(test_Age[i])
        #b2.append(test_vintage[i])
        #b2.append(test_Deli[i])
        #b2.append(test_calendar[i])
        if(int(test_Age[i])<=2):
            b2.append(0)
        else:
            s_p3 = test_data[i-3].split(' ')
            b2.append(int(s_p3[16]))
        current_deli.append(int(line_data[16]))
        final_data.append(line_data)
        #b2.append(test_Deli[i])
        y.append(test_Def[i])
        x.append(b2)


        if(str(new_seq) == str(last_seq) and test_Age[i] > max_seqlen):
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
            print('Exception1!')
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
            print('Exception2!')
        if(str(new_seq) != str(last_seq)):
            if((max_seqlen+1 != (len(current_deli)-1)) or (max_seqlen == 0)):

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
            s_test = test_data[i].split(' ')
            slen = len(s_test)
            current_deli.append(int(s_test[slen-1])) 

        last_seq = new_seq        
        b2=[]

    if(max_seqlen != 0):
        if(max_seqlen+1 != len(current_deli)):
            #print('last one')
            #print('max_seqlen+1:',max_seqlen+1)
            #print('current_deli:',len(current_deli)-1)
            #print('current_seq:',final_data[len(final_data)-1])
            #print('current_x:',x[len(x)-1])
            #print('current_y:',y[len(y)-1])

            #important!!!, cause to wrong training: need to delete the from the last index, not the current index
            del x[len(x)-len(current_deli):len(x)]
            del y[len(y)-len(current_deli):len(y)]
            del final_data[len(final_data)-len(current_deli):len(final_data)]
            #print('after_delete_seq:',final_data[len(final_data)-1])
            #print('after_delete_x:',x[len(x)-1])
            #print('after_delete_y:',y[len(y)-1])


    #print(x2.shape)
    #print(y2.shape)



    #print(len(x),len(x[0]))

    #最终版本Linear DTSM代码 -- 给类别变量编码的部分
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #For delinquency version
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    for i in range(len(x)):
        #FTHF:N=1,Y=2,9=3
        #OS:P=1,I=2,S=3,9=4
        if(x[i][1]=='P'):
            x[i][1]=1
        if(x[i][1]=='I'):
            x[i][1]=2
        if(x[i][1]=='S'):
            x[i][1]=3   
        #if(x[i][2]=='9'):
            #x[i][2]=4 
        #Channel:R=1,B=2,C=3,T=4,9=5
        #PT:PU=1,SF=2,CO=3,MH&CP&9=4
        if(x[i][6]=='PU'):
            x[i][6]=1
        if(x[i][6]=='SF'):
            x[i][6]=2
        if(x[i][6]=='CO'):
            x[i][6]=3   
        if(x[i][6]=='MH'):
            x[i][6]=2 
        if(x[i][6]=='CP'):
            x[i][6]=2 
        #if(x[i][8]=='9'):
            #x[i][8]=4 
        #LP:P=1,C=2,N=3,R=4,9=5
        if(x[i][7]=='P'):
            x[i][7]=1
        if(x[i][7]=='C'):
            x[i][7]=2
        if(x[i][7]=='N'):
            x[i][7]=3   
        #if(x[i][9]=='R'):
            #x[i][9]=4 
        #if(x[i][9]=='9'):
            #x[i][9]=5 
        #NB:1=1,2=2,9=3
        if(x[i][8]==1):
            x[i][8]=0 
        if(x[i][8]==2):
            x[i][8]=1 
        if(x[i][8]==99):
            x[i][8]=0.5   
        #Delinquency:0=0,1=1,2=2,3=3
        if(x[i][10]==0):
            x[i][10]=0 
        if(x[i][10]==1):
            x[i][10]=1 
        if(x[i][10]==2):
            x[i][10]=2 
        if(x[i][10]==3):
            x[i][10]=3 

    #print(x[0])


    x = np.array(x)
    y = np.array(y)
    #model = LogisticRegression(solver='newton-cg',penalty='l2',C=0.1).fit(x, y)
    #gam = LogisticGAM(f(1)+f(6)+f(7)+s(11)+f(12),fit_intercept = True).fit(x,y)
    #gam = LogisticGAM(f(1)+f(6)+f(7)+s(11)+f(12),fit_intercept = True)

    # Fit the model to the data
    #objective = 'AIC'
    #gam.gridsearch(x, y,lam=np.logspace(0, 2, 10),n_splines=np.arange(10,35,5),spline_order=[2,3,4])
    #gam.gridsearch(x, y,lam=np.logspace(0, 1, 2),n_splines=np.arange(10,35,5),spline_order=[2,3])
    #gam = LogisticGAM(f(1)+f(6)+f(7)+s(11)+f(12),fit_intercept = True)
    #gam.gridsearch(x, y,lam=np.logspace(0, 1, 2),n_splines=np.arange(20,45,5),spline_order=[2,3])
    #gam.gridsearch(x, y,lam=np.logspace(0, 1, 2))

    #print('best:',gam.terms[0].lam,gam.terms[0].n_splines,gam.terms[0].spline_order)
    #print('best:',gam.terms[1].lam,gam.terms[1].n_splines,gam.terms[1].spline_order)
    #print('best:',gam.terms[2].lam,gam.terms[2].n_splines,gam.terms[2].spline_order)
    #print('best:',gam.terms[3].lam,gam.terms[3].n_splines,gam.terms[3].spline_order)
    #print('best:',gam.terms[4].lam,gam.terms[4].n_splines,gam.terms[4].spline_order)

    gam = LogisticGAM(l(0)+f(1)+l(2)+l(3)+l(4)+l(5)+f(6)+f(7)+l(8)+s(9,n_splines=25,spline_order = 2)+f(10),lam = 1,fit_intercept = True).fit(x,y)

    T2 = time.perf_counter()
    #print('Forecasting Time:' , ((T2 - T1)))

    #gam.summary()

    #predictions = gam.predict(x2)

    if(int(ym%100)<10):
        path = os.getcwd()
        new_path = path.replace("\\","/")
        TEST_FILE = new_path + "/Replication_IJF/data/2259/" + "test" + "0" + str(int(ym%100)) + "_" + str(q) + "_unbalanced_dtsm_x_Deli.txt"
    else:
        path = os.getcwd()
        new_path = path.replace("\\","/")
        TEST_FILE = new_path + "/Replication_IJF/data/2259/" + "test" + str(int(ym%100)) + "_" + str(q) + "_unbalanced_dtsm_x_Deli.txt"
    f2 = open(TEST_FILE)
    vintage = 200401
    #f2 = open("./test4_data.txt")
    #f2 = open("./test3.txt")
    #train_data = f1.readline()
    #test_data = f2.readline()

    #test_data = f2.readline()
    test_data = []
    for line in f2:
        test_data.append(line)


    test_credit = []
    test_DTI = []
    test_UPB = []
    test_LTV = []
    test_IR = []
    test_HPI = []
    test_UNRATE = []
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
    seqlen = []



    x2 = []
    y2 = []
    b2 = []

    last_seq = ''
    new_seq = ''
    last_default = 0
    new_default = 0
    is_default = 0
    max_seqlen = 0
    count_len = 0

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
        #test_HPI.append(float(line_data[12]))
        #test_UNRATE.append(float(line_data[13]))
        test_Age.append(int(line_data[12]));
        test_vintage.append(vt.transform([int(line_data[13])])[0]);
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
        b2.append(test_OS[i])
        b2.append(test_DTI[i])
        b2.append(test_UPB[i])
        b2.append(test_LTV[i])
        b2.append(test_IR[i])
        #b2.append(test_Channel[i])
        b2.append(test_PT[i])
        b2.append(test_LP[i])
        b2.append(test_NB[i])
        #b2.append(test_HPI[i])
        #b2.append(test_UNRATE[i])
        b2.append(test_Age[i])
        #b2.append(test_vintage[i])
        #b2.append(test_calendar[i])
        #b2.append(test_SEQ[i]) just for test if the result excluded the exception
        if(int(test_Age[i])<=2):
            b2.append(0)
        else:
            s_p3 = test_data[i-3].split(' ')
            b2.append(int(s_p3[16]))
        #b2.append(test_Deli[i])
        current_deli.append(int(line_data[16]))
        final_data.append(line_data)
        y2.append(test_Def[i])
        x2.append(b2)



        if(str(new_seq) == str(last_seq) and test_Age[i] > max_seqlen):
            max_seqlen = test_Age[i]

        if(test_Def[i] == 1):
            is_default = 1
        if(len(last_seq) ==  0):
            last_seq = test_SEQ[0] 
        if(str(new_seq) == str(last_seq) and is_default == 0 and int(test_Age[i]) < max_seqlen ):
            del x2[len(x2)-1]
            del y2[len(y2)-1]
            del current_deli[len(current_deli)-1]
            #print('last one: ',final_data[len(final_data)-2])
            #print('delete current: ',final_data[len(final_data)-1])
            del final_data[len(final_data)-1]

            #print('Exception1!')
        if(str(new_seq) == str(last_seq) and is_default == 1 and test_Def[i] == 0):
            del x2[len(x2)-1]
            del y2[len(y2)-1]
            del current_deli[len(current_deli)-1]
            #print('last one: ',final_data[len(final_data)-2])
            #print('delete current: ',final_data[len(final_data)-1])

            #print('after delete: ',final_data[len(final_data)-1])
            #print('after delete x: ',x2[len(x2)-1])
            #print('after delete y:',y2[len(y2)-1])
            #print('Exception2!')
        if(str(new_seq) != str(last_seq)):
            if((max_seqlen+1 != (len(current_deli)-1)) or (max_seqlen == 0)):

                #print('max_seqlen+1:',max_seqlen+1)
                #print('current_deli:',len(current_deli)-1)
                #print('current_seq:',final_data[len(final_data)-2])
                #print('current_x:',x2[len(x2)-2])
                #print('current_y:',y2[len(y2)-2])

                #important!!!, cause to wrong training: need to delete the from the last index, not the current index
                del x2[len(x2)-len(current_deli):len(x2)-1]
                del y2[len(y2)-len(current_deli):len(y2)-1]
                del final_data[len(final_data)-len(current_deli):len(final_data)-1]

                #print('after_delete_seq:',final_data[len(final_data)-2])
                #print('after_delete_x:',x2[len(x2)-2])
                #print('after_delete_y:',y2[len(y2)-2])



            is_default = 0
            max_seqlen = 0 
            current_deli = []
            count_len = 1
            #cleaned but need to return this current data(current data belongs to the new seq)
            s = test_data[i].split(' ')
            slen = len(s)
            current_deli.append(int(s[slen-1])) 


        last_seq = new_seq        
        b2=[]

    if(max_seqlen != 0):
        if(max_seqlen+1 != len(current_deli)):

            #print('max_seqlen+1:',max_seqlen+1)
            #print('current_deli:',len(current_deli)-1)
            #print('current_seq:',final_data[len(final_data)-1])
            #print('current_x:',x2[len(x2)-1])
            #print('current_y:',y2[len(y2)-1])

            #important!!!, cause to wrong training: need to delete the from the last index, not the current index
            del x2[len(x2)-len(current_deli):len(x2)]
            del y2[len(y2)-len(current_deli):len(y2)]
            del final_data[len(final_data)-len(current_deli):len(final_data)]
            #print('after_delete_seq:',final_data[len(final_data)-1])
            #print('after_delete_x:',x2[len(x2)-1])
            #print('after_delete_y:',y2[len(y2)-1])

    count_len = 0
    last_age = 0
    is_nonzero = 0
    for i in range(1,len(x2)):
        if(x2[i][9] >  x2[i-1][9]):
            count_len += 1
        else:
            seqlen.append(count_len+1)
            count_len = 0
            if(x2[i][9] !=0):
                #print(x2[i][9])
                is_nonzero = 1
    if(is_nonzero == 0):
        print('ok')

    if(count_len != 0 ):
        seqlen.append(count_len+1)

    #print(x2.shape)
    #print(y2.shape)

    #print(len(x2),len(x2[0]))

    import math
    good = 0
    bad = 0

    #print(len(y2))

    for i in range(len(y2)):
        if(y2[i]==1):
            bad = bad+1
        else:
            good = good + 1

    #print("good: ",good)
    #print("bad: ",bad)
    #print('Real DR:',(bad/(bad+good))*100)
    #print("baseline: ", -(bad/(bad+good))*math.log(bad/(bad+good))-(1-bad/(bad+good))*math.log(1-bad/(bad+good)))

    for i in range(len(x2)):
        #FTHF:N=1,Y=2,9=3
        #OS:P=1,I=2,S=3,9=4
        if(x2[i][1]=='P'):
            x2[i][1]=1
        if(x2[i][1]=='I'):
            x2[i][1]=2
        if(x2[i][1]=='S'):
            x2[i][1]=3   
        #if(x[i][1]=='9'):
            #x[i][1]=4 
        #Channel:R=1,B=2,C=3,T=4,9=5
        #PT:PU=1,SF=2,CO=3,MH&CP&9=4
        if(x2[i][6]=='PU'):
            x2[i][6]=1
        if(x2[i][6]=='SF'):
            x2[i][6]=2
        if(x2[i][6]=='CO'):
            x2[i][6]=3   
        if(x2[i][6]=='MH'):
            x2[i][6]=2 
        if(x2[i][6]=='CP'):
            x2[i][6]=2 
        #if(x[i][8]=='9'):
            #x[i][8]=4 
        #LP:P=1,C=2,N=3,R=4,9=5
        if(x2[i][7]=='P'):
            x2[i][7]=1
        if(x2[i][7]=='C'):
            x2[i][7]=2
        if(x2[i][7]=='N'):
            x2[i][7]=3   
        #if(x[i][9]=='R'):
            #x[i][9]=4 
        #if(x[i][9]=='9'):
            #x[i][9]=5 
        #NB:1=1,2=2,9=3
        if(x2[i][8]==1):
            x2[i][8]=0 
        if(x2[i][8]==2):
            x2[i][8]=1 
        if(x2[i][8]==99):
            x2[i][8]=0.5   

    #最终版本Linear DTSM代码 -- 测试每个季度模型的部分
    from sklearn import preprocessing
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import auc

    #x_s=x


    #x2 = scaler.transform(x2)




    #x2=ct.transform(x2)


    predict_gam = gam.predict_proba(x2)
    #print(predict_gam[0])
    prediction = []
    for i in range(len(predict_gam)):
        prediction.append(predict_gam[i])

    #auc_score = roc_auc_score(y2,prediction)
    #print('AUC: ',auc_score)

    default = 0
    for i in range(len(predict_gam)):
        default = default + (predict_gam[i])  


    unbalanced_prediction = []
    for i in range(len(predict_gam)):
        unbalanced_prediction.append(0.1*predict_gam[i]/(0.1*predict_gam[i] - predict_gam[i] + 1))

    unbalanced_default = 0
    for i in range(len(unbalanced_prediction)):
        unbalanced_default = unbalanced_default + unbalanced_prediction[i]

    #print('DR: ', default/len(predict_gam)*100)
    #print('Unbalanced DR predicted by DTSM: ', unbalanced_default/len(predict_gam)*100)

    unbalanced_log_likeli_dtsm = 0
    for i in range(len(unbalanced_prediction)):
        if(y2[i] == 0):
            unbalanced_log_likeli_dtsm = unbalanced_log_likeli_dtsm + math.log(1-unbalanced_prediction[i])
        else:
            unbalanced_log_likeli_dtsm = unbalanced_log_likeli_dtsm + math.log(unbalanced_prediction[i])

    log_likely_dtsm = 0
    for i in range(len(predict_gam)):
        if(y2[i] == 0):
            log_likely_dtsm  = log_likely_dtsm  + math.log(1-predict_gam[i])
        else:
            log_likely_dtsm  = log_likely_dtsm  + math.log(predict_gam[i])
    #print('log_likelyhood_dtsm : ',(-log_likely_dtsm )/len(predict_gam))

    #print('Unbalanced log_likelyhood_dtsm : ',(-unbalanced_log_likeli_dtsm )/len(predict_gam))

    auc_score = roc_auc_score(y2,unbalanced_prediction)
    if(auc_score<0.5):
        auc_score = 1 - auc_score
    #print('DTSM AUC: ',auc_score)

    T3 = time.perf_counter()
    #print('Forecasting Time:' , ((T3 - T2)))

    #dtsm+in_time
    general_auc = auc_score

    #llr ratior test
    llr_test_unbalanced_dtsm = unbalanced_log_likeli_dtsm

    dr = bad/(bad + good)

    baseline_llr = -dr*math.log(dr)-(1-dr)*math.log(1-dr)

    pseudo_gam.append(1-((-unbalanced_log_likeli_dtsm)/len(unbalanced_prediction))/baseline_llr)

    #for exposure
    exposure_pd = unbalanced_prediction
    #for time-dependent AUC
    time_auc = unbalanced_prediction
    #for delongtest
    prediction_dtsm = unbalanced_prediction

    #time dependent AUC

    #print(len(time_auc))

    count = 0 
    for i in range(len(seqlen)):
        count = count + seqlen[i]
    #print(count)

    #print('time window 24,36,60==============================')
    #set the time windows:
    time_window = [24,36,60]
    is_default = 0
    duration = []

    AUC = []

    #get the conditional default status based on the current time window
    conditional_labels = []

    predict = []
    prediction = []

    s = np.array(s)

    id = 0
    for m in range(len(time_window)): 
        id = 0
        prediction = [] 
        conditional_labels = []
        is_default = 0
        for i in range(len(seqlen)):
            #if max_age <= time_window
            if(seqlen[i] <= time_window[m]):
                #1
                predict = time_auc[id:id+int(seqlen[i])]
                #prediction.append(predict.mean())
                h0 = predict[0]
                survival = 1-h0
                for j in range(1,len(predict)):
                    survival = survival*(1-predict[j])
                #print('survival1:',survival)
                prediction.append(1-survival)
                #print(x2[id+seqlen[i]-1])
                if(y2[id+int(seqlen[i])-1] == 0):
                    conditional_labels.append(0)
                else:
                    conditional_labels.append(1)
                    is_default = 1

            #max_age > time_window
            else:
                predict = time_auc[id:int(id+time_window[m])]
                h0 = predict[0]
                survival = 1-h0
                for j in range(1,len(predict)):
                    survival = survival*(1-predict[j])
                #print('survival2:',survival)
                prediction.append(1-survival)

                #prediction.append(predict.mean())
                conditional_labels.append(0)
            id = id + int(seqlen[i])
        #AUC
        if(is_default == 1):
            auc_score = roc_auc_score(conditional_labels,prediction)
            if(auc_score < 0.5):
                auc_score = 1- auc_score
            AUC.append(auc_score)
            #print('DTSM AUC ',time_window[m],':',auc_score)

    #print(conditional_labels)
    AUC = np.array(AUC)    
    #print('DTSM avg_AUC: ', AUC.mean())

    AUC24 = AUC


    brier_score = 0
    for i in range(len(time_auc)):
        brier_score = brier_score + (time_auc[i]-y2[i])**2
    #print('DTSM Brier Score: ',brier_score/len(time_auc))

    #whether to open the calculation of exposure
    is_exposure_cph = True
    is_exposure_wb = True

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
    #import pandas as pd
    #import lifelines
    #from lifelines import CoxPHFitter
    #Linear DTSM + delinquency

    print('Cox PH & Weibell ...')
    is_exposure = False
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
    import pandas as pd


    #data for forecast
    if(prs<48):
        path = os.getcwd()
        new_path = path.replace("\\","/")
        f2 = open(new_path + "/Replication_IJF/data/2259/continuous_survival_train_account.txt")
    else:    
        path = os.getcwd()
        new_path = path.replace("\\","/")
        f2 = open(new_path + "/Replication_IJF/data/2259/continuous16to21_survival_train_account.txt")

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
        #test_vintage.append(int(line_data[13]));
        #test_calendar.append((int(line_data[14])));
        test_Def.append(int(line_data[14]));
        #test_Deli.append(int(line_data[16]));



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
        b2.append(test_OS[i])
        b2.append(test_DTI[i])
        b2.append(test_UPB[i])
        b2.append(test_LTV[i])
        b2.append(test_IR[i])
        #b2.append(test_Channel[i])
        b2.append(test_PT[i])
        b2.append(test_LP[i])
        b2.append(test_NB[i])
        b2.append(test_Age[i])
        #b2.append(test_vintage[i])


        #b2.append(test_Deli[i])
        b2.append(test_Def[i])
        x.append(b2)

        b2 = []

    #print(len(x),len(x[0]))
    #print(x[0])


    if(prs<48):
        path = os.getcwd()
        new_path = path.replace("\\","/")
        f2 = open(new_path + "/Replication_IJF/data/2259/continuous_survival_test_account.txt")
    else:
        path = os.getcwd()
        new_path = path.replace("\\","/")
        f2 = open(new_path + "/Replication_IJF/data/2259/continuous16to21_survival_test_account.txt")
    #f2 = open("./test4_data.txt")
    #f2 = open("./test3.txt")
    #train_data = f1.readline()
    #test_data = f2.readline()

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



    x2 = []
    y2 = []
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
        #test_vintage.append(int(line_data[13]));
        #test_calendar.append((int(line_data[14])));
        test_Def.append(int(line_data[14]));
        #test_Deli.append(int(line_data[16]));


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
        b2.append(test_OS[i])
        b2.append(test_DTI[i])
        b2.append(test_UPB[i])
        b2.append(test_LTV[i])
        b2.append(test_IR[i])
        #b2.append(test_Channel[i])
        b2.append(test_PT[i])
        b2.append(test_LP[i])
        b2.append(test_NB[i])
        b2.append(test_Age[i])
        #b2.append(test_vintage[i])
        #b2.append(test_calendar[i])
        #b2.append(test_SEQ[i]) just for test if the result excluded the exception
        #if(int(test_Age[i])<=2):
            #b2.append(0)
        #else:
            #s_p3 = test_data[i-3].split(' ')
            #b2.append(int(s_p3[16]))
        #current_deli.append(int(line_data[16]))
        #final_data.append(line_data)
        b2.append(test_Def[i])
        x2.append(b2)


        b2=[]



    #print(len(x2),len(x2[0]))
    #print(x2[0])

    #最终版本Linear DTSM代码 -- 给类别变量编码的部分
    for i in range(len(x)):
        #FTHF:N=1,Y=2,9=3
        #OS:P=1,I=2,S=3,9=4
        if(x[i][1]=='P'):
            x[i][1]=1
        if(x[i][1]=='I'):
            x[i][1]=2
        if(x[i][1]=='S'):
            x[i][1]=3   
        #if(x[i][2]=='9'):
            #x[i][2]=4 
        #Channel:R=1,B=2,C=3,T=4,9=5
        #PT:PU=1,SF=2,CO=3,MH&CP&9=4
        if(x[i][6]=='PU'):
            x[i][6]=1
        if(x[i][6]=='SF'):
            x[i][6]=2
        if(x[i][6]=='CO'):
            x[i][6]=3   
        if(x[i][6]=='MH'):
            x[i][6]=2 
        if(x[i][6]=='CP'):
            x[i][6]=2 
        #if(x[i][8]=='9'):
            #x[i][8]=4 
        #LP:P=1,C=2,N=3,R=4,9=5
        if(x[i][7]=='P'):
            x[i][7]=1
        if(x[i][7]=='C'):
            x[i][7]=2
        if(x[i][7]=='N'):
            x[i][7]=3   
        #if(x[i][9]=='R'):
            #x[i][9]=4 
        #if(x[i][9]=='9'):
            #x[i][9]=5 
        #NB:1=1,2=2,9=3
        if(x[i][8]==1):
            x[i][8]=0 
        if(x[i][8]==2):
            x[i][8]=1 
        if(x[i][8]==99):
            x[i][8]=0.5   

    for i in range(len(x2)):
        #FTHF:N=1,Y=2,9=3
        #OS:P=1,I=2,S=3,9=4
        if(x2[i][1]=='P'):
            x2[i][1]=1
        if(x2[i][1]=='I'):
            x2[i][1]=2
        if(x2[i][1]=='S'):
            x2[i][1]=3   
        #if(x[i][1]=='9'):
            #x[i][1]=4 
        #Channel:R=1,B=2,C=3,T=4,9=5
        #PT:PU=1,SF=2,CO=3,MH&CP&9=4
        if(x2[i][6]=='PU'):
            x2[i][6]=1
        if(x2[i][6]=='SF'):
            x2[i][6]=2
        if(x2[i][6]=='CO'):
            x2[i][6]=3   
        if(x2[i][6]=='MH'):
            x2[i][6]=2 
        if(x2[i][6]=='CP'):
            x2[i][6]=2 
        #if(x[i][8]=='9'):
            #x[i][8]=4 
        #LP:P=1,C=2,N=3,R=4,9=5
        if(x2[i][7]=='P'):
            x2[i][7]=1
        if(x2[i][7]=='C'):
            x2[i][7]=2
        if(x2[i][7]=='N'):
            x2[i][7]=3   
        #if(x[i][9]=='R'):
            #x[i][9]=4 
        #if(x[i][9]=='9'):
            #x[i][9]=5 
        #NB:1=1,2=2,9=3
        if(x2[i][8]==1):
            x2[i][8]=0 
        if(x2[i][8]==2):
            x2[i][8]=1 
        if(x2[i][8]==99):
            x2[i][8]=0.5   

    df = pd.DataFrame(x)
    df2 = pd.DataFrame(x2)

    #extract application variables for standatization
    df_standard = df
    df_standard2 = df2

    df_standard = df_standard.drop(1,axis = 1)
    df_standard2 = df_standard2.drop(1,axis = 1)

    df_standard = df_standard.drop(6,axis = 1)
    df_standard2 = df_standard2.drop(6,axis = 1)

    df_standard = df_standard.drop(7,axis = 1)
    df_standard2 = df_standard2.drop(7,axis = 1)

    df_standard = df_standard.drop(8,axis = 1)
    df_standard2 = df_standard2.drop(8,axis = 1)

    df_standard = df_standard.drop(9,axis = 1)
    df_standard2 = df_standard2.drop(9,axis = 1)

    df_standard = df_standard.drop(10,axis = 1)
    df_standard2 = df_standard2.drop(10,axis = 1)

    from sklearn import preprocessing

    df_standard_scaler = df_standard

    scaler = preprocessing.StandardScaler().fit(df_standard_scaler)
    df_standard = scaler.transform(df_standard)
    #print(x[0],x[1])

    #用训练集的标准差进行标准化还是用测试集自己的标准差进行标准化？
    #x_s2=x2
    #scaler = preprocessing.StandardScaler().fit(x_s2)
    df_standard2 = scaler.transform(df_standard2)

    df_standard = pd.DataFrame(df_standard)
    df_standard2 = pd.DataFrame(df_standard2)

    import lifelines
    from lifelines import CoxPHFitter

    # Get one hot encoding of columns 1
    one_hot1 = pd.get_dummies(df[1],prefix = 'OS')


    # Get one hot encoding of columns 1
    one_hot_test1 = pd.get_dummies(df2[1],prefix = 'OS')

    # Join the encoded df
    df_standard = df_standard.join(one_hot1)

    # Join the encoded df
    df_standard2 = df_standard2.join(one_hot_test1)

    # Get one hot encoding of columns 6,7
    one_hot2 = pd.get_dummies(df[6],prefix = 'PT')


    one_hot3 = pd.get_dummies(df[7],prefix = 'LP')


    # Get one hot encoding of columns 6,7
    one_hot_test2 = pd.get_dummies(df2[6],prefix = 'PT')

    one_hot_test3 = pd.get_dummies(df2[7],prefix = 'LP')

    # Join the encoded df
    df_standard = df_standard.join(one_hot2)
    df_standard = df_standard.join(one_hot3)

    # Join the encoded df
    df_standard2 = df_standard2.join(one_hot_test2)
    df_standard2 = df_standard2.join(one_hot_test3)

    # Join the NB,age,default
    df_standard = df_standard.join(df[8])
    df_standard = df_standard.join(df[9])
    df_standard = df_standard.join(df[10])

    df_standard2 = df_standard2.join(df2[8])
    df_standard2 = df_standard2.join(df2[9])
    df_standard2 = df_standard2.join(df2[10])

    #注意，因为月份从0开始，所以需要加1
    df_standard[9] = df_standard[9] + 1
    df_standard2[9] = df_standard2[9] + 1

    #df_standard = df_standard.sample(frac=1).reset_index(drop=True)
    #df_standard2 = df_standard2.sample(frac=1).reset_index(drop=True)

    #CoxPH
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(df_standard, 9, 10)
    #cph.print_summary()
    #cph.score(df_standard)

    #Weibell
    from lifelines import WeibullFitter


    wbf = WeibullFitter()
    wbf.fit(df_standard[9], df_standard[10])

    print('Cox PH and Weibell fitted')

    #f1 = open("./train_data.txt")
    if(int(ym%100)<10):
        path = os.getcwd()
        new_path = path.replace("\\","/")
        TEST_FILE = new_path +  "/Replication_IJF/data/2259/" + "test" + "0" + str(int(ym%100)) + "_" + str(q) + "_unbalanced_dtsm_x_Deli.txt"
    else:
        path = os.getcwd()
        new_path = path.replace("\\","/")
        TEST_FILE = new_path + "/Replication_IJF/data/2259/" + "test" + str(int(ym%100)) + "_" + str(q) + "_unbalanced_dtsm_x_Deli.txt"
    f2 = open(TEST_FILE)
    #f2 = open("./test4_data.txt")
    #f2 = open("./test3.txt")
    #train_data = f1.readline()
    #test_data = f2.readline()

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
    seqlen = []



    x2 = []
    y2 = []
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
        b2.append(test_OS[i])
        b2.append(test_DTI[i])
        b2.append(test_UPB[i])
        b2.append(test_LTV[i])
        b2.append(test_IR[i])
        #b2.append(test_Channel[i])
        b2.append(test_PT[i])
        b2.append(test_LP[i])
        b2.append(test_NB[i])
        b2.append(test_SEQ[i]) #just for test if the result excluded the exception
        b2.append(test_Age[i])
        #b2.append(test_vintage[i])
        #b2.append(test_calendar[i])
        current_deli.append(int(line_data[16]))
        final_data.append(line_data)
        y2.append(test_Def[i])
        #def也放入x2中，为后续coxphfitter做准备，更方便
        b2.append(test_Def[i])
        x2.append(b2)


        if(str(new_seq) == str(last_seq) and test_Age[i] > max_seqlen):
            max_seqlen = test_Age[i]

        if(test_Def[i] == 1):
            is_default = 1
        if(len(last_seq) ==  0):
            last_seq = test_SEQ[0] 
        if(str(new_seq) == str(last_seq) and is_default == 0 and int(test_Age[i]) < max_seqlen ):
            del x2[len(x2)-1]
            del y2[len(y2)-1]
            del current_deli[len(current_deli)-1]
            #print('last one: ',final_data[len(final_data)-2])
            #print('delete current: ',final_data[len(final_data)-1])
            del final_data[len(final_data)-1]
            #print('Exception1!')
        if(str(new_seq) == str(last_seq) and is_default == 1 and test_Def[i] == 0):
            del x2[len(x2)-1]
            del y2[len(y2)-1]
            del current_deli[len(current_deli)-1]
            #print('last one: ',final_data[len(final_data)-2])
            #print('delete current: ',final_data[len(final_data)-1])
            del final_data[len(final_data)-1]
            #print('after delete: ',final_data[len(final_data)-1])
            #print('after delete x: ',x2[len(x2)-1])
            #print('after delete y:',y2[len(y2)-1])
            #print('Exception2!')
        if(str(new_seq) != str(last_seq)):
            if((max_seqlen+1 != (len(current_deli)-1)) or (max_seqlen == 0)):

                #print('max_seqlen+1:',max_seqlen+1)
                #print('current_deli:',len(current_deli)-1)
                #print('current_seq:',final_data[len(final_data)-2])
                #print('current_x:',x2[len(x2)-2])
                #print('current_y:',y2[len(y2)-2])

                #important!!!, cause to wrong training: need to delete the from the last index, not the current index
                del x2[len(x2)-len(current_deli):len(x2)-1]
                del y2[len(y2)-len(current_deli):len(y2)-1]
                del final_data[len(final_data)-len(current_deli):len(final_data)-1]
                #print('after_delete_seq:',final_data[len(final_data)-2])
                #print('after_delete_x:',x2[len(x2)-2])
                #print('after_delete_y:',y2[len(y2)-2])
            #else:
                #seqlen.append(max_seqlen+1) 



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
        if(max_seqlen+1 != len(current_deli)):

            #print('max_seqlen+1:',max_seqlen+1)
            #print('current_deli:',len(current_deli)-1)
            #print('current_seq:',final_data[len(final_data)-1])
            #print('current_x:',x2[len(x2)-1])
            #print('current_y:',y2[len(y2)-1])

            #important!!!, cause to wrong training: need to delete the from the last index, not the current index
            del x2[len(x2)-len(current_deli):len(x2)]
            del y2[len(y2)-len(current_deli):len(y2)]
            del final_data[len(final_data)-len(current_deli):len(final_data)]
            #print('after_delete_seq:',final_data[len(final_data)-1])
            #print('after_delete_x:',x2[len(x2)-1])
            #print('after_delete_y:',y2[len(y2)-1])
        #else:
            #seqlen.append(max_seqlen+1) 

    #print(x2.shape)
    #print(y2.shape)
    labels3 = [] #for account level
    labels4 = [] #for observation level
    seqlen = []
    #print(x2[1])
    #print(len(x2),len(x2[0]))

    count_len = 0
    last_age = 0
    is_nonzero = 0
    for i in range(1,len(x2)):
        if(x2[i][10] >  x2[i-1][10]):
            count_len += 1
        else:
            seqlen.append(count_len+1)
            labels3.append(x2[i-1][11])
            #print(x2[i-1][11])
            count_len = 0
            if(x2[i][10] !=0):
                print(x2[i-1][10])
                is_nonzero = 1
    if(is_nonzero == 0):
        print('ok')

    if(count_len != 0 ):
        seqlen.append(count_len+1)
        labels3.append(x2[len(x2)-1][11])
        count_len = 0

    #labels4 for whole event status
    for i in range(len(seqlen)):
        for j in range(seqlen[i]-1):
            labels4.append(0)
        labels4.append(labels3[i])

    import math
    good = 0
    bad = 0

    for i in range(len(y2)):
        if(y2[i]==1):
            bad = bad+1
        else:
            good = good + 1

    #print("good: ",good)
    #print("bad: ",bad)
    #print('DR:',(bad/(bad+good))*100)
    #print("baseline: ", -(bad/(bad+good))*math.log(bad/(bad+good))-(1-bad/(bad+good))*math.log(1-bad/(bad+good)))

    for i in range(len(x2)):
        #FTHF:N=1,Y=2,9=3
        #OS:P=1,I=2,S=3,9=4
        if(x2[i][1]=='P'):
            x2[i][1]=1
        if(x2[i][1]=='I'):
            x2[i][1]=2
        if(x2[i][1]=='S'):
            x2[i][1]=3   
        #if(x[i][1]=='9'):
            #x[i][1]=4 
        #Channel:R=1,B=2,C=3,T=4,9=5
        #PT:PU=1,SF=2,CO=3,MH&CP&9=4
        if(x2[i][6]=='PU'):
            x2[i][6]=1
        if(x2[i][6]=='SF'):
            x2[i][6]=2
        if(x2[i][6]=='CO'):
            x2[i][6]=3   
        if(x2[i][6]=='MH'):
            x2[i][6]=2 
        if(x2[i][6]=='CP'):
            x2[i][6]=2 
        #if(x[i][8]=='9'):
            #x[i][8]=4 
        #LP:P=1,C=2,N=3,R=4,9=5
        if(x2[i][7]=='P'):
            x2[i][7]=1
        if(x2[i][7]=='C'):
            x2[i][7]=2
        if(x2[i][7]=='N'):
            x2[i][7]=3   
        #if(x[i][9]=='R'):
            #x[i][9]=4 
        #if(x[i][9]=='9'):
            #x[i][9]=5 
        #NB:1=1,2=2,9=3
        if(x2[i][8]==1):
            x2[i][8]=0 
        if(x2[i][8]==2):
            x2[i][8]=1 
        if(x2[i][8]==99):
            x2[i][8]=0.5   

    #change the dsicrete time survival data to continuous survival data
    data = []
    data2 = []
    labels = []
    labels2 = []
    #seqlen = []
    load_data = []
    last_seq = ''
    new_seq = ''
    last_default = 0
    new_default = 0
    is_default = 0
    max_seqlen = 0


    COUNT = 0
    max_d = -1

    load_data = x2            
    last_seq = load_data[0][9]

    for i in range(len(load_data)):
        #if COUNT > 10000 and LOAD_LITTLE_DATA:
            #break\n",
        COUNT += 1
        s = load_data[i]
        slen = len(s)

        new_seq = s[9]

        if(int(s[slen-1]) == 0):
            labels2.append(0)

        else:
            labels2.append(1)

        if(int(s[slen-1]) == 1):
            is_default = 1
            #print('is_default')


        if(int(s[slen-2]) > max_seqlen):
            max_seqlen = int(s[slen-2])

        if(len(last_seq) ==  0):
            last_seq = test_SEQ[i] 
        if(str(new_seq) == str(last_seq) and is_default == 0 and int(s[slen-2]) < max_seqlen ):
            del labels2[len(labels2)-1]
            #print('Exception1!')
        if(str(new_seq) == str(last_seq) and is_default == 1 and int(s[slen-1]) == 0):
            #print(load_data[i-1])
            #print(load_data[i])
            #print(labels2[len(labels2)-1])
            del labels2[len(labels2)-1]
            #print(labels2[len(labels2)-1])
            #print('Exception2!')
        if(str(new_seq) != str(last_seq)):
            s = load_data[i-1]
            slen = len(s)
            t_indices = []
            t_indices2 = []
            #max_d = max(td,max_d)
            t_indices.append(float(s[0]))
            t_indices.append(float(s[2]))
            t_indices.append(float(s[3]))
            t_indices.append(float(s[4]))
            t_indices.append(float(s[5]))
            t_indices.append(float(s[8]))
            t_indices.append(int(s[10])+1)

            #t_indices.append(vt.transform([int(s[13])]))

            t_indices2.append(s[1])
            t_indices2.append(s[6])
            t_indices2.append(s[7])
            #t_indices2.append(s[9]) #add seq jus for test
            #t_indices2.append(self.vt.transform([int(s[13])])[0])

            data2.append(t_indices)
            data.append(t_indices2) 

            #default = 1
            if(is_default == 1):
                labels.append(1)
            #default = 0
            else:
                labels.append(0)


            #e.g., s[slen-3]=7,means at 7th quarter, but totally 8 quarters
            #self.seqlen.append(int(s[slen-4])+1)

            #exclude the exception of default = 0 after default = 1 within one account


            #case for solving initial state problem
            #seqlen.append(max_seqlen+1)

            is_default = 0
            max_seqlen = 0

        last_seq = new_seq

    if(max_seqlen != 0):
        s = load_data[len(load_data)-1]
        slen = len(s)
        t_indices = []
        t_indices2 = []
        #max_d = max(td,max_d)
        t_indices.append(float(s[0]))
        t_indices.append(float(s[2]))
        t_indices.append(float(s[3]))
        t_indices.append(float(s[4]))
        t_indices.append(float(s[5]))
        t_indices.append(float(s[8]))
        t_indices.append(float(s[10])+1)

        #t_indices.append(vt.transform([int(s[13])]))

        t_indices2.append(s[1])
        t_indices2.append(s[6])
        t_indices2.append(s[7])
        #t_indices2.append(s[9]) #add seq jus for test
        #t_indices2.append(self.vt.transform([int(s[13])])[0])

        data2.append(t_indices)
        data.append(t_indices2) 

        #default = 1
        if(is_default == 1):
            labels.append(1)
        #default = 0
        else:
            labels.append(0)


        #e.g., s[slen-3]=7,means at 7th quarter, but totally 8 quarters
        #self.seqlen.append(int(s[slen-4])+1)

        #exclude the exception of default = 0 after default = 1 within one account


        #case for solving initial state problem
        #seqlen.append(max_seqlen+1)
        last_seq = new_seq
        is_default = 0
        max_seqlen = 0
        #print('max_seqlen')

    df = pd.DataFrame(data)

    df_standard = pd.DataFrame(data2)

    df_NB = df_standard[5]

    df_age =  df_standard[6]

    df_standard = df_standard.drop(5,axis=1)
    df_standard = df_standard.drop(6,axis=1)

    df_standard = scaler.transform(df_standard)
    #print(x[0],x[1])


    df_standard = pd.DataFrame(df_standard)


    # Get one hot encoding of columns 1
    one_hot1 = pd.get_dummies(df[0],prefix = 'OS')


    # Join the encoded df
    df_standard = df_standard.join(one_hot1)

    # Get one hot encoding of columns 6,7
    one_hot2 = pd.get_dummies(df[1],prefix = 'PT')


    one_hot3 = pd.get_dummies(df[2],prefix = 'LP')


    # Join the encoded df
    df_standard = df_standard.join(one_hot2)
    df_standard = df_standard.join(one_hot3)

    # Join the NB,Age,default
    df_def = pd.DataFrame(labels)
    df_def = df_def.add_prefix('Def_')

    df_standard = df_standard.join(df_NB)
    df_standard = df_standard.join(df_age)
    df_standard = df_standard.join(df_def)

    #rename the column name to keep consistent with training dataset
    df_standard = df_standard.rename(columns = {'Def_0':'Def'})
    df_standard = df_standard.rename(columns = {5:8})
    df_standard = df_standard.rename(columns = {6:9})
    df_standard = df_standard.rename(columns = {'Def':10})

    #df_standard = df_standard.sample(frac=1).reset_index(drop=True)

    #for the non_avg version, that is directly use the designted corresponding final time point
    from sklearn.metrics import auc
    from sklearn.metrics import roc_auc_score
    #time dependent AUC

    unbalanced_prediction_cph = []
    #make the unbalanced PD dtsm

    baseline_hazard = np.array(cph.baseline_hazard_)

    initial_hazard = 0
    for i in range(len(seqlen)):
        pd_balanced = np.float(initial_hazard)*np.float(cph.predict_partial_hazard(df_standard.iloc[i]))
        unbalanced_prediction_cph.append(0.1*pd_balanced/(0.1*pd_balanced - pd_balanced + 1))
        for j in range(seqlen[i]-1):
            s = j
            if(s >= 187):
                s = 187

            pd_balanced = np.float(baseline_hazard[s])*np.float(cph.predict_partial_hazard(df_standard.iloc[i]))
            unbalanced_prediction_cph.append(0.1*pd_balanced/(0.1*pd_balanced - pd_balanced + 1))

    unbalanced_default = 0
    for i in range(len(unbalanced_prediction_cph)):
        unbalanced_default = unbalanced_default + unbalanced_prediction_cph[i]
    #print('Unbalanced DR predicted by CPH: ', unbalanced_default/len(unbalanced_prediction_cph)*100)
    unbalanced_cph_default = unbalanced_default/len(unbalanced_prediction_cph)*100

    unbalanced_log_likeli_cph = 0
    for i in range(len(unbalanced_prediction_cph)):
        if(y2[i] == 0):
            unbalanced_log_likeli_cph = unbalanced_log_likeli_cph + math.log(1-unbalanced_prediction_cph[i])
        else:
            if(unbalanced_prediction_cph[i] == 0):
                1
            else:
                unbalanced_log_likeli_cph = unbalanced_log_likeli_cph + math.log(unbalanced_prediction_cph[i])

    #print('Unbalanced log_likelyhood_dtsm : ',(-unbalanced_log_likeli_cph )/len(unbalanced_prediction_cph))
    unbalanced_cph_log_likeli = (-unbalanced_log_likeli_cph )/len(unbalanced_prediction_cph)

    #llr ratior test
    llr_unbalanced_test_cph = unbalanced_log_likeli_cph

    dr = bad/(bad + good)

    baseline_llr = -dr*math.log(dr)-(1-dr)*math.log(1-dr)

    pseudo_cox.append(1-((-unbalanced_log_likeli_cph)/len(unbalanced_prediction_cph))/baseline_llr)

    auc_score = roc_auc_score(y2,unbalanced_prediction_cph)
    #print('Cox PH AUC: ',auc_score)
    cph_auc = auc_score

    #print('Cox PH============================================')
    #print('time windoe 24,36,60=============================')
    #set the time windows:
    time_window = [24,36,60]
    is_default = 0
    duration = []

    AUC = []

    #get the conditional default status based on the current time window
    conditional_labels = []

    prediction = []

    id = 0

    for m in range(len(time_window)):
        prediction = []
        conditional_labels = []
        id = 0
        is_default = 0
        for i in range(len(seqlen)):
            #if max_age <= time_window

            if(seqlen[i] <= time_window[m]):
                #1
                predict = unbalanced_prediction_cph[id:id+np.int(seqlen[i])]
                #prediction.append(predict.mean())
                h0 = predict[0]
                survival = 1-h0
                for j in range(1,len(predict)):
                    survival = survival*(1-predict[j])
                #print('survival1:',survival)
                prediction.append(survival)
                #predict = np.array(cph.predict_survival_function(df_standard.iloc[i],time_window[m]))

                #prediction.append(1-np.float(predict))
                #print(x2[id+seqlen[i]-1])
                #conditional_labels.append(x2[id+seqlen[i]-1][11])
                conditional_labels.append(labels3[i])
                #if(x2[id+seqlen[i]-1][11] == 1):
                    #is_default = 1
                if(labels3[i] == 1):
                    is_default = 1
            #max_age > time_window
            else:
                predict = unbalanced_prediction_cph[id:id+np.int(seqlen[i])]
                #prediction.append(predict.mean())
                h0 = predict[0]
                survival = 1-h0
                for j in range(1,len(predict)):
                    survival = survival*(1-predict[j])
                #print('survival1:',survival)
                prediction.append(survival)
                #predict = np.array(cph.predict_survival_function(df_standard.iloc[i],time_window[m]))
                #prediction.append(1-np.float(predict))

                conditional_labels.append(0)
                #print(x2[id+time_window[m]-1][11])
                #conditional_labels.append(0)
            id = id + seqlen[i]
        if(is_default == 1):
            auc_score = roc_auc_score(conditional_labels,prediction)
            if(auc_score<0.5):
                auc_score = 1 - auc_score
            #print('time_window:',time_window[m],': ',auc_score)
            AUC.append(auc_score)

    #print(conditional_labels)

    #AUC
    AUC = np.array(AUC)
    
    cph_auc24 = AUC


    brier_score = 0
    for i in range(len(unbalanced_prediction_cph)):
        brier_score = brier_score + (unbalanced_prediction_cph[i]-labels4[i])**2
    #print('Cox PH brier score: ',brier_score/len(unbalanced_prediction_cph))
    cph_brier = brier_score/len(unbalanced_prediction_cph)


    print('Weibell...')

    unbalanced_prediction_wb = []
    #make the unbalanced PD

    for i in range(len(seqlen)):
        for j in range(seqlen[i]):
            pd_balanced = (wbf.rho_/wbf.lambda_)*((j)/wbf.lambda_)**((wbf.rho_)-1)
            unbalanced_prediction_wb.append(0.1*pd_balanced/(0.1*pd_balanced - pd_balanced + 1))

    unbalanced_default = 0
    for i in range(len(unbalanced_prediction_wb)):
        unbalanced_default = unbalanced_default + unbalanced_prediction_wb[i]
    #print('Unbalanced DR predicted by Weibull: ', unbalanced_default/len(unbalanced_prediction_wb)*100)
    unbalanced_wb_default = unbalanced_default/len(unbalanced_prediction_wb)*100

    unbalanced_log_likeli_wb = 0
    for i in range(len(unbalanced_prediction_wb)):
        if(y2[i] == 0):
            unbalanced_log_likeli_wb = unbalanced_log_likeli_wb + math.log(1-unbalanced_prediction_wb[i])
        else:
            unbalanced_log_likeli_wb = unbalanced_log_likeli_wb + math.log(unbalanced_prediction_wb[i])

    #print('Unbalanced log_likelyhood_wb : ',(-unbalanced_log_likeli_wb )/len(unbalanced_prediction_wb))
    unbalanced_wb_log_likeli = (-unbalanced_log_likeli_wb )/len(unbalanced_prediction_wb)

    #llr ratior test
    llr_unbalanced_test_wb = unbalanced_log_likeli_wb

    dr = bad/(bad + good)

    baseline_llr = -dr*math.log(dr)-(1-dr)*math.log(1-dr)

    pseudo_weibull.append(1-((-unbalanced_log_likeli_wb)/len(unbalanced_prediction_wb))/baseline_llr)

    auc_score = roc_auc_score(y2,unbalanced_prediction_wb)
    #print('Weibull AUC: ',auc_score)
    wb_auc = auc_score

    #print('time window 24,36,60=============================')
    #set the time windows:
    time_window = [24,36,60]
    is_default = 0
    duration = []

    AUC = []

    #get the conditional default status based on the current time window
    conditional_labels = []

    prediction = []

    id = 0

    for m in range(len(time_window)):
        prediction = []
        conditional_labels = []
        id = 0
        is_default = 0
        for i in range(len(seqlen)):
            #if max_age <= time_window

            if(seqlen[i] <= time_window[m]):
                #1
                predict = unbalanced_prediction_wb[id:id+np.int(seqlen[i])]
                #prediction.append(predict.mean())
                h0 = predict[0]
                survival = 1-h0
                for j in range(1,len(predict)):
                    survival = survival*(1-predict[j])
                #print('survival1:',survival)
                prediction.append(survival)
                #predict = np.array(cph.predict_survival_function(df_standard.iloc[i],time_window[m]))

                #prediction.append(1-np.float(predict))
                #print(x2[id+seqlen[i]-1])
                #conditional_labels.append(x2[id+seqlen[i]-1][11])
                conditional_labels.append(labels3[i])
                #if(x2[id+seqlen[i]-1][11] == 1):
                    #is_default = 1
                if(labels3[i] == 1):
                    is_default = 1
            #max_age > time_window
            else:
                predict = unbalanced_prediction_wb[id:id+np.int(seqlen[i])]
                #prediction.append(predict.mean())
                h0 = predict[0]
                survival = 1-h0
                for j in range(1,len(predict)):
                    survival = survival*(1-predict[j])
                #print('survival1:',survival)
                prediction.append(survival)
                #predict = np.array(cph.predict_survival_function(df_standard.iloc[i],time_window[m]))
                #prediction.append(1-np.float(predict))

                conditional_labels.append(0)
                #print(x2[id+time_window[m]-1][11])
                #conditional_labels.append(0)
            id = id + seqlen[i]
        if(is_default == 1):
            auc_score = roc_auc_score(conditional_labels,prediction)
            if(auc_score<0.5):
                auc_score = 1 - auc_score
            
            AUC.append(auc_score)

    #print(conditional_labels)

    #AUC
    AUC = np.array(AUC)
    
    wb_auc24 = AUC

    brier_score = 0
    for i in range(len(unbalanced_prediction_cph)):
        brier_score = brier_score + (unbalanced_prediction_wb[i]-labels4[i])**2
    
    wb_brier = brier_score/len(unbalanced_prediction_wb)


    print("DeepHit...")

    #DeepHit
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
    from tensorflow.keras.regularizers import l2

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
            return self.data[self.index], self.seqlen[self.index], self.labels[self.index], self.data2[self.index], self.seq[self.index]

        def __init__(self, INPUT_FILE, win, all,discount,fixed_batch_size):
            self.data = []
            self.data2 = []
            self.labels = []
            self.labels2 = []
            self.deli = []
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
            #print(len(self.load_data))    
            last_seq = self.load_data[0].split(' ')[11]


            for i in range(len(self.load_data)):
                #if COUNT > 10000 and LOAD_LITTLE_DATA:
                    #break\n",
                COUNT += 1
                s = self.load_data[i].split(' ')
                slen = len(s)

                #delinquency,lag 3 months
                if(int(s[slen-5])<=2):
                    self.deli.append(0)
                else:
                    s_p3 = self.load_data[i-3].split(' ')		
                    self.deli.append(int(s_p3[slen-1]))

                current_deli.append(int(s[slen-1]))
                self.final_data.append(s)


                new_seq = s[11] 
                if(int(s[slen-2]) == 0):
                    self.labels2.append([1., 0.])

                else:
                    self.labels2.append([0., 1.])

                if(int(s[slen-2]) == 1):
                    is_default = 1


                if(int(s[slen-5]) > max_seqlen):
                    max_seqlen = int(s[slen-5])

                if(len(last_seq) ==  0):
                    last_seq = s[11] 
                    #print('fist seqqqqqqqqqq!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!',last_seq)
                if(str(new_seq) == str(last_seq) and is_default == 0 and int(s[slen-5]) < max_seqlen ):
                    del self.labels2[len(self.labels2)-1]
                    del self.deli[len(self.deli)-1]
                    del current_deli[len(current_deli)-1]
                    del self.final_data[len(self.final_data)-1]
                    #print('Exception1!')
                if(str(new_seq) == str(last_seq) and is_default == 1 and int(s[slen-2]) == 0):
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
                if(str(new_seq) != str(last_seq)):
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

                    #t_indices.append(self.vt.transform([int(s[13])]))

                    t_indices2.append(s[2])
                    t_indices2.append(s[8])
                    t_indices2.append(s[9])
                    t_indices2.append(self.vt.transform([int(s[13])])[0])
                    #t_indices2.append(self.vt.transform([201304])[0])


                    #if(max_seqlen+1 = len(current_deli)):
                        ##this kind of account just lose the fisrt record with month 0. First store the last seqeucne and add the data with month 0. Then add back the last sequence
                        #stored_deli = self.deli[len(self.deli)-len(current_deli):len(self.deli)]
                        #stored_labels2 = self.labels2[len(self.labels2)-len(current_deli):len(self.labels2)]
                        #stored_final_data =  self.final_data[len(self.final_data)-len(current_deli):len(self.final_data)] 

                        #del self.deli[len(self.deli)-len(current_deli):len(self.deli)-1]
                        #del self.labels2[len(self.labels2)-len(current_deli):len(self.labels2)-1]
                        #del self.final_data[len(self.final_data)-len(current_deli):len(self.final_data)-1]   

                        ##create data with month 0
                        #temp_deli = [0]
                        #temp_labels2 = [1.,0.]
                        #s_temp = self.load_data[i-1].split(' ')
                        #s_temp[len(s_temp)-1] = 0
                        #s_temp[len(s_temp)-2] = s_temp[len(s_temp)-2] - max_seqlen
                        #s_temp[len(s_temp)-5] = 0
                        #temp_final_data = s_temp

                        ##add all sequences
                        #self.deli =self.deli + temp_deli + stored_deli
                        #self.labels2 =self.labels2 + temp_labels2 + stored_labels2
                        #self.final_data =self.final_data + temp_final_data + stored_final_data                

                    if(max_seqlen+1 != len(current_deli)-1):
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

                        #default = 1
                        if(is_default == 1):
                            self.labels.append([0., 1.])
                        #default = 0
                        else:
                            self.labels.append([1., 0.])


                        #e.g., s[slen-5]=7,means at 7th quarter, but totally 8 quarters
                        #self.seqlen.append(int(s[slen-5])+1)

                        #exclude the exception of default = 0 after default = 1 within one account


                        #case for solving initial state problem
                        self.seqlen.append(max_seqlen+1)

                    is_default = 0
                    is_break = 0
                    max_seqlen = 0

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

                #t_indices.append(self.vt.transform([int(s[13])]))

                t_indices2.append(s[2])
                t_indices2.append(s[8])
                t_indices2.append(s[9])
                t_indices2.append(self.vt.transform([int(s[13])])[0])
                #t_indices2.append(self.vt.transform([201304])[0])

                if(max_seqlen+1 != len(current_deli)):
                    is_break = 1
                    del self.deli[len(self.deli)-len(current_deli):len(self.deli)]
                    del self.labels2[len(self.labels2)-len(current_deli):len(self.labels2)]
                    del self.final_data[len(self.final_data)-len(current_deli):len(self.final_data)]

                if(is_break != 1):
                    self.seq.append(last_seq)
                    self.data2.append(t_indices)
                    self.data.append(t_indices2) 

                    #default = 1
                    if(is_default == 1):
                        self.labels.append([0., 1.])
                    #default = 0
                    else:
                        self.labels.append([1., 0.])


                    #e.g., s[slen-5]=7,means at 7th quarter, but totally 8 quarters
                    #self.seqlen.append(int(s[slen-5])+1)

                    #exclude the exception of default = 0 after default = 1 within one account


                    #case for solving initial state problem
                    self.seqlen.append(max_seqlen+1)

                is_default = 0
                is_break = 0
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
            self.seqlen = np.array(self.seqlen)
            self.deli = np.array(self.deli)
            self.seq = np.array(self.seq)

            #standard transformation
            self.data2 = scaler.transform(self.data2)
            #print('data2:',self.data2[0])

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
                #self.data, self.seqlen, self.labels, self.data2, self.seq= self.shuffle()
                #self.finish_epoch = True

                #del self.labels2                
                #b = []
                #for i in range(len(self.data)):                
                    #for j in range(self.seqlen[i]-41):
                        #b.append([1., 0.])
                    #b.append(self.labels[i])
                #self.labels2 = b        

                ##random the delinqeuncy but store the sequential information
                #del self.deli

                #a = []
                #for i in range(len(self.load_data)):
                    #s = self.load_data[i].split(' ')
                    #if(int(s[len(s)-5]) <= 2):
                        #a.append(int(0))
                    #else:
                        #s_p3 = self.load_data[i-3].split(' ')		
                        #a.append(int(s_p3[len(s_p3)-1]))

                #search_data = []
                #for i in range(len(self.load_data)):
                    #s = self.load_data[i].split(' ')
                    #if(int(s[len(s)-5]) <= 2):
                        #s[len(s)-1]= int(0)
                        #search_data.append(s)
                    #else:
                        #s_p3 = self.load_data[i-3].split(' ')		
                        #search_data.append(s_p3)

                #b = []
                #temp = search_data
                #for i in range(len(self.seq)):      
                    #temp.sort(key = lambda x:x[11]!=self.seq[i]) 
                    #print(i,'/',len(self.seq))
                    #t =  search_data.index(temp[0])   
                    #b = b + a[t:t+self.seqlen[i]-40]
                    ##for j in range(len(self.load_data)):
                        ##if(self.seq[i] == self.load_data[j].split(' ')[11]):
                            ##b = b + a[j:j+self.seqlen[i]-40]
                            ##print(len(b))
                            ##print(i)
                            ##break

                #self.deli = b        
                #print('deli finish shuffle!!')

                #self.batch_id = 0
                #self.batch_all_id = 0
                #self.batch_all_count = 0
                #self.batch_all_deli_id = 0
                #self.batch_all_seq_id = 0

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
                    batch_seqlen.append(1)


            all_count = 0
            #for i in range(512):
                #all_count = all_count + self.seqlen[i]

            #for i in range(512,512+32-1):
                #print(self.labels2[all_count:all_count+self.seqlen[i]])
                #all_count = all_count + self.seqlen[i]
                #print('=======================================')



            for i in range(batch_size):
                all_count = all_count + batch_seqlen[i] - 0



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

            #print('return')
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
                return a, b, c, d, e, f, g ,True
            else:
                a, b, c, d, e, f, g = self.loseData.next(batch)
                return a, b, c, d, e, f, g, False

    class BASE_RNN():

        train_data = None
        def init_matrix(self, shape):
            return tf.random_normal(shape, stddev=0.1)

        def __init__(self,  EMB_DIM = 16,
                            FEATURE_SIZE = 3,
                            FEATURE_SIZE2 = 7,
                            BATCH_SIZE = 128,
                            TRUE_BATCH_SIZE = 128,
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
                            NUM_LAYERS = 2
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
            self.TRUE_BATCH_SIZE = TRUE_BATCH_SIZE
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
            self.batch_seq = []


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


            self.tf_lr = tf.placeholder(tf.float32,name="tf_lr")

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
                        #washout_one_hot_deli = tf.one_hot(tf.tile([tf.cast(0, dtype=tf.int32)],[40]),depth = 4)
                        #one_hot_deli = tf.concat([washout_one_hot_deli,tf.one_hot(tf.cast(self.tf_x_deli[0:self.tf_bid_len[0]-0], dtype=tf.int32),depth = 4)],0)
                        one_hot_deli = tf.one_hot(tf.cast(self.tf_x_deli[0:self.tf_bid_len[0]-0], dtype=tf.int32),depth = 4)
                        left_deli = tf.one_hot(tf.tile([tf.cast(0,dtype=tf.int32)], [self.MAX_SEQ_LEN-self.tf_bid_len[0] + 40-40]),depth = 4)
                        one_hot_deli = tf.concat([one_hot_deli,tf.cast(left_deli,dtype=tf.float32)],0)
                        id = self.tf_bid_len[0] - 0
                    else:
                        #append the first x washout value accotding to the washout month
                        #washout_one_hot_deli = tf.one_hot(tf.tile([tf.cast(0, dtype=tf.int32)],[40]),depth = 4)
                        #one_hot_deli = tf.concat([one_hot_deli,washout_one_hot_deli],0)
                        one_hot_deli = tf.concat([one_hot_deli,tf.one_hot(tf.cast(self.tf_x_deli[id:id+self.tf_bid_len[i]-0], dtype=tf.int32),depth = 4)],0)
                        left_deli = tf.one_hot(tf.tile([tf.cast(0,dtype=tf.int32)], [self.MAX_SEQ_LEN-self.tf_bid_len[i]+40-40]),depth = 4)
                        one_hot_deli = tf.concat([one_hot_deli,tf.cast(left_deli,dtype=tf.float32)],0)
                        id = id + self.tf_bid_len[i] - 0


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

            if True:
                outlist = []
                input_x = tf.reshape(input_x, [self.MAX_SEQ_LEN * self.BATCH_SIZE, -1])
                sigleout = tf.layers.dense(input_x, 3, tf.nn.relu,kernel_initializer='glorot_uniform',kernel_regularizer=l2(1e-4))

                h = tf.concat([input_x, sigleout], axis=1)
                h = tf.layers.dense(h, 5, tf.nn.relu,kernel_initializer='glorot_uniform',kernel_regularizer=l2(1e-4))
                h = tf.nn.dropout(h, keep_prob=0.6)
                h = tf.layers.dense(h, 55, tf.nn.relu,kernel_initializer='glorot_uniform',kernel_regularizer=l2(1e-4))
                #for i in range(0,self.BATCH_SIZE):
                    #sigleout = tf.layers.dense(input_x[i], 3, tf.nn.relu)
                    #h = tf.concat([input_x, shared_out], axis=1)
                    #sigleout = tf.layers.dense(sigleout, 5, tf.nn.relu)
                    #outlist.append(sigleout)
                out = tf.nn.dropout(h, keep_prob=0.6)
                preds = tf.layers.dense(out, 1, tf.nn.sigmoid,kernel_initializer='glorot_uniform',kernel_regularizer=l2(1e-4))
                #preds = tf.reshape(tf.stack(outlist, axis=0), [self.BATCH_SIZE, self.MAX_SEQ_LEN], name="preds")
                #print(self.BATCH_SIZE)

                #1
                #preds = tf.transpose(tf.nn.sigmoid(logits, name="preds"), name="preds")[0]
                #preds = tf.reshape(tf.stack(outlist, axis=0), [self.BATCH_SIZE, self.MAX_SEQ_LEN], name="preds")
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
                #create trainable attention weighted vector


                #initialize a trainable weight to mul with outputs, and then use the multiplied weighted matrix with the original outputs for self-attention
                #W_attention2 = tf.Variable(tf.random_normal([self.STATE_SIZE,self.STATE_SIZE], stddev=0.1),name='trainable_attention_mul_weight2')
                #e = tf.matmul(outputs,W_attention2) 
                #e = tf.matmul(e,tf.transpose(outputs,[0,2,1])) 
                #initialize a trainable weight to mul with outputs, and then softmax it to create attention score weight
                #W_attention = tf.Variable(tf.random_normal([self.STATE_SIZE,self.MAX_SEQ_LEN], stddev=0.1),name='trainable_attention_mul_weight')
                #e = tf.matmul(outputs,W_attention) 

                #switch or switch off the attention mechanism
                e = tf.matmul(outputs,tf.transpose(outputs,[0,2,1])) #(batch * time step * hidden size) * (batch * hidden size * time step) ---> batch * time step * time step
                self.attention = tf.nn.softmax(e,dim=-1) #batch * time step * time step
                self.attention_score_seq = tf.add(self.attention,tf.zeros([self.BATCH_SIZE,self.MAX_SEQ_LEN,self.MAX_SEQ_LEN]),name='attention_score_seq')

                #test: whether attention actually works,make the same score for each time steps
                #e = tf.random.uniform([self.BATCH_SIZE,self.MAX_SEQ_LEN,self.MAX_SEQ_LEN],0.,5.)
                #self.attention = tf.nn.softmax(e,dim=-1)
                outputs = tf.matmul(self.attention,outputs) # (batch * time step * time step) * (batch * time step * hidden size) ---> batch * time step * hidden size
                self.attention_weighted_output = outputs

                self.attention_weighted_output = tf.add(outputs,tf.zeros([self.BATCH_SIZE,self.MAX_SEQ_LEN,self.STATE_SIZE]),name='attention_weighted_output')

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
            self.id = []

            #
            default = []
            good = []
            bad = []
            count_predict = 0

            for i in range(self.BATCH_SIZE):  
                if(i == 0):
                    id = 0
                if(id == 0):
                    survival_rate = self.map_parameter[i][0:self.tf_bid_len[i]]
                    dead_rate = tf.subtract(tf.constant(1.0, dtype=tf.float32), survival_rate)
                    predict = tf.transpose(tf.stack([survival_rate, dead_rate])) 
                    self.cross_entropy2 = self.cross_entropy2 -tf.reduce_sum(self.tf_y2[0:self.tf_bid_len[i]-0]*tf.log(tf.clip_by_value(predict,1e-10,1.0)))
                    predicted_label = predict
                    true_label = self.tf_y2[id:id+self.tf_bid_len[i]-0]
                    id = self.tf_bid_len[i]-0
                    #need count_predict + 1? Does this line forgeted before?
                    count_predict = count_predict + 1
                    #self.id = id
                    #true_label.append(self.tf_y2[id:id+self.tf_bid_len[i]])


                    #predicted_label = tf.concat([input_x, tf.cast(input_x2, dtype=tf.float32)], 0)
                else:
                    survival_rate = self.map_parameter[i][0:self.tf_bid_len[i]]
                    dead_rate = tf.subtract(tf.constant(1.0, dtype=tf.float32), survival_rate)
                    predict = tf.transpose(tf.stack([survival_rate, dead_rate])) 

                    self.cross_entropy2 = self.cross_entropy2 -tf.reduce_sum(self.tf_y2[id:id+self.tf_bid_len[i]-0]*tf.log(tf.clip_by_value(predict,1e-10,1.0)))
                    predicted_label = tf.concat([predicted_label, tf.cast(predict, dtype=tf.float32)], 0)
                    true_label = tf.concat([true_label, tf.cast(self.tf_y2[id:id+self.tf_bid_len[i]-0], dtype=tf.float32)], 0)
                    count_predict = count_predict + 1
                    id = id + self.tf_bid_len[i]-0
                    #self.id = tf.concat([self.id, tf.cast(self.tf_bid_len[i], dtype=tf.int32)], 1)
                self.count_predict = count_predict



                #predicted_label.append(predict)
                #for i in range(self.tf_bid_len[i].shape):
                    #default.append(dead_rate[i])



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

            #self.BATCH_SIZE OR count_predict
            #for i in range(count_predict):
            for i in range(self.BATCH_SIZE):
                all_count = all_count + self.tf_bid_len[i]-0
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
            cost = tf.add(self.cross_entropy2, 0, name = "cost")
            self.cost = tf.add(cost, 0, name="cost")

            #self.cost = tf.add(self.cross_entropy2, 0, name="cost")

            #self.cost = tf.add(self.cross_entropy2, 0, name="cost")
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
                warmup_start_lr = 1e-5
                warmup_steps = 2000
                lr0 = 1e-2
                warmup_factor = (lr0/warmup_start_lr)**(1/warmup_steps)
                power = 0.9

                batch_x, batch_x2, batch_y, batch_len, batch_y2, batch_x_deli, batch_seq, win = self.train_data.next(self.BATCH_SIZE)

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
                        #warm up phase/learning rate
                        #if warmup_steps and step <= warmup_steps:

                            #learning_rate = warmup_start_lr*(warmup_factor**step)
                        #else:
                            #factor = (1-(step-warmup_steps)/(self.TRAING_STEPS-warmup_steps))**power
                            #learning_rate = lr0*factor
                        #
                        _,  train_loss, train_cross_entropy2, survival_rate,  train_predicted_label,train_vintage, train_input_x2,train_tf_x= sess.run([self.com_train_op, self.cost, self.cross_entropy2, self.survival_rate,self.predicted_label,self.vintage,self.input_x2,self.tf_x],
                                                                        feed_dict={self.tf_x: batch_x,
                                                                                    self.tf_x2: batch_x2,
                                                                                    self.tf_y: batch_y,
                                                                                    self.tf_y2: batch_y2,
                                                                                    self.tf_x_deli: batch_x_deli,      
                                                                                    self.tf_seq: batch_seq,                                                                           
                                                                                    self.tf_bid_len: batch_len,
                                                                                    #self.tf_lr: learning_rate, 
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
                    #print('learning rate:',learning_rate)
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
                        #1
                        self.run_test(sess)

                        self.save_model()
                elif self.global_step < 1000:
                    self.SHOW_SURVIVAL_CURVE = True
                    if self.global_step % 100 == 0:
                        #1
                        self.run_test(sess)
                        self.save_model()
                elif self.global_step < 3000:
                    #self.SHOW_SURVIVAL_CURVE = True
                    if self.global_step % 200 == 0:
                        #1
                        self.run_test(sess)
                        self.save_model()
                elif self.global_step < 7000:
                    #self.SHOW_SURVIVAL_CURVE = True
                    if self.global_step % 200 == 0:
                        #1
                        self.run_test(sess)
                        self.save_model()        
                elif self.global_step < 12000:
                    #self.SHOW_SURVIVAL_CURVE = True
                    if self.global_step % 300 == 0:
                        #1
                        self.run_test(sess)
                        self.save_model()
                elif self.global_step < 21000:
                    #self.SHOW_SURVIVAL_CURVE = True
                    if self.global_step % 500 == 0:
                        #1
                        self.run_test(sess)
                        self.save_model()
                elif self.global_step <= 30000:
                    #self.SHOW_SURVIVAL_CURVE = True
                    if self.global_step % 500 == 0:
                        #1
                        self.run_test(sess)
                        self.save_model()
                elif self.global_step <= 50000:
                    self.SHOW_SURVIVAL_CURVE = True
                    if self.global_step % 500 == 0:
                        self.run_test(sess)
                        self.save_model()
                elif self.global_step <= 70000:
                    self.SHOW_SURVIVAL_CURVE = True
                    if self.global_step % 1000 == 0:
                        1
                        #self.run_test(sess)
                        #self.save_model()
                #elif self.global_step <= 102000:
                    #self.SHOW_SURVIVAL_CURVE = True
                    #if self.global_step % 3000 == 0:
                        #self.run_test(sess)
                        #self.save_model()
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
            #config.gpu_options.per_process_gpu_memory_fraction = 0.5
            with tf.Session(config=config) as sess:
                self.train_test(sess)

        def save_model(self):
            print("model name: ", self.filename, " ", self.global_step, "\n")
            self.saver.save(self.sess, "E:/saved_model/model04to12_DeepHit_gpu_forecast_deli_noL2_512_16_exclude_exception/" + self.filename, global_step=self.global_step)

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
            #config.gpu_options.per_process_gpu_memory_fraction = 0.5
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
            #self.preds = graph.get_tensor_by_name("preds:0")
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
                bid_loss, bid_test_cross_entropy2, bid_true_label, bid_predicted_label,survival_rate,test_vintage= sess.run(
                    [self.cost,  self.cross_entropy2, self.true_label,self.predicted_label,self.survival_rate,self.vintage],
                    feed_dict={self.tf_x: test_batch_x,
                                self.tf_x2: test_batch_x2,
                                self.tf_y: test_batch_y,
                                self.tf_y2: test_batch_y2,
                                self.tf_seq: test_batch_seq,   
                                self.tf_bid_len: test_batch_len,
                                self.tf_x_deli: test_batch_x_deli,
                                self.tf_control_parameter:[self.ALPHA, self.BETA]
                                #self.tf_market_price: test_batch_market_price
                                })


                #
                if(count == 0):
                    #fig = plt.figure()
                    #plt.grid(False)
                    # draw the survival curve
                    #print('vintage:',test_vintage)
                    x_axis = []
                    #print('test_survival_rate',survival_rate)
                    for i in range(0,len(survival_rate)):

                        x_axis.append((i+1))

                    #plt.plot(x_axis, survival_rate,'b')
                    #plt.ylabel('Survival Rate')
                    #plt.xlabel('Time')
                    #plt.show()
                    count = count + 1


                #for i in range(len(bid_true_label)):
                    #for j in range(len(bid_true_label[i])):

                        #self.TRUE_LABEL2.append(bid_true_label[i][j])
                        #self.PREDICTED_LABEL2.append(bid_predicted_label[i][j])


                #self.WEIGHT.append(bid_weight)

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

    

    #deephit standardscaler
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

    #DeepHit
    #Forecast
    #initial_extend8 + washout:30，take care of the length of the tf__y2 from the perspective of cross ectropy code and sparsedata code
    #use the second copy of the lstm code
    #remember to substract 8 also in count length of the crossentropy
    #import seaborn as sns
    from scipy import stats
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import auc

    #deephit tesing
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
        meta = new_path + "/Replication_IJF/saved_model/model04to12_DeepHit_gpu_forecast_deli_noL2_512_16_exclude_exception" + "/drsa32_512_16_0.000100_0.100000_2259_1.20_0.20_True_False_1_1.meta"
        ckpt = new_path + "/Replication_IJF/saved_model/model04to12_DeepHit_gpu_forecast_deli_noL2_512_16_exclude_exception" + "/drsa32_512_16_0.000100_0.100000_2259_1.20_0.20_True_False_1_1"
    else:
        path = os.getcwd()
        new_path = path.replace("\\","/")
        meta = new_path + "/Replication_IJF/saved_model/model16to21_DeepHit_gpu_forecast_deli_noL2_512_16_exclude_exception" + "/drsa32_512_16_0.000100_0.100000_2259_1.20_0.20_True_False_1_1.meta"
        ckpt = new_path + "/Replication_IJF/saved_model/model16to21_DeepHit_gpu_forecast_deli_noL2_512_16_exclude_exception" + "/drsa32_512_16_0.000100_0.100000_2259_1.20_0.20_True_False_1_1"
    step = 100000

    sess = RUNNING_MODEL.load(meta,ckpt,step)

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

    #print(RUNNING_MODEL.BATCH_SIZE)        
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
                    survival_rate0 = RUNNING_MODEL.map_parameter[j][0:RUNNING_MODEL.tf_bid_len[j]]
                survival_rate = RUNNING_MODEL.map_parameter[j][0:RUNNING_MODEL.tf_bid_len[j]]
                dead_rate = tf.subtract(tf.constant(1.0, dtype=tf.float32), survival_rate)
                predict = tf.transpose(tf.stack([survival_rate, dead_rate])) 
                id = RUNNING_MODEL.tf_bid_len[j]-0
                cross_entropy2 = -tf.reduce_sum(RUNNING_MODEL.tf_y2[0:id]*tf.log(tf.clip_by_value(predict,1e-10,1.0)))
                if(count_predict == 0):
                    predicted_label2 = predict
                    true_label2 = RUNNING_MODEL.tf_y2[0:id]
                    seqlen2 = tf.stack([tf.cast(RUNNING_MODEL.tf_bid_len[0]-0,dtype=tf.float32),0.0])
                else:
                    predicted_label2 = tf.concat([predicted_label2, tf.cast(predict, dtype=tf.float32)], 0)
                    true_label2 = tf.concat([true_label2, tf.cast(RUNNING_MODEL.tf_y2[0:id], dtype=tf.float32)], 0)
                    seqlen2 = tf.concat([seqlen2, tf.stack([tf.cast(RUNNING_MODEL.tf_bid_len[0]-0,dtype=tf.float32),0.0])], 0)
                count_predict = count_predict + 1
                #id = id + RUNNING_MODEL.tf_bid_len[i]
            else:
                survival_rate = RUNNING_MODEL.map_parameter[j][0:RUNNING_MODEL.tf_bid_len[j]]
                dead_rate = tf.subtract(tf.constant(1.0, dtype=tf.float32), survival_rate)
                predict = tf.transpose(tf.stack([survival_rate, dead_rate])) 
                cross_entropy2 = cross_entropy2 -tf.reduce_sum(RUNNING_MODEL.tf_y2[id:id+RUNNING_MODEL.tf_bid_len[j]-0]*tf.log(tf.clip_by_value(predict,1e-10,1.0)))

                if(count_predict == 0):
                    predicted_label2 = predict
                    true_label2 = RUNNING_MODEL.tf_y2[id:id+RUNNING_MODEL.tf_bid_len[j]-0]
                    seqlen2 = tf.stack([tf.cast(RUNNING_MODEL.tf_bid_len[j]-0,dtype=tf.float32),0.0])
                else:
                    predicted_label2 = tf.concat([predicted_label2, tf.cast(predict, dtype=tf.float32)], 0)
                    true_label2 = tf.concat([true_label2, tf.cast(RUNNING_MODEL.tf_y2[id:id+RUNNING_MODEL.tf_bid_len[j]-0], dtype=tf.float32)], 0)
                    seqlen2 = tf.concat([seqlen2, tf.stack([tf.cast(RUNNING_MODEL.tf_bid_len[j]-0,dtype=tf.float32),0.0])], 0)
                count_predict = count_predict + 1
                id = id + RUNNING_MODEL.tf_bid_len[j] - 0
            #true_label.append(self.tf_y2[id:id+self.tf_bid_len[i]])

            #predicted_label = tf.concat([input_x, tf.cast(input_x2, dtype=tf.float32)], 0)



            #predicted_label.append(predict)
            #for i in range(self.tf_bid_len[i].shape):
                #default.append(dead_rate[i])

        all_count = 0
        for m in range(RUNNING_MODEL.BATCH_SIZE):
            all_count = all_count + RUNNING_MODEL.tf_bid_len[m] - 0
        cross_entropy2 = cross_entropy2/tf.cast(all_count,dtype=tf.float32)


        bid_loss,test_survival_rate,test_t2,test_true_label,test_predicted_label,test_seqlen,test_count= sess.run(
            [cross_entropy2,survival_rate,RUNNING_MODEL.t2,true_label2,predicted_label2,seqlen2,all_count],
            feed_dict={RUNNING_MODEL.tf_x: test_batch_x,
                        RUNNING_MODEL.tf_x2: test_batch_x2,
                        RUNNING_MODEL.tf_y: test_batch_y,
                        RUNNING_MODEL.tf_y2: test_batch_y2,
                        RUNNING_MODEL.tf_x_deli: test_batch_x_deli,
                        RUNNING_MODEL.tf_seq: test_batch_seq,
                        RUNNING_MODEL.tf_bid_len: test_batch_len
                        #RUNNING_MODEL.tf_training: False    
                        #self.tf_market_price: test_batch_market_price
                        })
        #print(bid_loss)
        count = count + test_count
        loss_arr.append(bid_loss*test_count)
        true_label.append(test_true_label)
        predicted_label.append(test_predicted_label)
        seqlen.append(test_seqlen)
        #if(RUNNING_MODEL.SHOW_SURVIVAL_CURVE == True):

        #draw the conditional survival curve
        #x_axis = []
        #hr = []
        #print('test_conditional_survival_rate',test_survival_rate)
        #for k in range(0,len(test_survival_rate)):
            #hr.append(1-test_survival_rate[k])
            #x_axis.append((k+1))

        #plt.plot(x_axis, hr,'b')
        #plt.ylabel('Hazard Rate')
        #plt.xlabel('Time')
        #plt.show() 

        #draw the survival curve
        #x_axis = []
        #print('test_survival_rate',test_survival_rate)
        #for k in range(0,len(test_survival_rate)):

            #x_axis.append((k+1))
        #sr = []
        #product = 1
        #for k in range(0,len(test_survival_rate)):
            #product = product*test_survival_rate[k]
            #sr.append(product)
        #plt.plot(x_axis, sr,'b')
        #plt.ylabel('Survival Rate')
        #plt.xlabel('Time')
        #plt.show() 

    #usually the number of the last test batch is not equal to the batch size
    remaining = RUNNING_MODEL.test_data_win.size - int(RUNNING_MODEL.test_data_win.size / RUNNING_MODEL.BATCH_SIZE)*RUNNING_MODEL.BATCH_SIZE
    #print('remaining:',remaining)
    if(remaining != 0):
        test_batch_x, test_batch_x2, test_batch_y, test_batch_len, test_batch_y2, test_batch_x_deli, test_batch_seq = RUNNING_MODEL.test_data_win.next(
            remaining)
        #test_batch_y = tf.ragged.RaggedTensorValue(test_batch_y,np.array([self.BATCH_SIZE]))
        #start_time = time.time()

    for i in range(remaining):
        survival_rate = RUNNING_MODEL.map_parameter[i][0:RUNNING_MODEL.tf_bid_len[i]]

    count_predict = 0

    #customized loss function according to the number of the remaining test data
    for i in range(remaining):  
        if(i == 0):

            survival_rate = RUNNING_MODEL.map_parameter[i][0:RUNNING_MODEL.tf_bid_len[i]]
            dead_rate = tf.subtract(tf.constant(1.0, dtype=tf.float32), survival_rate)
            predict = tf.transpose(tf.stack([survival_rate, dead_rate])) 
            id = RUNNING_MODEL.tf_bid_len[i]-0
            cross_entropy2 = -tf.reduce_sum(RUNNING_MODEL.tf_y2[0:id]*tf.log(tf.clip_by_value(predict,1e-10,1.0)))



            if(count_predict == 0):
                predicted_label2 = predict
                true_label2 = RUNNING_MODEL.tf_y2[0:id]
                seqlen2 = tf.stack([tf.cast(RUNNING_MODEL.tf_bid_len[0]-0,dtype=tf.float32),0.0])
            else:
                predicted_label2 = tf.concat([predicted_label2, tf.cast(predict, dtype=tf.float32)], 0)
                true_label2 = tf.concat([true_label2, tf.cast(RUNNING_MODEL.tf_y2[0:id], dtype=tf.float32)], 0)
                seqlen2 = tf.concat([seqlen2, tf.stack([tf.cast(RUNNING_MODEL.tf_bid_len[0]-0,dtype=tf.float32),0.0])], 0)
            count_predict = count_predict + 1
            #id = id + RUNNING_MODEL.tf_bid_len[i]
        else:
            if(i == 12):
                length = RUNNING_MODEL.tf_bid_len[i]
                survival_rate0 = RUNNING_MODEL.map_parameter[i][0:RUNNING_MODEL.tf_bid_len[i]]
            survival_rate = RUNNING_MODEL.map_parameter[i][0:RUNNING_MODEL.tf_bid_len[i]]
            dead_rate = tf.subtract(tf.constant(1.0, dtype=tf.float32), survival_rate)
            predict = tf.transpose(tf.stack([survival_rate, dead_rate])) 
            cross_entropy2 = cross_entropy2 -tf.reduce_sum(RUNNING_MODEL.tf_y2[id:id+RUNNING_MODEL.tf_bid_len[i]-0]*tf.log(tf.clip_by_value(predict,1e-10,1.0)))

            if(count_predict == 0):
                predicted_label2 = predict
                true_label2 = RUNNING_MODEL.tf_y2[id:id+RUNNING_MODEL.tf_bid_len[i]-0]
                seqlen2 = tf.stack([tf.cast(RUNNING_MODEL.tf_bid_len[i]-0,dtype=tf.float32),0.0])
            else:
                predicted_label2 = tf.concat([predicted_label2, tf.cast(predict, dtype=tf.float32)], 0)
                true_label2 = tf.concat([true_label2, tf.cast(RUNNING_MODEL.tf_y2[id:id+RUNNING_MODEL.tf_bid_len[i]-0], dtype=tf.float32)], 0)
                seqlen2 = tf.concat([seqlen2, tf.stack([tf.cast(RUNNING_MODEL.tf_bid_len[i]-0,dtype=tf.float32),0.0])], 0)
            count_predict = count_predict + 1
            id = id + RUNNING_MODEL.tf_bid_len[i] - 0
        #true_label.append(self.tf_y2[id:id+self.tf_bid_len[i]])


        #predicted_label = tf.concat([input_x, tf.cast(input_x2, dtype=tf.float32)], 0)



        #predicted_label.append(predict)
        #for i in range(self.tf_bid_len[i].shape):
            #default.append(dead_rate[i])

    all_count = RUNNING_MODEL.tf_bid_len[0] - 0
    for i in range(1,remaining):
        all_count = all_count + RUNNING_MODEL.tf_bid_len[i] - 0
    cross_entropy2 = cross_entropy2/tf.cast(all_count,dtype=tf.float32)



    bid_loss,test_survival_rate,test_t2,test_true_label,test_predicted_label,test_seqlen,test_count= sess.run(
        [cross_entropy2,survival_rate0,RUNNING_MODEL.t2,true_label2,predicted_label2,seqlen2,all_count],
        feed_dict={RUNNING_MODEL.tf_x: test_batch_x,
                    RUNNING_MODEL.tf_x2: test_batch_x2,
                    RUNNING_MODEL.tf_y: test_batch_y,
                    RUNNING_MODEL.tf_y2: test_batch_y2,
                    RUNNING_MODEL.tf_x_deli: test_batch_x_deli,
                    RUNNING_MODEL.tf_seq: test_batch_seq,
                    RUNNING_MODEL.tf_bid_len: test_batch_len
                    #RUNNING_MODEL.tf_training: False
                    #self.tf_market_price: test_batch_market_price
                    })
    #print('lentgh:',test_length)

    #print(bid_loss)
    count = count + test_count
    loss_arr.append(bid_loss*test_count)
    true_label.append(test_true_label)
    seqlen.append(test_seqlen)
    predicted_label.append(test_predicted_label)
    #if(RUNNING_MODEL.SHOW_SURVIVAL_CURVE == True):

    #print('test_trainable_attention_mul_weight',test_trainable_attention_mul_weight)

    #print(attention_weighted_output):
    #print('first attention_weighted_output:',test_attention_weighted_output[0])

    #draw the conditional survival curve
    x_axis = []
    hr = []
    #print('test_conditional_survival_rate',test_survival_rate)
    #for k in range(0,len(test_survival_rate)):
        #hr.append(1-test_survival_rate[k])
        #x_axis.append((k+1))

    #plt.plot(x_axis, hr,'b')
    #plt.ylabel('Hazard Rate')
    #plt.xlabel('Time')
    #plt.show() 

    #draw the survival curve
    x_axis = []
    #print('test_survival_rate',test_survival_rate)
    #for k in range(0,len(test_survival_rate)):

        #x_axis.append((k+1))
    #sr = []
    #product = 1
    #for k in range(0,len(test_survival_rate)):
        #product = product*test_survival_rate[k]
        #sr.append(product)
    #plt.plot(x_axis, sr,'b')
    #plt.ylabel('Survival Rate')
    #plt.xlabel('Time')
    #plt.show() 

    mean_loss = 0
    for i in range(len(loss_arr)):
        mean_loss = mean_loss + loss_arr[i]
    mean_loss = mean_loss/count    

    #for log_likelihoo_ratio test
    #llr_test_deephit = -mean_loss*count 

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

    unbalanced_log_likeli_deephit = 0
    for i in range(len(unbalanced_prediction)):
        if(true_label[i][0] == 1.0):
            unbalanced_log_likeli_deephit = unbalanced_log_likeli_deephit + math.log(1-unbalanced_prediction[i])
        else:
            unbalanced_log_likeli_deephit = unbalanced_log_likeli_deephit + math.log(unbalanced_prediction[i])


    #AUC    
    auc_score = roc_auc_score(label_true,prediction)
    #print('AUC: ', auc_score)

    auc_score = roc_auc_score(label_true,unbalanced_prediction)
    #print('unbalanced_AUC: ', auc_score)

    general_auc = auc_score

    #print('DR: ', default/(len(predicted_label))*100)
    #print('Unbalanced DR: ', unbalanced_default/(len(predicted_label))*100)

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

    #print('baseline DR: ',count_bad/(count_bad + count_good)*100)

    #mean_anlp = np.array(anlp_arr).mean()
    #mean_auc = np.array(auc_arr).mean()
    #delete mean_auc for a moment
    #print("TEST DATA LOSS:",mean_loss)
    #print('Unbalanced LLR: ',(-unbalanced_log_likeli_deephit)/len(unbalanced_prediction))

    #for log_likelihoo_ratio test
    llr_test_unbalanced_deephit = unbalanced_log_likeli_deephit 

    unbalanced_prediction_deephit = unbalanced_prediction

    dr = count_bad/(count_bad + count_good)

    baseline_llr = -dr*math.log(dr)-(1-dr)*math.log(1-dr)

    

    seqlen2 = []
    for i in range(len(seqlen)):
        if(seqlen[i]!=0):
            seqlen2.append(int(seqlen[i]))

   # print('time window 24,36,60==============================')


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
                predict = unbalanced_prediction_deephit[id:id+int(seqlen2[i])]
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
                predict = unbalanced_prediction_deephit[id:int(id+time_window[m])]
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
            #print('AUC ',time_window[m],':',auc_score)

    #print(conditional_labels)
    AUC = np.array(AUC)    

    AUC24 = AUC.mean()

    #print('avg_AUC: ', AUC.mean())

    brier_score = 0
    for i in range(len(unbalanced_prediction_deephit)):
        brier_score = brier_score + (unbalanced_prediction_deephit[i]-int(true_label[i][1]))**2
    #print('Deephit brier score: ',brier_score/len(unbalanced_prediction_deephit))
    
    pseudo_deephit.append(AUC24)

    
    print('LSTM+Washout+3LSTM Attention+Deli...')

    #moved from lstm washout attention deli
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

            #print(RUNNING_MODEL.BATCH_SIZE)        
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
                                RUNNING_MODEL.tf_seq: test_batch_seq,
                                RUNNING_MODEL.tf_bid_len: test_batch_len
                                #RUNNING_MODEL.tf_training: False    
                                #self.tf_market_price: test_batch_market_price
                                })
                #print(bid_loss)
                count = count + test_count
                loss_arr.append(bid_loss*test_count)
                true_label.append(test_true_label)
                predicted_label.append(test_predicted_label)
                seqlen.append(test_seqlen)
                #if(RUNNING_MODEL.SHOW_SURVIVAL_CURVE == True):

                #draw the conditional survival curve
                #x_axis = []
                #hr = []
                #print('test_conditional_survival_rate',test_survival_rate)
                #for k in range(0,len(test_survival_rate)):
                    #hr.append(1-test_survival_rate[k])
                    #x_axis.append((k+1))

                #plt.plot(x_axis, hr,'b')
                #plt.ylabel('Hazard Rate')
                #plt.xlabel('Time')
                #plt.show() 

                #draw the survival curve
                #x_axis = []
                #print('test_survival_rate',test_survival_rate)
                #for k in range(0,len(test_survival_rate)):

                    #x_axis.append((k+1))
                #sr = []
                #product = 1
                #for k in range(0,len(test_survival_rate)):
                    #product = product*test_survival_rate[k]
                    #sr.append(product)
                #plt.plot(x_axis, sr,'b')
                #plt.ylabel('Survival Rate')
                #plt.xlabel('Time')
                #plt.show() 

            #usually the number of the last test batch is not equal to the batch size
            remaining = RUNNING_MODEL.test_data_win.size - int(RUNNING_MODEL.test_data_win.size / RUNNING_MODEL.BATCH_SIZE)*RUNNING_MODEL.BATCH_SIZE
            #print('remaining:',remaining)
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
                [cross_entropy2,survival_rate0,RUNNING_MODEL.t2,true_label2,predicted_label2,seqlen2,all_count],
                feed_dict={RUNNING_MODEL.tf_x: test_batch_x,
                            RUNNING_MODEL.tf_x2: test_batch_x2,
                            RUNNING_MODEL.tf_y: test_batch_y,
                            RUNNING_MODEL.tf_y2: test_batch_y2,
                            RUNNING_MODEL.tf_x_deli: test_batch_x_deli,
                            RUNNING_MODEL.tf_seq: test_batch_seq,
                            RUNNING_MODEL.tf_bid_len: test_batch_len
                            #RUNNING_MODEL.tf_training: False
                            #self.tf_market_price: test_batch_market_price
                            })
            #print('lentgh:',test_length)

            #print(bid_loss)
            count = count + test_count
            loss_arr.append(bid_loss*test_count)
            true_label.append(test_true_label)
            seqlen.append(test_seqlen)
            predicted_label.append(test_predicted_label)
            #if(RUNNING_MODEL.SHOW_SURVIVAL_CURVE == True):

            #print('test_trainable_attention_mul_weight',test_trainable_attention_mul_weight)

            #print(attention_weighted_output):
            #print('first attention_weighted_output:',test_attention_weighted_output[0])

            #draw the conditional survival curve
            #x_axis = []
            #hr = []
            #print('test_conditional_survival_rate',test_survival_rate)
            #for k in range(0,len(test_survival_rate)):
                #hr.append(1-test_survival_rate[k])
                #x_axis.append((k+1))

            #plt.plot(x_axis, hr,'b')
            #plt.ylabel('Hazard Rate')
            #plt.xlabel('Time')
            #plt.show() 

            #draw the survival curve
            #x_axis = []
            #print('test_survival_rate',test_survival_rate)
            #for k in range(0,len(test_survival_rate)):

                #x_axis.append((k+1))
            #sr = []
            #product = 1
            #for k in range(0,len(test_survival_rate)):
                #product = product*test_survival_rate[k]
                #sr.append(product)
            #plt.plot(x_axis, sr,'b')
            #plt.ylabel('Survival Rate')
            #plt.xlabel('Time')
            #plt.show() 

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
            #print('AUC: ', auc_score)

            auc_score = roc_auc_score(label_true,unbalanced_prediction)
            #print('unbalanced_AUC: ', auc_score)

            general_auc = auc_score

            #print('DR: ', default/(len(predicted_label))*100)
            #print('Unbalanced DR: ', unbalanced_default/(len(predicted_label))*100)

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

            #print('baseline DR: ',count_bad/(count_bad + count_good)*100)

            #mean_anlp = np.array(anlp_arr).mean()
            #mean_auc = np.array(auc_arr).mean()
            #delete mean_auc for a moment
            #print("TEST DATA LOSS:",mean_loss)
            #print('Unbalanced LLR: ',(-unbalanced_log_likeli_3lstm_attention_deli)/len(unbalanced_prediction))

            #for log_likelihoo_ratio test
            llr_test_unbalanced_3lstm_attention_deli = unbalanced_log_likeli_3lstm_attention_deli 

            dr = count_bad/(count_bad + count_good)

            baseline_llr = -dr*math.log(dr)-(1-dr)*math.log(1-dr)
            pseudo_3lstm.append(1-((-unbalanced_log_likeli_3lstm_attention_deli)/len(unbalanced_prediction))/baseline_llr)

            unbalanced_prediction_3lstm_attention_deli = unbalanced_prediction

            seqlen2 = []
            for i in range(len(seqlen)):
                if(seqlen[i]!=0):
                    seqlen2.append(int(seqlen[i]))

            #print('time window 24,36,60==============================')

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
                        predict = unbalanced_prediction_3lstm_attention_deli[id:id+int(seqlen2[i])]
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
                        predict = unbalanced_prediction_3lstm_attention_deli[id:int(id+time_window[m])]
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
                    #print('AUC ',time_window[m],':',auc_score)

            #print(conditional_labels)
            AUC = np.array(AUC)    

            AUC24 = AUC.mean()

            #print('avg_AUC: ', AUC.mean())


            brier_score = 0
            for i in range(len(unbalanced_prediction_3lstm_attention_deli)):
                brier_score = brier_score + (unbalanced_prediction_3lstm_attention_deli[i]-int(true_label[i][1]))**2
            #print('3lstm_attention_deli brier score: ',brier_score/len(unbalanced_prediction_3lstm_attention_deli))
            #pseudo_r = 1-((-unbalanced_log_likeli_3lstm_attention_deli)/len(unbalanced_prediction))/baseline_llr
            #return sess #general testing
            return AUC24 #specific testing

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

    #Forecast
    #initial_extend8 + washout:30，take care of the length of the tf__y2 from the perspective of cross ectropy code and sparsedata code
    #use the second copy of the lstm code
    #remember to substract 8 also in count length of the crossentropy
    #import seaborn as sns
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

    pseudo_3lstm.append(RUNNING_MODEL.load(meta,ckpt,step))
    
    
    if((q+1)<=4):
        q = q + 1
    else:
        q = 1
        ym += 1
    
    #The experimental results can be reproduced by running the corresponding trainning code (located in the training directory) to perform model training under different washout step configurations.
    #Model weights are saved in the saved_model folder within the saved directory.           
    #The code presented here enables readers to train and validate the model independently.
    #The following code provides an example of validating the output results. Readers may adapt it as needed.
    ########################################################################################################
    #index_g = prs
    #if(prs > 47):
        #index_g = prs - 48
    #print('Pseudo-R-Square for each model in this round.')
    #print('As can be seen, our proposed model"3LSTM+attn+wo+deli" performs best most of the time.')
    #print("Linear DTSM|","GAM|","Cox PH|","Weibull|","DeepHit|","3LSTM")
    #print(f"{(pseudo_dtsm[index_g]):.5f}","|",f"{(pseudo_gam[index_g]):.5f}","|",f"{(pseudo_cox[index_g]):.5f}","|",f"{(pseudo_weibull[index_g]):.5f}","|",f"{(pseudo_deephit[index_g]):.5f}",f"{(pseudo_3lstm[index_g]):.5f}")
    
    #if(prs == 47):
        ##draw the conditional survival curve
        #x_axis = []
        #hr = []
        #print('Pseudo-R-Square(2004-2015): ')
        #for k in range(0,len(pseudo_washout)):
            #x_axis.append((k+1))

        #plt.plot(x_axis,pseudo_dtsm,'blue',label='Linear DTSM',)
        #plt.plot(x_axis,pseudo_gam,'red',label='GAM',linestyle='--')
        #plt.plot(x_axis,pseudo_cox,'grey',label='Cox PH' )
        #plt.plot(x_axis,pseudo_weibull,'green',label='Weibull')
        #plt.plot(x_axis,pseudo_deephit,'black',label='DeepHit',linewidth=2)
        #plt.plot(x_axis,pseudo_3lstm,'black',label='3LSTM',linewidth=2)

        #plt.legend(prop = {'size':5})
        #plt.title('Pseudo-R-Square(2004-2015)')
        #plt.xlabel('Time')
        #plt.show() 
    #if(prs == 81):
        ##draw the conditional survival curve
        #x_axis = []
        #hr = []
        #print('Pseudo-R-Square(2016-2024): ')
        #for k in range(0,len(pseudo_washout)):
            #x_axis.append((k+1))

        #plt.plot(x_axis,pseudo_dtsm,'blue',label='Linear DTSM',)
        #plt.plot(x_axis,pseudo_gam,'red',label='GAM',linestyle='--')
        #plt.plot(x_axis,pseudo_cox,'grey',label='Cox PH' )
        #plt.plot(x_axis,pseudo_weibull,'green',label='Weibull')
        #plt.plot(x_axis,pseudo_deephit,'black',label='DeepHit',linewidth=2)
        #plt.plot(x_axis,pseudo_3lstm,'black',label='3LSTM',linewidth=2)

        #plt.legend(prop = {'size':5})
        #plt.title('Pseudo-R-Square(2016-2024)')
        #plt.xlabel('Time')
        #plt.show()    
        
import os
import tensorflow.compat.v1 as tf
from sklearn.metrics import roc_curve, auc
import matplotlib as mpl  
import matplotlib.pyplot as plt
import numpy as nplanced
import sklearn
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
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
import csv
import pandas as pd

#2004to2015
pseudo_dtsm = []
pseudo_gam = []
pseudo_cox = []
pseudo_weibull = []
pseudo_deephit = []
pseudo_3lstm = []
path = os.getcwd()
new_path = path.replace("\\","/")

csvFile = open(new_path + "/Replication_IJF/data/2259/AUC(04to15).csv", "r",encoding='gb18030', errors='ignore')
reader = csv.reader(csvFile)


result = []
for item in reader:
    data = []

    #if reader.line_num == 1:
        #continue
    for i in range(12):
        data.append(item[i])
    result.append(data)

csvFile.close()
df = pd.DataFrame(result)

pseudo_dtsm = df.iloc[2:50,1]
pseudo_gam = df.iloc[2:50,3]
pseudo_cox = df.iloc[2:50,5]
pseudo_weibull = df.iloc[2:50,7]
pseudo_deephit = df.iloc[2:50,9]
pseudo_3lstm = df.iloc[2:50,11]

#draw the curve of AUC from 2004 to 2015
x_axis = []
hr = []
print('AUC(2004-2015): ')
for k in range(0,len(pseudo_3lstm)):
    x_axis.append((k+1))

plt.figure(figsize=(9, 5))

plt.plot(x_axis,pseudo_dtsm.astype(float),'blue',label='Linear DTSM')
plt.plot(x_axis,pseudo_gam.astype(float),'red',label='Linear DTSM + MEVS',linestyle='--')
plt.plot(x_axis,pseudo_cox.astype(float),'grey',label='DeepHit' )
plt.plot(x_axis,pseudo_weibull.astype(float),'green',label='DeepHit + MEVs')
plt.plot(x_axis,pseudo_deephit.astype(float),'grey',label='DeepHit',linestyle='--')
plt.plot(x_axis,pseudo_3lstm.astype(float),'black',label='3LSTM',linewidth=2)


plt.legend(prop = {'size':5})
plt.title('AUC(2004-2015)')
plt.xlabel('Time')
plt.show()  

#2016to2024
pseudo_dtsm = []
pseudo_gam = []
pseudo_cox = []
pseudo_weibull = []
pseudo_deephit = []
pseudo_3lstm = []
path = os.getcwd()
new_path = path.replace("\\","/")

csvFile = open(new_path + "/Replication_IJF/data/2259/AUC(16to24).csv", "r",encoding='gb18030', errors='ignore')
reader = csv.reader(csvFile)


result = []
for item in reader:
    data = []

    #if reader.line_num == 1:
        #continue
    for i in range(12):
        data.append(item[i])
    result.append(data)

csvFile.close()
df = pd.DataFrame(result)

pseudo_dtsm = df.iloc[2:36,1]
pseudo_gam = df.iloc[2:36,3]
pseudo_cox = df.iloc[2:36,5]
pseudo_weibull = df.iloc[2:36,7]
pseudo_deephit = df.iloc[2:36,9]
pseudo_3lstm = df.iloc[2:36,11]

#draw the curve of AUC from 2016 to 2024
x_axis = []
hr = []
print('AUC(2016-2024): ')
for k in range(0,len(pseudo_3lstm)):
    x_axis.append((k+1))

plt.figure(figsize=(9, 5))

plt.plot(x_axis,pseudo_dtsm.astype(float),'blue',label='Linear DTSM')
plt.plot(x_axis,pseudo_gam.astype(float),'red',label='Linear DTSM + MEVS',linestyle='--')
plt.plot(x_axis,pseudo_cox.astype(float),'grey',label='DeepHit' )
plt.plot(x_axis,pseudo_weibull.astype(float),'green',label='DeepHit + MEVs')
plt.plot(x_axis,pseudo_deephit.astype(float),'grey',label='DeepHit',linestyle='--')
plt.plot(x_axis,pseudo_3lstm.astype(float),'black',label='3LSTM',linewidth=2)

plt.legend(prop = {'size':5})
plt.title('AUC(2016-2024)')
plt.xlabel('Time')
plt.show()   