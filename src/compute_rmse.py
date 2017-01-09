import numpy as np
import codecs
import math
import operator
from random import randint

k=125
while True:
    SUM_RMSE=0
    NUM_RMSE=0
    for i in range(5):
    
        file_train='C:/hoc tap/big data/data/ml-1m/ml-1m/split/train'+str(i+1)+'.txt'
        file_test='C:/hoc tap/big data/data/ml-1m/ml-1m/split/test'+str(i+1)+'.txt'
        
        item_vectors={}
        user_vectors={}
        overall_mean_rating=0
        num_overall_rating=0
        
        infile=codecs.open(file_train , 'r' , encoding='utf-8' )
        for line in infile :
            line=line.split('::')
            user_id=int(line[0])
            item_id=int(line[1])
            rating=int(line[2])   
            
            overall_mean_rating+=rating
            num_overall_rating+=1
             
            if item_id in item_vectors:
                item_vectors[item_id][user_id]=rating
            else:
                item_vectors[item_id]={}
                item_vectors[item_id][user_id]=rating
                
            if user_id in user_vectors:
                user_vectors[user_id][item_id]=rating
            else:
                user_vectors[user_id]={}
                user_vectors[user_id][item_id]=rating
        
        #tinh average 
        overall_mean_rating=overall_mean_rating/num_overall_rating
        #print('overall_mean_rating : '+str(overall_mean_rating))
        
        item_average_rate={}       
        user_average_rate={}
        
        for item_id in item_vectors:
            dic=item_vectors[item_id]
            sum=0
            num=0
            #print(item_id)
            for user_id in dic:        
                num+=1
                sum+=dic[user_id]
            average=sum/num    
            item_average_rate[item_id]=average
            
        for user_id in user_vectors:
            dic=user_vectors[user_id]
            sum=0
            num=0
            #print(user_id)
            for item_id in dic:        
                num+=1
                sum+=dic[item_id]
            average=sum/num    
            user_average_rate[user_id]=average    
        
        del user_vectors
        
        #load item similarity    
        item_similarity={}
        item_similarity_list=np.load('file npy/item_similarity'+str(i+1)+'.npy').tolist()
        
        for h in range(len(item_similarity_list)):
            item_id=item_similarity_list[h][0]
            tuples=item_similarity_list[h][1]
            item_similarity[item_id]=tuples
        
        #tinh rmse
        rmse=0
        num_rmse=0
        
        num_zeros_item=0 #so item co sim voi moi item khac deu la 0
        num_unknow_item=0 # so item ko co trong tap train
        
        infile=codecs.open(file_test , 'r' , encoding='utf-8' )
        for line in infile :
            #print('new')
            line=line.split('::')
            user_id=int(line[0])
            item_id=int(line[1])
            rating=int(line[2])   
            predict_rate=0
            b_x=user_average_rate[user_id]-overall_mean_rating
            if item_id not in item_average_rate:
                num_unknow_item+=1
                predict_rate=user_average_rate[user_id]
                rmse+=((predict_rate-rating)*(predict_rate-rating))
                #rmse+=(math.fabs(predict_rate-rating))
                num_rmse+=1        
                continue
            b_i=item_average_rate[item_id]-overall_mean_rating
            b_x_i=overall_mean_rating+b_x+b_i
            
            tuples=item_similarity[item_id]
            current_k=0
            first=0
            second=0
            for h in range(len(tuples)):
                tle=tuples[h]
                iid=tle[0]
                sim=tle[1]
                if sim<=0: #quan trong
                    break
                if user_id in item_vectors[iid] :
                    current_k+=1
                    r_x_j=item_vectors[iid][user_id]
                    s_i_j=sim
                    b_j=item_average_rate[iid]-overall_mean_rating
                    b_x_j=overall_mean_rating+b_x+b_j
                    first+=s_i_j*(r_x_j-b_x_j)
                    second+=s_i_j
                    if current_k==k:
                        break
                else:
                    continue    
            #print(item_similarity[item_id]) 
           
            if second==0:
                predict_rate= user_average_rate[user_id] 
                num_zeros_item+=1
            else:    
                predict_rate=b_x_i+first/second
            rmse+=((predict_rate-rating)*(predict_rate-rating))
            #rmse+=(math.fabs(predict_rate-rating))
            num_rmse+=1
        
        rmse=(math.sqrt(rmse/num_rmse))
        #rmse=rmse/num_rmse
        SUM_RMSE+=rmse
        NUM_RMSE+=1  
        #rmse=rmse/num_rmse
    print('k = '+str(k))
    
    rmse_average=SUM_RMSE/NUM_RMSE
    print('rmse average : '+str(rmse_average)) 
    
    k+=1
    #print('num rmse : '+str(num_rmse))
    
    #print('num zeros item : '+str(num_zeros_item))
    
    #print('num unknow item : '+str(num_unknow_item))
             