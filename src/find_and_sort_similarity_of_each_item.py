import numpy as np
import codecs
import math
import operator

def cosine_similarity(dic1,dic2):
    first=0
    len1=0
    len2=0
    num_mutual=0
    for key in dic1.keys():
        if key in dic2:
            num_mutual+=1
            x=dic1[key]
            len1+=(x*x)            
            y=dic2[key]
            len2+=(y*y)
            first+=x*y
            
    len1=math.sqrt(len1)  
    len2=math.sqrt(len2)
    
    if len1==0 or len2==0:
        return 0 
      
    return first/(len1*len2)+num_mutual*0.000000000000001    #cang co nhieu filed chung thi vector cang tuong tu hon   
        
n_movie=3,900

file_train='C:/hoc tap/big data/data/ml-1m/ml-1m/split/train3.txt'
 
item_vectors={}
item_average_rate={}
item_similarity={}
item_similarity_list=[]

infile=codecs.open(file_train , 'r' , encoding='utf-8' )
for line in infile :
    line=line.split('::')
    user_id=int(line[0])
    item_id=int(line[1])
    rating=int(line[2])    
    if item_id in item_vectors:
        item_vectors[item_id][user_id]=rating
    else:
        item_vectors[item_id]={}
        item_vectors[item_id][user_id]=rating
    
#chuan hoa cac vector item
for item_id in item_vectors:
    dic=item_vectors[item_id]
    sum=0
    num=0
    print(item_id)
    for user_id in dic:        
        num+=1
        sum+=dic[user_id]
    average=sum/num    
    item_average_rate[item_id]=average
    
for item_id in item_vectors:
    dic=item_vectors[item_id]
    average=item_average_rate[item_id]  
    for user_id in dic: 
        dic[user_id]=dic[user_id]-average 

#tinh toan do tuong dong cosine giua cac vector item         
for key in item_vectors:
    item_similarity[key]={}

for key1 in item_vectors:
    dic_rate1=item_vectors[key1]
    dic_similarity1=item_similarity[key1]
    print(key1)
    for key2 in item_vectors:    
        if (key2 != key1) and (key2 not in dic_similarity1) :
            #print(key2)
            dic_rate2=item_vectors[key2]
            dic_similarity2=item_similarity[key2]
            cos_val=cosine_similarity(dic_rate1, dic_rate2)
            dic_similarity1[key2]=cos_val
            dic_similarity2[key1]=cos_val

#sap xep cac  gia tri cosine theo thu tu giam dan cho moi item
#dong thoi chuyen dict thanh list de luu thanh file npy ben ngoai
for key in item_similarity:
    dic=item_similarity[key]
    sorted_tuple_list = sorted(dic.items(), key=operator.itemgetter(1),reverse=True)
    list=[key,sorted_tuple_list] 
    item_similarity_list.append(list) 

item_similarity_list=np.array(item_similarity_list,dtype=object)

print(item_similarity_list[1])

np.save('file npy/item_similarity1.npy',item_similarity_list)
             
            
            
            
            