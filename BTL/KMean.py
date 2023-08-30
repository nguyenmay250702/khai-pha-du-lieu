import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import silhouette_score

data= pd.read_csv('milk.csv')
X = data.drop('chat_luong', axis=1).copy()

#khởi tạo k tâm cụm ngẫu nhiên
k=2
Centroids = (X.sample(n=k))
print("- Tâm cụm khởi tạo: \n",Centroids)

#tính khoảng cách giữa 2 điểm
def khoang_cach(row_c, row_x):
    d = sqrt((row_c["ph"]-row_x["ph"])**2 + (row_c["nhiet_do"]-row_x["nhiet_do"])**2
           + (row_c["huong_vi"]-row_x["huong_vi"])**2 + (row_c["mui"]-row_x["mui"])**2
           + (row_c["chat_beo"]-row_x["chat_beo"])**2 + (row_c["mau"]-row_x["mau"])**2
           + (row_c["do_trong"]-row_x["do_trong"])**2)
    return d

#thuật toán k-mean
diff=1
j=0
lap = 1
while(diff!=0):
    print("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Lần lặp: ",lap)
    i=1
    for index1,row_c in Centroids.iterrows():   #1 row_c là 1 Series: chứa thông tin của 1 dòng dữ liệu
        ED=[]
        for index2,row_x in X.iterrows():
            d=khoang_cach(row_c, row_x)
            ED.append(d)
        X["d(C"+str(i)+")"]=ED
        i=i+1

    C=[]
    for index,row in X.iterrows():
        min_dist=row["d(C1)"]
        pos=1
        for i in range(k):
            if row["d(C"+str(i+1)+")"] < min_dist:
                min_dist=row["d(C"+str(i+1)+")"]
                pos=i+1
        C.append(pos)
        
    X["Cum"]=C
    #print("\n+ Dữ liệu X:\n",X)

    Centroids_new = X.groupby(["Cum"]).mean()[["ph","nhiet_do","huong_vi","mui","chat_beo","mau","do_trong"]]
    print("\n+ Centroids_new:\n",Centroids_new)
    print("\n+ Tâm cụm cũ: \n",Centroids)

    diff = (Centroids_new["ph"]-Centroids["ph"]).sum() + (Centroids_new["nhiet_do"]-Centroids["nhiet_do"]).sum()
    + (Centroids_new["huong_vi"]-Centroids["huong_vi"]).sum() + (Centroids_new["mui"]-Centroids["mui"]).sum()
    + (Centroids_new["chat_beo"]-Centroids["chat_beo"]).sum() + (Centroids_new["mau"]-Centroids["mau"]).sum()
    + (Centroids_new["do_trong"]-Centroids["do_trong"]).sum()
        
    if j==0:
        diff=1
        j=j+1

    print("\n+ diff = ",diff)
    lap+=1   
    Centroids = Centroids_new

print("\n- Tâm cụm cuối cùng:\n",Centroids)

#tính độ đo: ilhouette_score
print("\n- Mức độ phù hợp silhouette_score = ",silhouette_score(X.iloc[:,:7],X["Cum"]))


#dự đoán cụm trên 1 mẫu dữ liệu cụ thể
map1 = {"ph": 9, "nhiet_do": 1, "huong_vi": 23, "mui": 7.54, "chat_beo": 0, "mau": 220, "do_trong": 11.3}
def du_doan_cum(map_data):
    min_d = 9999999999
    C = 1
    for index1,row_c in Centroids.iterrows():
        d=khoang_cach(row_c, map_data)
        #print("\n- khoảng cách: ", d)
        if d < min_d:
            min_d = d
            C = index1
    return C

print("\n- Điểm map1 thuộc cụm: ", du_doan_cum(map1))
           
            









    
            
##def calculate_euclidean_distance(point1, point2):
##    return np.sqrt(np.sum((point1 - point2) ** 2))
##
##def calculate_silhouette_score(data, labels):
##    num_samples = len(data)
##    silhouette_scores = np.zeros(num_samples)
##    
##    for i in range(num_samples):
##        point = data[i]
##        cluster_label = labels[i]
##        
##        # Tính a(i) - trung bình khoảng cách từ điểm i tới các điểm trong cùng cluster
##        dist_in_cluster = []
##        for j in range(num_samples):
##            if labels[j] == cluster_label and i != j:
##                dist_in_cluster.append(calculate_euclidean_distance(point, data[j]))
##        a_i = np.mean(dist_in_cluster)
##        
##        # Tính b(i) - trung bình khoảng cách từ điểm i tới các điểm trong cluster khác
##        dist_other_clusters = []
##        for label in set(labels):
##            if label != cluster_label:
##                dist_to_other_cluster = []
##                for j in range(num_samples):
##                    if labels[j] == label:
##                        dist_to_other_cluster.append(calculate_euclidean_distance(point, data[j]))
##                dist_other_clusters.append(np.mean(dist_to_other_cluster))
##        b_i = np.min(dist_other_clusters)
##        
##        # Tính Silhouette score cho điểm i
##        silhouette_scores[i] = (b_i - a_i) / max(a_i, b_i)
##    
##    # Tính trung bình Silhouette score của tất cả các điểm
##    silhouette_avg = np.mean(silhouette_scores)
##    
##    return silhouette_avg
##
##print("\n- Silhouette:\n",calculate_silhouette_score(np.array(data.drop('chat_luong', axis=1)), np.array(X['Cum'])))
##
