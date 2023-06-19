import csv
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import distance
from collections import Counter

class p:
    def __init__(self,data,k,p_id):
            self.data = data
            self.k = k
            #initialize random mean values
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(data)
            self.mean_prime = np.zeros((self.k,len(self.data.iloc[0])))
            self.p_id = p_id
            
            
    def iteration(self):
        self.mean = self.mean_prime
        dists = []
        for i in range(len(self.data)):
            dist = []
            for j in range(self.k):
                dist.append(distance.euclidean(list(self.data.iloc[i]),self.mean[j]))
            dists.append(dist)
            
        self.distances = dists 
        return dists
    
    def receive_cluster_assignemt(self,clusters):
        new_mean = np.zeros((self.k,len(self.data.iloc[0])))
        for i in range(len(clusters)):
            new_mean[clusters[i]] += list(self.data.iloc[i]) 
        freq = Counter(clusters)
        for i in range(len(freq.keys())):
            new_mean[list(freq.keys())[i]] /= list(freq.values())[i]
        self.prev_mean = self.mean_prime
        self.mean_prime = new_mean
       
        dist = 0    
        for i in range(self.k):    
            dist += distance.euclidean(self.prev_mean[i],self.mean_prime[i])      
        return dist
        
    
                           
                              
