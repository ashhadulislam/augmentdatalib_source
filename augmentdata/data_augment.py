import numpy as np
import random as rd
from imblearn.over_sampling import SMOTE 

# following libraries for classification test
# import test_nn
from . import test_nn

import statistics 
from statistics import mode 
import math


class DataAugment:





	def augment(self,**params):
		'''function creation: 5 inputs
		(data=df.values,k=k,class_ind=1,N=45000,randmx=randmx)
		data is the array like input of data		
		last column of data is class label
		k is number of neighbors, it should be bigger or equal to 1
		class_ind is the value of data that needs to be augmented
		for example, if the class labels are 0 or 1
		and the datapoints for 0 need to be upsampled then:
		class_ind=0
		N number of Datapoints that needs to be added
		randmx will be a value between 0 and 1, inclusive
		smaller the randmx, closer is the data to each original data		
		randmx, uniform[0,randmx], ; randmx<=1
		The outputs are
		Data_a: complete data with augmented datapoints
		Ext_d: Only the augmented data points
		Ext_not: The data that was created but ignored

		dist_percent: decides the threshold value for 
		permissible distance in order to use a point 
		for generating augmented values

		'''

		# extract parameters
		# set default first
		randmx=1
		blind=False
		verbose=False
		dist_percent=0.6
		

		if 'data' in params.keys():
			data=params["data"]
		if 'k' in params.keys():
			k=params['k']
		if 'class_ind' in params.keys():
			class_ind=params['class_ind']
		if 'N' in params.keys():
			N=params['N']
		if 'randmx' in params.keys():
			randmx=params['randmx']
		if 'blind' in params.keys():
			blind=params['blind']			
		if 'verbose' in params.keys():
			verbose=params['verbose']						
		if 'dist_percent' in params.keys():
			dist_percent=params['dist_percent']									


		
		

		# print("to check if data is correct type")
		
		is_numeric=np.issubdtype(data.dtype, np.number)
		if not is_numeric:
			print(data.dtype, " is not the correct data type for data, exiting")
			return None, None, None

		# print("Data shape is ",data.shape,"\n*****")		
		
		


		
		# now begin operation
		x=data[np.where(data[:,len(data[0])-1]==class_ind)]
		# print("x shape\n",x.shape,"\n*****")
		x=x[:,:len(data[0])-1]

		# x is the dataset for the minor class
		# print("x inside\n",x,"\n****")
		# print("x shape\n",x.shape,"\n*****")
		
		y=data[np.where(data[:,len(data[0])-1]!=class_ind)]
		other_class_index = y[:,len(y[0])-1];
		y=y[:,:len(data[0])-1]
		# print("y inside\n",y,"\n*****")
		# print("y shape\n",y.shape,"\n*****")
		# y is the dataset for majority class

		# calculate k neighbors for each point
		# d_k
		# sort them 
		# threshold is distance from 80th point


		# for each point we will have a list of distances
		# calculate distance of each point from
		# every other point
		dist_matrix=np.linalg.norm(x - x[:,None], axis=-1)
		# print("The distance matrix is \n",dist_matrix)
		dist_matrix.sort(axis=1)
		# print("the distance matrix, sorted is \n",dist_matrix)
		kth_distances=dist_matrix[:, (k-1)]
		# print("The kth neighbor distance array is \n",kth_distances)
		kth_distances_sorted=np.sort(kth_distances)
		# print("The old kth neighbor distance array is \n",kth_distances)
		# print("The sorted kth neighbor distance array is \n",kth_distances_sorted)
		threshold_dist=kth_distances_sorted[math.floor(dist_percent*len(kth_distances_sorted))]
		# print("The threshold distance is ",threshold_dist)

		# plt.plot([i for i in range(len(kth_distances))], kth_distances)
		# plt.show()
		


		Data=np.concatenate((x,y))
		# print("Data inside\n", Data,"\n*****")
		# data is just all features with minority class on top
		# followed by majority class features
		Ext_d=np.zeros(len(Data[0]))
		Ext_d=Ext_d.reshape(1,len(Data[0]))
		# print("Ext_d inside\n",Ext_d,"\n*****")
		# print("Ext_d shape\n",Ext_d.shape,"\n*****")
		# ext_d is a 1d array same size as the feature vector

		Ext_not = np.zeros(len(Data[0]))
		Ext_not=Ext_not.reshape(1,len(Data[0]))
		# print("Ext_not inside\n",Ext_not,"\n*****")
		# print("Ext_not shape\n",Ext_not.shape,"\n*****")
		# Ext_not is a 1d array same size as the feature vector

		psOUT=[]
		# print("N = ",N)
		Q=N//len(x)+1
		
		# len(x0) after removing the 20%

		# +1 will handle 0 cases
		# print("value of Q is ",Q)
		# print("length of x is ",len(x))
		# print("Data is ",Data)
		index_increament = len(x) * Q
		# print("Index increment is ",index_increament)
		while N>0:
			# print("N = ",N)
			# print("randmx = ",randmx)
			# print("index_increament = ",index_increament)
			# print("len(x) = ",len(x))
			# print("Q = ",Q)
			# print("Calculating randmx * index_increament / (len(x) * Q)")
			# print("randmx is going to be",randmx * index_increament / (len(x) * Q))
			randmx = randmx * index_increament / (len(x) * Q)
			
			## print("index_increament", index_increament, "N=", N, "X size=", len(x), "Q=", Q)
			index_increament = 0
			# print("randmx=",randmx)

			# print("Next loop runs ",len(x)," times")
			# this loops runs for all the features
			# in the minority set


			for i in range(len(x)):

				# if the distance of x_i
				# is less than the threshold
				if kth_distances[i]>threshold_dist:
					# print("More than threshold, skipping ",x[i])
					continue




				# print("i=",i)

				if N==0:
					break
				v = x[i,:]
				# extract the ith minority feature
				# basically taking each  minority feature at a time
				# print("v = \n",v,"\n******")

				val=np.sort( abs((x-v)*(x-v)).sum(axis=1) )
				# sorted list of distance of val from x
				# x being the minority class				
				# print("val = \n",val,"\n******")

				## print("val",val)
				posit=np.argsort(abs((x-v)*(x-v)).sum(axis=1))
				# print("posit = \n",posit,"\n******")
				# posit is the list of indices of sorted
				# distance array

				kv = x[posit[1:k+1],:]
				# skip the first element as that will be 0
				# then take all the k closest neighbors

				# print("kv = \n",kv,"\n******")
				# # print("Next loop runs ",Q," times")


				for kk in range(Q):
					# print("kk=",kk)
					m0 = v
					# print("m0 = \n",m0,"\n******")
					# the minority feature vector to start with
					# every minority feature vector
					# will get a chance, unless N == 0

					if N==0:
						break
					alphak = rd.uniform(0,randmx)
					for j in range(k):
						
						m1 = m0 + alphak * (kv[j,:] - m0)
						m0 = m1
						

					# print("m1 = ",m1)
					# above for loop is what sets the code apart
					# loop runs for the number of neighbors
					# to be considered
					# a uniform var

					# blind is the variable that user can tweak to decide
					# whether to go for sub classification or not
					if blind==False:


						# test to see if m1 belongs to the
						# required class through knn
						num_neighbors_to_test=math.floor(math.sqrt(k))
			
						# print("Number of neighbors to test with = ",num_neighbors_to_test)
						# this will be used now 
						# for testing the artificial data

						can_use=test_nn.predict_classification(data,m1, num_neighbors_to_test,class_ind)
						# print("Usability ",can_use)

					elif blind == True:
						can_use=True


					
					
					# here we are doing 1 NN to validate the 
					# artificial data that we just created
					# print("Validating m1 = ",m1)
					# test_val=(abs((Data-m1)*(Data-m1)).sum(axis=1))					
					# sqrt_val=np.sqrt(test_val)				
					# val=np.sort(np.sqrt( abs((Data-m1)*(Data-m1)).sum(axis=1) ))
					# posit=np.argsort(np.sqrt( abs((Data-m1)*(Data-m1)).sum(axis=1) ))

					




					if can_use:
					# if posit[0]<=len(x) or (randmx<10**(-4)):					
						if verbose:
							if N%5000==0:
								print(N)
						m1=m1.reshape(1,len(Data[0]))
						#print("m1  Ext_d shape",m1,m1.shape,Ext_d.shape)
						Ext_d = np.concatenate((Ext_d, m1))
						#print("Ext_d =", Ext_d )
						N=N-1
						index_increament=index_increament+1
						if N==0:
							break
					else:
						m1 = m1.reshape(1, len(Data[0]))
						# print("m1  Ext_d shape",m1,m1.shape,Ext_d.shape)
						Ext_not = np.concatenate((Ext_not, m1))


		Ext_d = Ext_d[1:len(Ext_d)]
		Ext_not = Ext_not[1:len(Ext_not)]
		#print("Ext_d final",Ext_d)

		x = np.concatenate((x,Ext_d))
		x = np.concatenate((x, class_ind +np.zeros((len(x), 1))), axis=1)
		other_class_index=other_class_index.reshape(len(other_class_index),1)
		#print(y.shape,other_class_index.shape)
		y = np.concatenate((y, other_class_index), axis=1)

		Data_a = np.concatenate((x,y))
		#print("randmx",randmx)
		return Data_a,Ext_d,Ext_not

# below code is used to test the code
def main():
	'''
	x^2 + y^2 + z^2 < 15 => 0
	x^2 + y^2 + z^2 >= 15 => 1
	'''
	l=[
	
	[1.0,2.0,1.0,0],
	[1,3,1,0],
	[2,1,1,0],
	[3,2,1,0],
	[3,1,1,0],

	[1,3,4,1],
	[1,4,3,1],
	[1,4,4,1],

	[2,3,3,1],
	[2,3,4,1],
	[2,4,3,1],
	[2,4,4,1],

	[3,2,2,1],
	[3,3,2,1],
	[3,3,2,1],
	[3,4,2,1],

	[4,3,1,1]
	]

	l=np.array(l)
	print("Original Data:")
	print(l)
	print("************************************")

	k=2
	randmx=1
	N=4
	daug=DataAugment()
	[Data_a,Ext_d,Ext_not]=daug.augment(data=l,k=k,class_ind=0,N=N,randmx=randmx)
	print(Data_a.shape)

	print("KNNOR Data:")
	print(Data_a)
	print("************************************")
	try_SMOTE(l,N,k)


def try_SMOTE(l,N,k):

	samp_strategy=float((5+N)/12)
	sm = SMOTE(random_state=2,k_neighbors=k,sampling_strategy=samp_strategy)
	y_train = l[:, -1] # for last column
	X_train = l[:, :-1] # for all but last column


	X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())
	
	
	rows,cols=X_train_res.shape
	y_train_res=(y_train_res.reshape(rows,1))
	data_smote=np.concatenate([X_train_res, y_train_res], axis=1)
	print("SMOTE Data:")
	print(data_smote)
	print("************************************")

if __name__=="__main__":
	main()
