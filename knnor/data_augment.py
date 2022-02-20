import numpy as np
import random
# from imblearn.over_sampling import SMOTE 

# following libraries for classification test
# import test_nn

import math


class KNNOR:


    # def knnor_over_sample(X,y,n_to_sample,num_neighbors,proportion,max_dist_point,intra=True):
    def fit_resample(self,X,y,**params):
        threshold_cannot_use=10

        # check for number of neighbors
        if 'num_neighbors' in params.keys():
            num_neighbors=params['num_neighbors']
        else:
            good_neighbor_count=self.good_count_neighbors(X,y)
            if good_neighbor_count<=1:
                print("Too few neighbors")
                return X,y
            num_neighbors=random.randrange(1,good_neighbor_count)


        if 'max_dist_point' in params.keys():
            max_dist_point=params['max_dist_point']
        else:
            max_dist_point=self.max_threshold_dist(X,y,num_neighbors)

        if 'proportion_minority' in params.keys():
            '''
            proportion of minority population to use
            '''
            proportion_minority=params['proportion_minority']
            inter=False
        else:
            proportion_intra=self.calculate_distance_threshold(X,y,num_neighbors,intra=False)
            proportion_minority=proportion_intra
            inter=True



        if not self.check_enough_minorities(X,y,num_neighbors):
            print("Too few minorities")
            return X,y

        if 'final_proportion' in params.keys():
            '''
            final minority pop = what percentage of majority pop
            '''
            final_proportion=params['final_proportion']
            
        else:
            final_proportion=1


        n_to_sample=self.calculate_count_to_add(X,y,final_proportion)

        original_n_neighbors=num_neighbors
        original_max_dist_point=max_dist_point    
        original_proportion=proportion_minority
        
        minority_label,minority_indices=self.get_minority_label_index(X,y)
        X_minority=X[minority_indices]
        y_minority=y[minority_indices]
        majority_indices=[]
        for i in range(0,y.shape[0]):
            if i not in minority_indices:
                majority_indices.append(i)
        print(len(majority_indices),len(minority_indices),y.shape)
        X_majority=X[majority_indices]
        y_majority=y[majority_indices]
        
        if not inter:
            internal_distance = np.linalg.norm(X_minority - X_minority[:,None], axis = -1)
            internal_distance = np.sort(internal_distance)
            knd=internal_distance[:,num_neighbors]        
            knd_sorted = np.sort(knd)        
        else:
            external_distance=np.linalg.norm(X_majority - X_minority[:,None], axis = -1)
            external_distance = np.sort(external_distance)
            knd=external_distance[:,num_neighbors]   
            knd_sorted=-np.sort(-knd)
            
        threshold_dist = knd_sorted[math.floor(proportion_minority*len(knd_sorted))]
            
        X_new_minority=[]
        N = n_to_sample
        consecutive_cannot_use=0
        while N>0:
            for i in range(X_minority.shape[0]):
                if inter:
                    if knd[i]>threshold_dist:
                        continue
                else:
                    if knd[i]<threshold_dist:
                        continue
                if N==0:
                    break
                v = X_minority[i,:]
                val=np.sort( abs((X_minority-v)*(X_minority-v)).sum(axis=1) )
                # sort neighbors by distance
                # obviously will have to ignore the 
                # first term as its a distance to iteself
                # which wil be 0
                posit=np.argsort(abs((X_minority-v)*(X_minority-v)).sum(axis=1))
                kv = X_minority[posit[1:num_neighbors+1],:]
                alphak = random.uniform(0,max_dist_point)
                m0 = v
                for j in range(num_neighbors):
                    m1 = m0 + alphak * (kv[j,:] - m0)
                    m0 = m1
                num_neighbors_to_test=math.floor(math.sqrt(num_neighbors))
                can_use=self.predict_classification(X,y,m0, num_neighbors_to_test,minority_label)
                can_use=can_use and not(self.check_duplicates(m0,X_minority))
                can_use=can_use and not(self.check_duplicates(m0,X_new_minority))                            
                if can_use:
                    consecutive_cannot_use=0
                    num_neighbors=min(num_neighbors+1,original_n_neighbors)
                    max_dist_point=min(max_dist_point+0.01,original_max_dist_point)
                    proportion_minority=max(proportion_minority-0.01,original_proportion)
                    threshold_dist = knd_sorted[math.floor(proportion_minority*len(knd_sorted))]                
                    X_new_minority.append(m0)
                    N-=1
                else:
                    consecutive_cannot_use+=1
                    if consecutive_cannot_use>=threshold_cannot_use:
                        num_neighbors=max(num_neighbors-1,2)
                        max_dist_point=max(max_dist_point-0.01,0.01)
                        proportion_minority=min(proportion_minority+0.01,0.9)
                        threshold_dist = knd_sorted[math.floor(proportion_minority*len(knd_sorted))]
                        consecutive_cannot_use=0

        y_new_minority=[minority_label for i in range(len(X_new_minority))]        
        X_new_minority=np.array(X_new_minority)
        X_new_all=np.concatenate((X, X_new_minority), axis=0)
        y_new_all=np.concatenate((y, y_new_minority), axis=0)
        
        return X_new_all, y_new_all, X_new_minority, y_new_minority






    def predict_classification(self,X,y,new_vector, num_neighbors_to_test,expected_class_index):
        '''
        this function is used to validate
        whether new point generated is close to
        same label points
        '''
        from sklearn.neighbors import KNeighborsClassifier
        posit=np.argsort(abs((X-new_vector)*(X-new_vector)).sum(axis=1))
        classes = y[posit[0:num_neighbors_to_test]]
        return np.sum(classes==expected_class_index)==classes.shape[0]

    def check_duplicates(self, new_row,old_rows):
        '''
        check if the new row
        is already preent in the old rows
        '''
        for row in old_rows:
            same=True
            for i in range(len(row)):
                if new_row[i]!=row[i]:
                    same=False
                    continue
            if same:
                return True                            
        return False

    def get_minority_label_index(self,X,y):
        '''
        find the minority label
        and the indices at which minority label
        is present
        '''
        # find the minority label
        uniq_labels=np.unique(y)
        # count for each label
        dic_nry={}

        for uniq_label in uniq_labels:
            dic_nry[uniq_label]=0

        for y_val in y:
            dic_nry[y_val]+=1

        # then which one is the minority label?
        minority_label=-1
        minimum_count=np.inf
        for k,v in dic_nry.items():
            if minimum_count>v:
                minimum_count=v
                minority_label=k


        # now get the indices of the minority labels
        minority_indices=[]
        for i in range(y.shape[0]):
            if y[i]==minority_label:
                minority_indices.append(i)

        return minority_label,minority_indices

    def good_count_neighbors(self,X,y):
        '''
        find the good number of neighbors to use
        this function is used on auto pilot
        '''
        minority_label,minority_indices=self.get_minority_label_index(X,y)
        X_minority=X[minority_indices]
        y_minority=y[minority_indices]
        count_greater=y_minority.shape[0]
        for i in range(X_minority.shape[0]):
            this_point_features=X_minority[i]
            dist = ((X_minority-this_point_features)*(X_minority-this_point_features)).sum(axis=1)
            mean_dist=np.mean(dist)
    #         print(dist,mean_dist)
            this_point_count_lesser = (dist < mean_dist).sum()
            count_greater=min(this_point_count_lesser,count_greater)        
        return count_greater





    # following function
    # to get the savitzky golay filter
    # https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter
    # https://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay
    # https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way

    def savitzky_golay(self,y, window_size, order, deriv=0, rate=1):         
        import numpy as np
        from math import factorial
        
        try:
            window_size = np.abs(np.int(window_size))
            order = np.abs(np.int(order))
        except ValueError:
            raise ValueError("window_size and order have to be of type int")
        if window_size % 2 != 1 or window_size < 1:
            raise TypeError("window_size size must be a positive odd number")
        if window_size < order + 2:
            raise TypeError("window_size is too small for the polynomials order")
        order_range = range(order+1)
        half_window = (window_size -1) // 2
        # precompute coefficients
        b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
        m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
        # pad the signal at the extremes with
        # values taken from the signal itself
        firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
        lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
        y = np.concatenate((firstvals, y, lastvals))
        return np.convolve( m[::-1], y, mode='valid')


    def calculate_distance_threshold(self,X,y,num_neighbors,intra=True):
        '''
        returns the distance threshold, based on the intra parameter
        if intra is chosen, returns the cut-off point for distances to
        kth nearest neighbor of same class
        in inter is chosen, returns the cut-off point for distances to 
        kth nearest neighbor of opposite class
        
        '''
        win_size=5 #positive odd number
        pol_order=2
        alpha=0.0001 # low value for denominator 0 case

        minority_label,minority_indices=self.get_minority_label_index(X,y)
        X_minority=X[minority_indices]
        y_minority=y[minority_indices]
        majority_indices=[]
        for i in range(0,y.shape[0]):
            if i not in minority_indices:
                majority_indices.append(i)
        X_majority=X[majority_indices]
        y_majority=y[majority_indices]
        
        if intra:
            internal_distance = np.linalg.norm(X_minority - X_minority[:,None], axis = -1)
            internal_distance = np.sort(internal_distance)
            knd=internal_distance[:,num_neighbors]
            
            knd_sorted = np.sort(knd)
            
            
        else:
            # need to calculate the distance to the kth nearest neighbor of
            # opposite class
            # more is good
            # sometimes the kth values are all same
            # in that case, proportion turns out to be 0
            external_distance=np.linalg.norm(X_majority - X_minority[:,None], axis = -1)
            external_distance = np.sort(external_distance)
            knd=external_distance[:,num_neighbors]   
            knd_sorted=-np.sort(-knd)

            
        # normalize it        
        normalized_dist= (knd_sorted-np.min(knd_sorted))/(np.max(knd_sorted)-np.min(knd_sorted)+alpha)

        # apply golay        
        normalized_dist = self.savitzky_golay(normalized_dist, win_size, pol_order) # window size 51, polynomial order 3
    #     plt.plot(normalized_dist)
    #     plt.title("NOrmalized distance intra"+str(intra))
    #     plt.show()
        normalized_dist=np.diff(normalized_dist)

        sin_values=np.abs(np.sin(np.arctan(normalized_dist)))
    #     plt.title("Sin differential - to get maxima intra"+str(intra))
    #     plt.plot(sin_values)
    #     plt.show()
        first_maxima_index=np.argmax(sin_values)
    #     print("Maxima is at ",first_maxima_index)
        proportion=first_maxima_index/sin_values.shape[0]
        return proportion
            
            
            
    # following function to calculate maximum
    # threshold distance
    # while placing a point
    def max_threshold_dist(self,X,y,num_neighbors):
        '''
        This function calculates the maximum distance between any two points in the minority class
        It also calculates the minimum distance between a point in the minority and a point
        in the majority class
        the value returned is the minimum of the two
        '''
        minority_label,minority_indices=self.get_minority_label_index(X,y)
        X_minority=X[minority_indices]
        y_minority=y[minority_indices]
        majority_indices=[]
        for i in range(0,y.shape[0]):
            if i not in minority_indices:
                majority_indices.append(i)
        print(len(majority_indices),len(minority_indices),y.shape)
        X_majority=X[majority_indices]
        y_majority=y[majority_indices]
        
        
        
        # calculate inter distance
        internal_distance = np.linalg.norm(X_minority - X_minority[:,None], axis = -1)
        internal_distance=internal_distance.flatten()
        max_internal_distance=np.max(internal_distance)
        
        # calculate the external distance
        external_distance=np.linalg.norm(X_majority - X_minority[:,None], axis = -1)
        external_distance=external_distance.flatten()
        # remove 0s just in case
        external_distance=external_distance[external_distance!=0]    
        min_external_distance=np.min(external_distance)
        
        max_allowed_distance=min(max_internal_distance,min_external_distance)/max(max_internal_distance,min_external_distance)
        
        return max_allowed_distance
        
        
        
    def check_enough_minorities(self,X,y,num_neighbors):
        '''
        ideally, the total number of minority points should be
        1 more than the total number of neighbors    
        '''
        minority_label,minority_indices=self.get_minority_label_index(X,y)
        if len(minority_indices)<=num_neighbors:
            print("You want to use ",num_neighbors,"neighbors, but minority data size = ",len(minority_indices))
            return False
        return True


    def calculate_count_to_add(self,X,y,final_proportion):
        '''
        Calculate the number of artificial points to be generated so that
        (count_minority_existing+count_artificial_minority)/count_majority_existing=final_proportion
        '''
        minority_label,minority_indices=self.get_minority_label_index(X,y)
        majority_indices=[]
        for i in range(0,y.shape[0]):
            if i not in minority_indices:
                majority_indices.append(i)
        count_minority=len(minority_indices)
        count_majority=len(majority_indices)
        new_minority=int((final_proportion*count_majority)-count_minority)
        if new_minority<1:
            return -1
        return new_minority









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
    X=l[:,:-1]
    y=l[:,-1]
    print("X=",X.shape,"y=",y.shape)
    print("Original Data:")
    print(l)
    print("************************************")

    
    knnor=KNNOR()
    X_new,y_new,_,_=knnor.fit_resample(X,y)
    y_new=y_new.reshape(-1,1)
    print(X_new.shape,y_new.shape)

    print("KNNOR Data:")
    new_data=np.append(X_new, y_new, axis=1)
    print(new_data)
    print("************************************")

    




if __name__=="__main__":
    main()
