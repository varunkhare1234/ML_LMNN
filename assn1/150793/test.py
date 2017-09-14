from sklearn.datasets import load_svmlight_file
import numpy as np
import sys

def predict(Xtr, Ytr, Xts, yts,metric=None):

    N, D = Xtr.shape

    assert N == Ytr.shape[0], "Number of samples don't match"
    assert D == Xts.shape[1], "Train and test dimensions don't match"

    if metric is None:
        metric = np.identity(D)

    Yts = np.zeros((Xts.shape[0], 1))
    #the Cross Validated value for the parameter K 
    k=16

    for i in range(Xts.shape[0]):
        diff = Xtr - Xts[i]
#For nearest neighbors we dont actually need to calculate the square root
#this is the square of the distance calculated using the mahalanobis metric
        dist = (diff.dot(metric) * diff).sum(-1) 
#Indices capable of sorting the the distances        
        new = np.argsort(dist)
#sorting the distances according to the above indices        
        new = Ytr[new]
        new = new[0:k]
        z=np.zeros((4,1))
        #frequency of each label        
        for j in range(k):            
            if new[j]==1:
                z[1]+=1
            elif new[j]==2:
                z[2]+=1
            else :
                z[3]+=1
        if z[1]>=z[2]:
            if z[1]>=z[3]:
                Yts[i]=1
            else:
                Yts[i]=3
        else:
            if z[2]>=z[3]:
                Yts[i]=2
            else:
                Yts[i]=3
    return Yts

def main(): 

    # Get training and testing file names from the command line
    traindatafile = sys.argv[1]
    testdatafile = sys.argv[2]
	
    # The training file is in libSVM format
    tr_data = load_svmlight_file(traindatafile)

    Xtr = tr_data[0].toarray();
    Ytr = tr_data[1];
    # The testing file is in libSVM format too
    ts_data = load_svmlight_file(testdatafile)

    Xts = ts_data[0].toarray();
    yts = ts_data[1];
    yts = np.reshape(yts,(yts.shape[0],1))
    # The test labels are useless for prediction. They are only used for evaluation

    # Load the learned metric
    
    metric = np.load("model.npy")

    ### Do soemthing (if required) ###

    Yts = predict(Xtr, Ytr, Xts, metric)
    np.savetxt("testY.dat", Yts)
    #accuracy
    print (np.count_nonzero(Yts == yts)*100.0/(Xts.shape[0]*1.0))

if __name__ == '__main__':
    main()
