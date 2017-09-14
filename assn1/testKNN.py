import numpy as np
import sys
from modshogun import LMNN, RealFeatures, MulticlassLabels
from sklearn.datasets import load_svmlight_file
import sklearn import neighbors
def euclidean_distance(x,x1):
    y = np.square(x-x1)
    return np.sqrt(np.sum(y))

def main(): 

    # Get training file name from the command line
    traindatafile = sys.argv[1]

	# The training file is in libSVM format
    tr_data = load_svmlight_file(traindatafile);

    Xtr = tr_data[0].toarray(); # Converts sparse matrices to dense
    Ytr = tr_data[1]; # The trainig labels
    print(Xtr.shape)
    print(Xtr)
    # Cast data to Shogun format to work with LMNN
    features = RealFeatures(Xtr.T)
    labels = MulticlassLabels(Ytr.astype(np.float64))
    print(labels)
        
    ### Do magic stuff here to learn the best metric you can ###
    # Number of target neighbours per example - tune this using validation
    k = 2
    neigh = KNeighborsClassifier(n_neighbors=k, algorithm='kd_tree', p=2)
    neigh.fit(Xtr, Ytr)
    # Save the model for use in testing phase
	# Warning: do not change this file name
    # np.save("model.npy", M) 

if __name__ == '__main__':
    main()
