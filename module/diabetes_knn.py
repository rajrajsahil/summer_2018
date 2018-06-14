import numpy as np

class KNN(object):
    def __init__(self):
        pass
    def train(self,X,Y):
        self.X_train = X
        self.Y_train = Y
    def calculate_distance_l2(self,X_predict,K):
        X_train_square = np.sum(np.square(self.X_train),axis=1)
        X_test_square  = np.sum(np.square(X_predict),axis=1)
        test_train_dot = np.dot(X_predict,self.X_train.T)
        distance_matrix = np.ones(test_train_dot.shape,dtype=float)
        for i in range(X_test_square.shape[0]):
            distance_matrix[i,:] = X_test_square[i] + X_train_square[:]
        distance_matrix -= test_train_dot
        sorted_distance = np.array([np.argsort(row) for row in distance_matrix])
        sorted_distance = sorted_distance[:,0:K]
        closet_y = np.array([[self.Y_train[j] for j in x] for x in sorted_distance])
        closet_y = np.array([[int(i) for i in x] for x in closet_y])
        predicted_y = [np.argmax(np.bincount(x)) for x in closet_y]
        return predicted_y

        

        