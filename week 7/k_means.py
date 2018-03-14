import numpy as np

class KMeans(object):
  """ a K-means clustering with L2 distance """

  def __init__(self, num_clusters=3):
    self.num_clusters = num_clusters
    self.centroids = np.random.randn(self.num_clusters)

  def train(self, X, epsilon=1e-12):
    """
    Train the k-means clustering.

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - epsilon: (float) lower limit to stop cluster.
    """
    # save the change of centroids position after updating
    change = np.ones(self.num_clusters)
    self.centroids = X[np.random.choice(X.shape[0], self.num_clusters)]
    # it = 0
    while (True):
        # print(it)
        # print(self.centroids)
        # it+=1
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all cluster       #
        # centroid then assign them to the nearest centroid.                    #
        #########################################################################
        dists = self.compute_distances_no_loops(X)
        nearest_centroids = np.argmin(dists, axis=1)
        # print(nearest_centroids)
        clusters = {}
        for i in range(self.num_clusters):
          if(X[nearest_centroids==i].shape[0] > 0):
            clusters[i] = np.concatenate([x for x in X[nearest_centroids==i]]).reshape(-1, X.shape[1])
        # print(clusters)
        #########################################################################
        # TODO:                                                                 #
        # After assigning data to the nearest centroid, recompute the centroids #
        # then calculate the differrent between old centroids and the new one   #
        #########################################################################
        new_centroids = np.copy(self.centroids)
        for cen_indx in clusters:
          new_centroids[cen_indx] = np.mean(clusters[cen_indx], axis=0)
        for i in range(self.num_clusters):
          change[i] = np.linalg.norm(new_centroids[i] - self.centroids[i])
        self.centroids = new_centroids
        # print(change)
        isContinue = False
        for diff in change:
          if(diff > epsilon):
            isContinue = True
        if(not isContinue): break
        #########################################################################
        #                         END OF YOUR CODE                              #
        #########################################################################
        
  def predict(self, X, num_loops=0):
    """
    Predict labels for test data using this clustering.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - num_loops: Determines which implementation to use to compute distances
      between cluster centroids and testing points.
    Returns:
    - y: (A numpy array of shape (num_test,) containing predicted clusters for the
      test data, where y[i] is the predicted clusters for the test point X[i]).  
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value {} for num_loops'.format(num_loops))

    return self.predict_clusters(dists)

  def compute_distances_two_loops(self, X):
    """
    Compute the distance between each test point in X and each cluster centroid point
    in self.centroids using a nested loop over both the cluster centroids and the 
    test data.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_clusters) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth cluster centroid.
    """
    num_test = X.shape[0]
    dists = np.zeros((num_test, self.num_clusters))
    for i in range(num_test):
      for j in range(self.num_clusters):
        #####################################################################
        # TODO:                                                             #
        # Compute the l2 distance between the ith test point and the jth    #
        # cluster centroid, and store the result in dists[i, j]. You should #
        # not use a loop over dimension.                                    #
        #####################################################################
        dists[i][j] = np.linalg.norm(X[i] - self.centroids[j])
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
    return dists

  def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each cluster centroid
    in self.centroids using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    dists = np.zeros((num_test, self.num_clusters))
    for i in range(num_test):
      #######################################################################
      # TODO:                                                               #
      # Compute the l2 distance between the ith test point and all cluster  #
      # centroids, and store the result in dists[i, :].                     #
      #######################################################################
      dists[i] = np.linalg.norm(self.centroids - X[i], axis=1)
      #######################################################################
      #                         END OF YOUR CODE                            #
      #######################################################################
    return dists

  def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each cluster centroid
    in self.centroids using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    dists = np.zeros((num_test, self.num_clusters)) 
    # print(X.shape)
    # print(self.centroids.shape)
    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all cluster       #
    # centroid without using any explicit loops, and store the result in    #
    # dists.                                                                #
    #                                                                       #
    # You should implement this function using only basic array operations; #
    # in particular you should not use functions from scipy.                #
    #                                                                       #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################
    dists = np.sqrt(-2*np.matmul(X, self.centroids.transpose()) + \
            np.sum(X**2, axis=1).reshape(-1, 1) + \
            np.sum(self.centroids**2, axis=1))
    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists

  def predict_labels(self, dists):
    """
    Given a matrix of distances between test points and cluster centroids,
    predict a cluster for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_clusters) where dists[i, j]
      gives the distance betwen the ith test point and the jth cluster centroid.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted cluster for the
      test data, where y[i] is the predicted cluster for the test point X[i].  
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in range(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      #########################################################################
      # TODO:                                                                 #
      # Use the distance matrix to find the nearest cluster of the ith        #
      # testing point.                                                        #
      # Hint: Look up the function numpy.argsort.                             #
      #########################################################################
      y_pred[i] = np.argmin(dists[i])
      #########################################################################
      #                           END OF YOUR CODE                            # 
      #########################################################################

    return y_pred

