from sklearn.svm import SVC
import pickle


#docs: http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
class SVM:
	def __init__(self, C=1.0,beta=0.0, kernel='linear', poly_degree=3, max_iter=-1, tol=0.001):
		#possible kernels: linear, 'poly', 'rbf', 'sigmoid', 'precomputed' or a callable

		#data features
		self.n_features = None
		self.train_X = []
		self.train_Y = []
		self.val_X = []
		self.val_Y = []
		self.test_X = []
		self.test_Y = []

		#classifier features
		self.C = C
		self.beta = beta
		self.kernel = kernel
		self.poly_degree = poly_degree
		self.max_iter = max_iter
		self.tolerance = tol

		self.classifier = None


	#SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
	#	gamma=0.0, kernel='rbf', max_iter=-1, probability=False,
	#	random_state=None, shrinking=True, tol=0.001, verbose=False)

	def setTrainData(self, X, Y):
		self.train_X = X
		self.train_Y = Y

		self.n_features = self.train_X.shape[1]

	def setTestData(self, X, Y):
		self.test_X = X
		self.test_Y = Y

	def setC(self, c):
		self.C = c

	def setBeta(self, beta):
		if beta is None:
			self.beta = 0.0
		else:
			self.beta=beta

	def setKernel(self, kernel, poly_degree=3):
		self.kernel = kernel
		self.poly_degree = poly_degree

	def setValData(self, X, Y):
		self.val_X = X
		self.val_Y = Y

	def train(self):
		self.classifier = SVC(C=self.C, kernel=self.kernel, gamma=self.beta, degree=self.poly_degree, max_iter=self.max_iter,
							  tol=self.tolerance)
		self.classifier.fit(self.train_X, self.train_Y)

	def predict(self, X):
		return self.classifier.predict(X)

	def getScore(self, X, Y):
		#returns accuracy
		return self.classifier.score(X, Y)

	def getNumSupportVectors(self):
		return self.classifier.n_support_

	def getHingeLoss(self,X,Y):
		preds = self.predict(X)
		hinge_inner_func = 1.0 - preds*Y
		hinge_inner_func = [max(0,x) for x in hinge_inner_func]
		return sum(hinge_inner_func)

	def saveClassifierToFile(self, filepath):
		s = pickle.dumps(self.classifier)
		f = open(filepath, 'wb')
		f.write(s)

	def loadClassifierFromFile(self, filepath):
		f2 = open(filepath, 'rb')
		s2 = f2.read()
		self.classifier = pickle.loads(s2)
