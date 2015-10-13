import cv2
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import RandomizedPCA
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from canny_edges import edges


def get_features(im):
	#hist = cv2.calcHist([im], [0], None, [256], [0,256])
	hist = edges(im)
	return hist.flatten()

def transform(files):
	res = []
	for path in files:
		im = cv2.imread(path)
		im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		res.append(get_features(im))
	return res

if __name__ == "__main__":
	df = pd.read_csv("data.csv", header=None)

	X = transform(df[0])
	y = df[1]

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

	#pca = RandomizedPCA(300).fit(X_train)
	#X_train_pca = pca.transform(X_train)
	#X_test_pca = pca.transform(X_test)

	#clf = SVC()
	clf = LogisticRegression()
	clf.fit(X_train, y_train)

	y_pred = clf.predict(X_train)
	print(classification_report(y_train, y_pred))

	y_pred = clf.predict(X_test)
	print(classification_report(y_test, y_pred))

	if all([x == y_pred[0] for x in y_pred]):
		print("FUUUUUUUUUUUUCCCCCCCCCCCCKKKKKKK!")
		print(y_pred[0])
