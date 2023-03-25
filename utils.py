import numpy as np
from math import sqrt


def euclidean(array_x, array_y):
	n = array_x.shape[0]
	ret = 0.
	for i in range(n):
		ret += (array_x[i]-array_y[i])**2
	return sqrt(ret)

def average_hausdorff_distance(xa,yb,p=1):
	na=xa.shape[0]
	nb=yb.shape[0]

	sum_a=0

	for i in range(na):
		cmin=np.inf
		for j in range(nb):
			d=euclidean(xa[i,:],yb[j,:])
			if d<cmin:
				cmin=d

		sum_a+=(cmin**p)

	gd_ab=((1.0/na)*sum_a)**(1/p)


	sum_b=0

	for i in range(nb):
		cmin=np.inf
		for j in range(na):
			d=euclidean(yb[i,:],xa[j,:])
			if d<cmin:
				cmin=d

		sum_b+=(cmin**p)

	igd_ab=(1.0/nb*sum_b)**(1/p)

	average_haus=max(gd_ab,igd_ab)

	return average_haus



