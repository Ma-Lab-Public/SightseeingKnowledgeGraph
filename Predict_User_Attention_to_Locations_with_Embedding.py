import pickle
import numpy as np
import aesara
import pandas as pd
from aesara import tensor as Tr
from collections import OrderedDict
from sklearn import preprocessing


def NeuralNMF(X,S,T,
	weight_slope=0.0,
	sparsity_rate=0.8,
	k_v=25,
	k_r=20,
	numberofrelations=15,
	numberofneurals=20,
	Lambda_S=1.0,Lambda_T=1.0,Lambda_rs=1.0,
	lr=1e-5,
	itr=100,
	num_rows_to_sample=None):
	rng = np.random
	nonzero_list = np.nonzero(T)
	edges = list(zip(nonzero_list[0], nonzero_list[1]))

	#################observed parameter########################
	tX = aesara.shared(X.astype(aesara.config.floatX),name="X")
	tS = aesara.shared(S.astype(aesara.config.floatX),name="S")
	tT = aesara.shared(T.astype(aesara.config.floatX),name="T")

	#weight matrix for the X
	weightM = np.ones(X.shape)+weight_slope*X
	tW = aesara.shared(weightM.astype(aesara.config.floatX),name="W")
	###########################################################

	##################model parametrers########################
	numofusers = np.size(X, 0)
	numofgrids = np.size(X, 1)
	numofhours = np.size(S, 1)
	featurelen = k_v
	numberofedges = np.count_nonzero(T)
	U = rng.random((numofusers, featurelen)).astype(aesara.config.floatX)
	tU = aesara.shared(U,name="U")

	V = rng.random((numofgrids, k_v)).astype(aesara.config.floatX)
	tV = aesara.shared(V,name="V")

	E = rng.random((numofhours, k_v)).astype(aesara.config.floatX)
	tE = aesara.shared(E,name="E")

	I = rng.random((numberofedges, numberofrelations)).astype(aesara.config.floatX)
	tI = aesara.shared(I,name="I")

	R = rng.random((numberofrelations, k_r)).astype(aesara.config.floatX)
	tR = aesara.shared(R,name="R")

	Spar = (sparsity_rate/float(k_r))*np.ones((k_r,k_r))+(1.0-sparsity_rate)*np.identity(k_r)
	tSpar = aesara.shared(Spar.astype(aesara.config.floatX),name="Spar")

	W_neural = rng.random((numberofneurals, k_v*2+k_r)).astype(aesara.config.floatX)
	W_neural = W_neural*0.01
	tW_neural = aesara.shared(W_neural,name="W_neural")

	O_neural = rng.random((1, numberofneurals)).astype(aesara.config.floatX)
	O_neural = O_neural*0.01
	tO_neural = aesara.shared(O_neural,name="O_neural")
	###########################################################

	tLambda_S = Tr.scalar(name="Lambda_S")
	tLambda_T = Tr.scalar(name="Lambda_T")
	tLambda_rs = Tr.scalar(name="Lambda_rs")

	####################neural network related cost function#############
	all_R = Tr.dot(tI, Tr.dot(tR,tSpar))
	
	to_stack_Y = []
	for edge in edges:
		to_stack_Y.append(T[edge[0]][edge[1]])
	trY = np.array(to_stack_Y).reshape(len(edges),1)

	tC = Tr.concatenate((tV[nonzero_list[0],:],tV[nonzero_list[1],:],all_R), axis=1)
	
	h = Tr.sigmoid(Tr.dot(tC, tW_neural.T))
	y = Tr.dot(h, tO_neural.T)
    
	#print tC.shape.eval()
	#print trY.shape
	fun1 = ((trY-y)**2).sum()


	#######################NMF related cost function####################
	indexes=None
	first_item_1 = tX-Tr.dot(tU, tV.T)
	first_item_2 = tW*first_item_1

	if num_rows_to_sample is None:
		first_item = (first_item_2**2).sum()
	else:
		print(X.shape[1])
		indexes = np.random.choice(X.shape[1], num_rows_to_sample, replace=False)
		first_item = (first_item_2**2)[:, indexes].sum()

	second_item = ((tS-Tr.dot(tV, tE.T))**2).sum()*tLambda_S

	last_item = (tU**2).sum()+(tV**2).sum()+(tE**2).sum()+(tI**2).sum()+(tR**2).sum()
	############################################################


	#######################final cost function####################
	cost_function = first_item+second_item+fun1*tLambda_T+last_item*tLambda_rs
	print("set up the cost function")
	##############################################################


	#######################model training########################
	params = [tU,tV,tE,tI,tR, tW_neural, tO_neural]
	gU,gV,gE,gI,gR,gW_neural,gO_neural = Tr.grad(cost_function, params);

	print("define the learning function")
	train = aesara.function(inputs=[tLambda_S,tLambda_T,tLambda_rs],
							outputs=cost_function,
							allow_input_downcast=True,
							updates=OrderedDict({
							tU: Tr.clip((tU - lr * gU), 0.0, np.inf),
							tV: Tr.clip((tV - lr * gV), 0.0, np.inf),
							tE: Tr.clip((tE - lr * gE), 0.0, np.inf),
							tI: Tr.clip((tI - lr * gI), 0.0, np.inf),
							tR: Tr.clip((tR - lr * gR), 0.0, np.inf),
							tW_neural: (tW_neural - lr * gW_neural),
							tO_neural: (tO_neural - lr * gO_neural)
							}),
							name="train")

	print("begin training")
	re_U = tU.get_value()
	re_V = tV.get_value()
	re_E = tE.get_value()
	re_I = tI.get_value()
	re_R = tR.get_value()
	re_W_neural = tW_neural.get_value()
	re_O_neural = tO_neural.get_value()
	for i in range(0,itr):
		current_loss = train(Lambda_S, Lambda_T, Lambda_rs)
		if np.isnan(current_loss):
			break
		else:
			print(current_loss,i)
			re_U = tU.get_value()
			re_V = tV.get_value()
			re_E = tE.get_value()
			re_I = tI.get_value()
			re_R = tR.get_value()
			re_W_neural = tW_neural.get_value()
			re_O_neural = tO_neural.get_value()
	#############################################################

	return re_U,re_V,re_E,re_I,re_R,re_W_neural,re_O_neural,indexes

def Load_data(normalized=True):
	X = None #observed user attention
	S = None #observed time series
	T = None #observed transition matrix

	# f_X = "plkfiles/X.plk" 
	# f_S = "plkfiles/S.plk"
	# f_T = "plkfiles/T.plk"
	f_X = "plkfiles/X_Gmap.plk" 
	f_S = "plkfiles/S_Gmap.plk"
	f_T = "plkfiles/T_Gmap.plk"
	with open(f_X, 'rb') as input1:
		X = pickle.load(input1, encoding='latin1')

	with open(f_S, 'rb') as input2:
		S = pickle.load(input2, encoding='latin1')

	with open(f_T, 'rb') as input3:
		T = pickle.load(input3, encoding='latin1')
	
	# df = pd.DataFrame(X)
	# df.to_csv(r'X.csv')
	# df = pd.DataFrame(S)
	# df.to_csv(r'S.csv')
	# df = pd.DataFrame(T)
	# df.to_csv(r'T.csv')
	
	if normalized:
		X = preprocessing.normalize(X)
		S = preprocessing.normalize(S)
		T = preprocessing.normalize(T)
	
	print(X.shape, X.mean())
	print(S.shape, S.mean())
	print(T.shape, T.mean())
	return X,S,T

def Recall_based_evaluation(observed_X, latent_U, latent_V, observed_indexes):
	#calculate the recall for all user groups, and 
	#return the average recalls@Ms
	predicted_X = np.dot(latent_U, latent_V.T)
	if observed_indexes is not None:
		predicted_X = np.delete(predicted_X, observed_indexes, 1)
		observed_X = np.delete(observed_X, observed_indexes, 1)

	Ms = [5, 10, 15, 20, 25, 30]#, 35, 40, 45, 50, 55, 
		#60, 65, 70, 75, 80, 85, 90, 95, 100]
	records = []
	for m in Ms:
		users_scores = []
		for i in range(len(observed_X)):
			relevant_grids = (observed_X[i]>0.0).sum()
			if relevant_grids>0.0:
				predicted_relevant_grids = 0.0
				rankings = sorted(list(zip(predicted_X[i], observed_X[i])), reverse=True)
				for rank in rankings[0:m]:
					match = 0.0
					if rank[1] > 0.0:
						match = 1.0
					predicted_relevant_grids = predicted_relevant_grids+match
				users_scores.append(predicted_relevant_grids/relevant_grids)
				# print(relevant_grids)
			else:
				continue
		records.append(np.mean(np.array(users_scores)))
	# print(records)
	return records


def test_NeuralNMF():
	X,S,T = Load_data()
	num_rows_to_sample = 10 #dense setting, i.e., x=30% \times 2500=750
	Lambda_S=1
	Lambda_T=100
	Lambda_rs=1
	lr=1e-5
	itr=50
	numberofrelations = 30
	k_v = 10
	k_r = 5

	total_recall = np.zeros((1,6))
	rounds = 3
	for i in range(rounds):
		U,V,E,I,R,W_neural,O_neural,indexes=NeuralNMF(X,S,T,
						k_v=k_v, k_r=k_r, numberofrelations=numberofrelations,
						Lambda_S=Lambda_S,Lambda_T=Lambda_T,Lambda_rs=Lambda_rs,
						lr=lr,itr=itr,
						num_rows_to_sample=num_rows_to_sample)
		# print("U: ", U.shape)
		# print("V: ", V.shape)
		with open("plkfiles/U_bert.plk", 'rb') as input1:
			U_b = pickle.load(input1, encoding='latin1')
		with open("plkfiles/V_bert.plk", 'rb') as input1:
			V_b = pickle.load(input1, encoding='latin1')
		U_n = np.add(U, U_b)
		V_n = np.add(V, V_b)
		recalls = Recall_based_evaluation(X, U, V, indexes)
		print(recalls)
		total_recall = total_recall+np.array(recalls)

	recalls = total_recall/float(rounds)
	print(recalls.tolist())

if __name__=="__main__":
	print("#######Neural Network Matrix Factorization#######")
	test_NeuralNMF()