# Flores-Saiffe Farías Adolfo
# Universidad de Guadalajara
# adolfo.flores.saiffe@gmail.com
# saiffe.adolfo@alumnos.udg.mx
# ORCID:0000-0003-4504-7251
# Researchgate: https://www.researchgate.net/profile/Adolfo_Flores_Saiffe2
# Github: https://github.com/fitobawino
#
# This program loads artificial neural network architectures from ANN_fMRIs_utils applied to fMRI-ROI based time-series 
#
# 1. Create database
# 2. Load data and data augmentation
# 3. Set experiment parameters
# 4. Run experiment


######################
# 1. Create database #
######################
import ANN_fMRIs_utils as ANN
from sklearn.preprocessing import StandardScaler
from os.path import join as opj
import numpy as np
from sklearn.preprocessing import MinMaxScaler

time_length = 192
subj_no     = 32
feature_no  = 567
loading_dir = '/...'
load_folder = opj(loading_dir, 's31_BOLD_par')
working_dir = "/.../NNResults"
subjs       = os.listdir(load_folder)
# Sort by last letter: S01C, S02P, ...
subjs       = sorted(subjs, key=lambda x:x[-1])

x_data = np.empty([subj_no, feature_no, time_length], dtype=float)
y_data = np.empty([subj_no])
for subj in range(0, len(subjs)):
	x_data[subj] = np.loadtxt(opj(load_folder,subjs[subj],'me_BOLD_par_t.csv'), delimiter=",")
	if subjs[subj][-1] == 'C': y_data[subj] = 0
	else: y_data[subj] = 1

#Change output as categorical
y_data = np_utils.to_categorical(y_data)

# Eliminate features with only zeros among all times and subjects
x_data = x_data[:,x_data[:,:,:].sum(axis=2).sum(axis=0)!=0,:]

#Scale data with min-max or standard
scaler = MinMaxScaler(feature_range=(-1, 1)) 
#scaler = StandardScaler()

x_data = [scaler.fit_transform(x.transpose(1,0)) for x in x_data[:]]
x_data = np.asarray(x_data).transpose(0,2,1)
ANN.export_to_h5(x_data, y_data, opj(working_dir, 'testing-erasethis'))


######################################
# 2. Load data and data augmentation #
######################################
import ANN_fMRIs_utils as ANN
from os.path import join as opj
import numpy as np
import itertools
from keras.callbacks import TensorBoard, EarlyStopping
from time import time


x_data, y_data = ANN.import_from_h5("/../working_memory_db.hdf5")
time_split = 4 # Number of cuts to make to columns.
x_data, y_data = ANN.data_augmentation(x_data, y_data, time_split)
x_data = x_data.transpose(0,2,1) # Optional: arrange your data accordingly to LSTMs (samples, time, features)
samples, time_len, ROIn  = x_data.shape


################################
# 3. Set experiment parameters #
################################
# Iterable parameters
modelnames = ["baseline_model"] # Architectures to use. Options are: ["baseline_model", "model_bidirectional", "model_deepdense", "model_nodense", "model_deepLSTM"]
LSTMouts   = [32, 64, 128, 400]
time_lens  = [48] # Cut the rows to a specified length. If you don't want a cut, just put the var 'time_len' 
drops      = [0.5] # Relative number of the dropout
L12_regs   = [0.01] # L1 and L2 regularization factor

# K-fold cross validation split
np.random.seed(7) # fix random seed
laps        = 8    # K-fold cross-validation slices
cross_verif = np.array(list(range(0,laps)) * int(len(x_data)/laps/2))
np.random.shuffle(cross_verif)
cross_verif = np.append(cross_verif, cross_verif)
#Save the scores from all of the iterations: LossAVG, AccAVG, LossSTD, AccSTD
sum_scores  = np.zeros([len(modelnames)*len(LSTMouts)*len(time_lens)*len(drops)*len(L12_regs), 4])


#####################
# 4. Run experiment #
#####################
working_dir = "/.../NNResults"
it = 0
for L12_reg, drop, time_len, LSTM_size, modelname in itertools.product(L12_regs, drops, time_lens, LSTMouts, modelnames):
	print("Model_name:   {0}\nLSTM_size:    {1}\nTime_length:  {2}\nDropout:      {3}\nL12_reg:      {4}\nAugmentation: {5}\n".format(modelname, LSTM_size, time_len, drop, L12_reg, laps))
	scores = np.zeros([laps, 2])
	lap = 0
	while lap < laps:
		print("K = " + str(lap + 1))
		if modelname == "baseline_model":      model = ANN.baseline_model(LSTM_size, time_len, ROIn, drop, L12_reg)
		if modelname == "model_bidirectional": model = ANN.model_bidirectional(LSTM_size, time_len, ROIn, drop, L12_reg)
		if modelname == "model_nodense":       model = ANN.model_nodense(LSTM_size, time_len, ROIn, drop, L12_reg)
		if modelname == "model_deepdense":     model = ANN.model_deepdense(LSTM_size, time_len, ROIn, drop, L12_reg)
		if modelname == "model_deepLSTM":      model = ANN.model_deepLSTM(LSTM_size, time_len, ROIn, drop, L12_reg)
		# Tensor board callback
		tbCallBack = TensorBoard(log_dir=opj(working_dir,"Graph_"+modelname+"_lap"+str(lap)+"/{}".format(time())),
			histogram_freq=0, 
			write_graph=True, 
			write_images=True)
		# Early stopping callback
		stop = EarlyStopping(monitor='loss', 
			patience = 30, 
			min_delta=0.01, 
			verbose=0)
		# Train!
		print("<<< Training model >>>")
		history = model.fit(x_data[~(cross_verif == lap),:time_len, :], y_data[~(cross_verif == lap)],
			validation_data=(x_data[cross_verif == lap,:time_len, :], y_data[cross_verif == lap]),
			callbacks=[stop, tbCallBack],
			epochs=300,
			shuffle=True,
			batch_size=int(32 * time_split * (laps - 1) / laps),
			verbose=0) #validation_split=0.1
		# Evaluate!
		print("<<< Evaluating model >>>\n")
		scores[lap] = model.evaluate(x_data[cross_verif == lap,:time_len, :], y_data[cross_verif == lap], 
			verbose=0)		
		if ~np.isnan(scores[lap,0]): lap += 1 #if diverge, try again
		#print(scores)
		ANN.save_and_plot(history.history, str(it), modelname, True, working_dir)
	model.summary()
	print("\nLoss     |   Accuracy")
	print(scores)	
	sum_scores[it,:]=np.array([np.append(scores.mean(axis=0), np.std(scores, axis=0)).reshape(1,4)])
	it += 1
	print("\nLossAVG | AccAVG | LossSTD | AccSTD")
	print(sum_scores)
	np.savetxt(opj("scores_2_" + modelname + ".csv"), scores, delimiter='\t')
np.savetxt(opj("means_stds_2" + modelname + ".csv"), sum_scores[1:], delimiter='\t')

