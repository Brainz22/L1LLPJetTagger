from numpy import loadtxt
from numpy import expand_dims
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import sys, os, numpy

import tensorflow
from sklearn.model_selection import train_test_split

from qkeras.utils import load_qmodel
from sklearn.preprocessing import MinMaxScaler

#os.environ['PATH'] = '/data/software/xilinx/Vivado/2020.1/bin:' + os.environ['PATH']
#BACKEND = "Vivado"
os.environ['PATH'] = '/data/software/xilinx/Vitis_HLS/2023.2/bin/' + os.environ['PATH']
BACKEND = "Vitis"

model = load_qmodel("noNorm_train_qkL1JetTagModel.h5")

model.compile(loss=tensorflow.keras.losses.BinaryCrossentropy(from_logits=True, name="binary_crossentropy"), 
                optimizer="adam", 
                metrics=["binary_accuracy"],
                weighted_metrics=[tensorflow.keras.metrics.AUC(name="auc")])

#Convert model to HLS
import hls4ml
config = hls4ml.utils.config_from_keras_model(model, 
                                            granularity='name', 
                                            backend=BACKEND,
                                            default_precision = 'fixed<14,8, AP_TRN, AP_SAT>')
print("-----------------------------------")

print("\n")
print(config)
print("\n")
print("---------------------------------")


config["LayerName"]["input_1"]["Precision"] = "fixed<12,6,AP_TRN, AP_SAT>"

config["LayerName"]["q_input"]["Precision"] = "fixed<12,6, AP_TRN, AP_SAT>"

config['LayerName']['q_conv1d']['ReuseFactor'] = 1
config['LayerName']['q_conv1d']["Precision"]["accum"] = "fixed<14,8, AP_TRN, AP_SAT>"
config['LayerName']['q_conv1d']["Precision"]["result"] = "fixed<14,8, AP_TRN, AP_SAT>"


config["LayerName"]["q_activation"]["Precision"]["result"] = "ufixed<14,8, AP_TRN, AP_SAT>"

config['LayerName']['q_conv1d_1']['ReuseFactor'] = 1
config['LayerName']['q_conv1d_1']["Precision"]["accum"] = "fixed<16, 12, AP_TRN, AP_SAT>"
config['LayerName']['q_conv1d_1']["Precision"]["result"] = "fixed<14, 8, AP_TRN, AP_SAT>"

config["LayerName"]["q_activation_1"]["Precision"]["result"] = "ufixed<10,5, AP_TRN, AP_SAT>"


config["LayerName"]["global_average_pooling1d"]["Precision"]["accum"] = "fixed<16,12, AP_TRN, AP_SAT>"
config["LayerName"]["global_average_pooling1d"]["Precision"]["result"] = "fixed<14,8, AP_TRN, AP_SAT>"


config['LayerName']['q_dense']["Precision"]["accum"] = "fixed<14, 10, AP_TRN, AP_SAT>"
config['LayerName']['q_dense']["Precision"]["result"] = "fixed<14, 8, AP_TRN, AP_SAT>"



config["LayerName"]["q_activation_2"]["Precision"]["result"] = "ufixed<14, 8, AP_TRN, AP_SAT>"


config['LayerName']['q_dense_1']["Precision"]["accum"] = "fixed<14, 10, AP_TRN, AP_SAT>"
config['LayerName']['q_dense_1']["Precision"]["result"] = "fixed<14, 8, AP_TRN, AP_SAT>"


#For Tracing
for layer in config['LayerName'].keys():
    print('Enable tracing for layer:', layer)
    config['LayerName'][layer]['Trace'] = True

hls_model = hls4ml.converters.convert_from_keras_model(model,
                                                       hls_config=config,
                                                       output_dir='qkmodel/hls4ml_prj',
                                                       part='xcvu13p-flga2577-2-e',)
                                                       #bit_exact=True)

hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file=os.getcwd() + "qkmodel.png")

#Compile model, no need to convert if we are plotting performance
hls_model.compile()

# Handle Data: 
#with h5py.File("/home/users/russelld/L1JetTagDaniel/hls4mlModifications/10-08-23/02-02_datasets/4b/M_LLP_30_ctau_10/newTestDatapt20_vDter_Signal_Only.h5", "r") as hf:
#with h5py.File("/home/users/russelld/L1JetTagDaniel/hls4mlModifications/10-08-23/02-02_Scripts/newTestDataST30.h5", "r") as hf:
with h5py.File("/home/users/russelld/TOOLLIP_TESTS/cmssw-tests/clean_SCRAM/CMSSW_15_1_0_pre4/src/L1LLPJetTag/data/test_merged/phi15_uuuu_merged_testPart.h5", "r") as hf:
    dataset = hf["jet_constituents"][:]
dataset = dataset[:, 0:141]
with h5py.File("/home/users/russelld/TOOLLIP_TESTS/cmssw-tests/clean_SCRAM/CMSSW_15_1_0_pre4/src/L1LLPJetTag/data/QCD_Pt15To3000_Flat_PU200/Bkg_test.h5", "r") as hf:
    datasetQCD = hf["jet_constituents"][:]
with h5py.File("/home/users/russelld/TOOLLIP_TESTS/cmssw-tests/clean_SCRAM/CMSSW_15_1_0_pre4/src/L1LLPJetTag/data/test_merged/phi15_uuuu_merged_testJet.h5", "r") as hf:
    jetDataSig = hf["test_jet_data"][:]
with h5py.File("/home/users/russelld/TOOLLIP_TESTS/cmssw-tests/clean_SCRAM/CMSSW_15_1_0_pre4/src/L1LLPJetTag/data/QCD_Pt15To3000_Flat_PU200/Bkg_testJets.h5", "r") as hf:
    jetDataQCD = hf["test_jet_data"][:]
    
dataset = np.concatenate((dataset,datasetQCD)) #Stacking datasets on top of another
jetData = np.concatenate((jetDataSig,jetDataQCD))
fullData = np.concatenate((dataset, jetData), axis=1)
np.random.shuffle(fullData) #shuffling rows
dataset = fullData[0:,0:141]
jetData = fullData[0:,141:]
   
N_PART_PER_JET = 10
N_FEAT = 14
A = dataset[:, 0 : len(dataset[0]) - 1]
b = dataset[:, len(dataset[0]) - 1]
#A = expand_dims(A, axis=3)
A = A.reshape((A.shape[0], N_PART_PER_JET, N_FEAT))

#plot kinematics
from plotting.kinematics_plotter import kinematics
#Normalization of impact parameter
normalizeIPs = False # Knob to say if I want to normalize IPs.

if max(A[:, :, 8].ravel()) < 2.0:
    print("\nImpact parameter was normalized beforehand.\n")
    norm_b4 = True
else:
    print("\nImpact parameter was not normalized beforehand.\n")
    norm_b4 = False

if norm_b4:
    print("\nImpact parameter was normalized beforehand.\n")
else:
    print("\nDecided not to normalize impact parameter. \n")
    tag = "noNorm/noNorm_test"
    kinematics(A, jetData, b, "stop_4b_4c", "noNorm/noNorm_test" )


from sklearn.metrics import roc_curve
from sklearn.metrics import auc

import matplotlib.pyplot as plt

import importlib.util, sys

spec = importlib.util.spec_from_file_location(
    "inputFixer",
    "/home/users/russelld/TOOLLIP_TESTS/cmssw-tests/clean_SCRAM/CMSSW_15_1_0_pre4/src/L1LLPJetTag/scripts/model/inputFixer.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

add_ip = mod.add_ip

#prediction with both models
A = add_ip(A)
print("Testing inputs shape: ", A.shape)

X_test = np.ascontiguousarray(A)


Ab_pred_qkeras = model.predict(A).ravel()
Ab_pred_hls_qkeras = hls_model.predict(X_test).ravel()

fpr_Ab_qkeras, tpr_Ab_qkeras, thresholds_Ab_qkeras = roc_curve(b, Ab_pred_qkeras)
auc_Ab_qkeras = auc(fpr_Ab_qkeras, tpr_Ab_qkeras)

fpr_Ab_hls, tpr_Ab_hls, thresholds_Ab_hls = roc_curve(b, Ab_pred_hls_qkeras)
auc_Ab_hls = auc(fpr_Ab_hls, tpr_Ab_hls)


#plt.plot(fpr_Ab_qkeras, tpr_Ab_qkeras, label=" qkeras AUC={:.3f}, M_LLP_30_ctau_10".format(auc_Ab_qkeras))
plt.figure()
plt.plot(fpr_Ab_qkeras, tpr_Ab_qkeras, label=" qkeras AUC={:.3f}".format(auc_Ab_qkeras))
plt.plot(fpr_Ab_hls, tpr_Ab_hls, "--" ,label=" HLS AUC={:.3f}".format(auc_Ab_hls))


plt.xlabel("Background Efficiency", fontsize=16)
plt.ylabel("Signal Efficiency", fontsize=16)
#plt.axvline(x=0.01, ymin=0, ymax=0.59, color="red")
#plt.axhline(y=0.6, xmin=0, xmax=0.573, color="red")
plt.title("L1 LLP Tag Qk Model ROC Curve", fontsize=16, weight="bold")
plt.legend(loc="best")
plt.xscale("log")
plt.grid(True)
#plt.savefig("HLS_qk_ROCCurve.pdf")
plt.savefig("HLS_qk_ROCCurve.pdf")


#Layer Tracing
#exit()
import hls4ml.model.profiling


y_hls, hls4ml_trace = hls_model.trace(X_test)
keras_trace = hls4ml.model.profiling.get_ymodel_keras(model, X_test)

for LAYER in hls4ml_trace.keys():
    if (LAYER == 'q_conv1d_linear') | (LAYER == 'q_conv1d_1_linear') \
     | (LAYER == 'q_dense_linear') | (LAYER == 'q_dense_1_linear') :
        continue
    plt.figure()
    plt.scatter(hls4ml_trace[LAYER].flatten(), keras_trace[LAYER].flatten())
    min_x = min(np.amin(hls4ml_trace[LAYER]), np.amin(keras_trace[LAYER]))
    max_x = max(np.amax(hls4ml_trace[LAYER]), np.amax(keras_trace[LAYER]))
    plt.plot([min_x, max_x], [min_x, max_x], c='gray')
    plt.xlabel('hls4ml {}'.format(LAYER))
    #plt.xlabel('hls4ml {}'.format(LAYER))
    plt.ylabel('QKeras {}'.format(LAYER))
    print(os.getcwd() + f'/LayerTraces/profiling_{LAYER}.png')
    plt.savefig(os.getcwd() + f'/LayerTraces/profiling_{LAYER}.png')   

#hls_model.build(csim=False)