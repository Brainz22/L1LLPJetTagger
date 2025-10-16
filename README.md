# L1JetTagger
Compilation of files relating to the L1 LLP Jet Tagger.

### Main files:

* `hls4ml_proj/firmware` contains my `myproject.cpp`. This file is the output design in `qkerasModel.py` produced by the hls4ml tool.

* `qkerasModel.py` has the qkeras model.

* `HLS_qk_Roc_Tracing.py` is the file used to produce `hls_Qk_ROCCurve.png` and the `LayerTraces` folder, where tracing results and a model graph is saved. 

* `qkL1JetTagModel_hls_config.ipynb` is the file used to convert the model saved in `<text>_qkL1JetTagModel.h5` to HLS. This jupyter notebooks shows all of the configurations used, i.e reuse factor, precision, etc...
