#!/bin/bash

export APP_ROOT=$HOME/workspace-cpp/BinVocabTree

export PATH=$APP_ROOT/VocabLearn/:$APP_ROOT/VocabBuildDB/:$APP_ROOT/VocabMatch/:$APP_ROOT/FeatureExtract/:$PATH:$APP_ROOT/ComputeMAP:$APP_ROOT/SelectDescriptors:$APP_ROOT/FeatureSelect:$APP_ROOT/GeomVerify:$APP_ROOT/OxDataReader

export LD_LIBRARY_PATH=$APP_ROOT/lib:$LD_LIBRARY_PATH

