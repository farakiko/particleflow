#!/bin/bash

export DATA_DIR=/pfvol/tensorflow_datasets2
export MANUAL_DIR=/pfvol/cld_edm4hep/2024_05/

tfds build mlpf/heptfds/cld_edm4hep/ttbar --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite &> tfds_ttbar.log
