#!/bin/bash

echo "Starting reward model server on IP:"
echo $1
echo "Role: $2"

source $HOME/.bashrc
source $HOME/OpenRLHF/bin/activate
port=6379

if [ "$2" == "head" ]; then
    # Start Ray head node
    ray start --head --node-ip-address=$1
elif [ "$2" == "worker" ]; then
    # Start Ray worker node and connect to the head node
    ray start --address=$1:$port --block
fi

# Run the reward server only on the head node
if [ "$2" == "head" ]; then
    export WORKING_DIR=$HOME/MolGenDocking
    export DATASET=molgendata

    cp $SCRATCH/MolGenData/$DATASET.tar.gz $SLURM_TMPDIR
    cd $SLURM_TMPDIR
    tar -xzf $DATASET.tar.gz
    cd $WORKING_DIR
    cp data/properties.csv $SLURM_TMPDIR
    export DATA_PATH=$SLURM_TMPDIR/$DATASET

    export docking_oracle=autodock_gpu
    export scorer_exhaustiveness=4
    uvicorn --host $1 --port 5001 molrgen.server:app
fi
