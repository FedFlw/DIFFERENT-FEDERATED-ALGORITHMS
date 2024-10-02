#!/bin/bash

# Array of alpha values
alpha_values=(0.1 0.5 1 1.5 2 10 100)

# Loop over alpha values
for alpha in "${alpha_values[@]}"
do
  echo "Running dataset preparation with alpha=${alpha}"
  python -m power_of_choice.dataset_preparation alpha=${alpha}

  echo "Running experiments for FedAvg"
  python -m power_of_choice.main variant="rand"

  echo "Running experiments for pow-d with d=20 and CK=9"
  python -m power_of_choice.main variant="base" strategy.d=20 strategy.ck=9

  echo "Running experiments for cpow-d with d=20 and CK=9"
  python -m power_of_choice.main variant="cpow" strategy.d=20 strategy.ck=9

  echo "Running experiments for rpow-d with d=60 and CK=9"
  python -m power_of_choice.main variant="rpow" strategy.d=60 strategy.ck=9

  echo "Running static optimizer"
  python -m power_of_choice.pow_so

  echo "Running uniform optimizer"
  python -m power_of_choice.pow_uo optimizer=uo

  echo "Running rt optimizer"
  python -m power_of_choice.pow_rt optimizer=rt

  echo "Running ecto optimizer"
  python -m power_of_choice.pow_ecto optimizer=ecto

done

