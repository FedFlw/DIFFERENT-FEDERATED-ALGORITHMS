#!/bin/bash

# Run FedAvg strategy
echo "Running FedAvg strategy"
python -m power_of_choice.main variant="rand"

# Run Power-of-choice selectors
echo "Running pow-d strategy with d=20 and CK=9"
python -m power_of_choice.main variant="base" strategy.d=20 strategy.ck=9

echo "Running cpow-d strategy with d=20 and CK=9"
python -m power_of_choice.main variant="cpow" strategy.d=20 strategy.ck=9

echo "Running rpow-d strategy with d=60 and CK=9"
python -m power_of_choice.main variant="rpow" strategy.d=60 strategy.ck=9

# Run Resource-aware workload allocators
echo "Running static optimizer"
python -m power_of_choice.pow_so

echo "Running uniform optimizer"
python -m power_of_choice.pow_uo optimizer=uo

echo "Running rt optimizer"
python -m power_of_choice.pow_rt optimizer=rt

echo "Running ecto optimizer"
python -m power_of_choice.pow_ecto optimizer=ecto

