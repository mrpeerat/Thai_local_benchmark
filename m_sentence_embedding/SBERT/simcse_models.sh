#!/bin/bash 

models=(128 256 512)

for margin in "${margin_all[@]}"
do
    for dropout_rate in "${dropout_rates[@]}"
    do
        for fc_dimension in "${fc_dimensions[@]}"
        do
            python use_finetune_cldr_no_regulator.py -replace $replace
        done
    done
done



