#!/bin/bash

traj_info="obj20_sg10_persg5"
model="maskedrl"

REWARD_LEARNING_CONFIG="../config/reward_learning/${traj_info}/${model}.yaml"
HUMAN_CONFIG="../config/humans/frankarobot_multiple_humans.yaml"

# for seed in 12345 23451 34512 45123 51234;
# for seed in 45123 51234;
for seed in 12345;
# for seed in 23451 34512 45123 51234;
do
    echo "Running evaluation with seed: $seed"
    # for demos in 5 10 20 50 100;
    # for demos in 5 10 20 50;
    # for demos in 10 1 2 3 4 5 6 7 8 9;
    # for demos in 1 2 3 4 5;
    # for demos in 4 5;
    # for demos in 1;
    # for demos in 10;
    for demos in 1 2 3 4 5 6 7 8 9;
    # for demos in 40 60 80;
    # for demos in 20 50 100 40 60 80;
    # for demos in 20 50 100;
    # for demos in 1000;
    do
    echo "IRL with $demos demos"

        # for num_thetas in 10 20 30 40 50 60 70 80 90 100;
        # for num_thetas in 2 3 4 5 6 7 8 9 10 20;
        for num_thetas in 50;
        # for num_thetas in 10 50 100;
        do
        echo "Number of thetas: $num_thetas"
    
        # Run the Python script and capture its output
        output=$(python3 scripts/train.py \
            --seed "$seed" \
            --config $REWARD_LEARNING_CONFIG \
            -hc $HUMAN_CONFIG \
            -dq $demos \
            --num_train_thetas $num_thetas \
            --wandb)

        
        # Extract the line containing "Average win rate:"
        # win_rate_line=$(echo "$output" | grep "Average win rate:")

        # # If the line exists, append it to the output file with the seed number
        # if [ -n "$win_rate_line" ]; then
        #     echo "Seed $seed, demos $demos: $win_rate_line" >> "$OUTPUT_FILE"
        # else
        #     echo "Seed $seed, demos $demos: No win rate found" >> "$OUTPUT_FILE"
        # fi
        done
    done
done
