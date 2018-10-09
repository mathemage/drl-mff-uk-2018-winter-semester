#!/usr/bin/env bash

PYTHON="/home/mathemage/anaconda3/envs/drl-mff-uk-2018-winter-semester/bin/python"
SCRIPT=./multiarmed_bandits.py

# 1) greedy
${PYTHON} ${SCRIPT} --epsilon 0.015625  # 1/64
${PYTHON} ${SCRIPT} --epsilon 0.03125   # 1/32
${PYTHON} ${SCRIPT} --epsilon 0.0625    # 1/16
${PYTHON} ${SCRIPT} --epsilon 0.125     # 1/8
${PYTHON} ${SCRIPT} --epsilon 0.25      # 1/4

# 2) greedy and alpha
${PYTHON} ${SCRIPT} --alpha 0.15 --epsilon 0.015625  # 1/64
${PYTHON} ${SCRIPT} --alpha 0.15 --epsilon 0.03125   # 1/32
${PYTHON} ${SCRIPT} --alpha 0.15 --epsilon 0.0625    # 1/16
${PYTHON} ${SCRIPT} --alpha 0.15 --epsilon 0.125     # 1/8
${PYTHON} ${SCRIPT} --alpha 0.15 --epsilon 0.25      # 1/4

# 3) greedy and alpha
${PYTHON} ${SCRIPT} --initial 1 --alpha 0.15 --epsilon 0.0078125 # 1/128
${PYTHON} ${SCRIPT} --initial 1 --alpha 0.15 --epsilon 0.015625  # 1/64
${PYTHON} ${SCRIPT} --initial 1 --alpha 0.15 --epsilon 0.03125   # 1/32
${PYTHON} ${SCRIPT} --initial 1 --alpha 0.15 --epsilon 0.0625    # 1/16

# 4) UCB
${PYTHON} ${SCRIPT} --mode ucb --c 0.0078125 # 1/128
${PYTHON} ${SCRIPT} --mode ucb --c 0.015625  # 1/64
${PYTHON} ${SCRIPT} --mode ucb --c 0.03125   # 1/32
${PYTHON} ${SCRIPT} --mode ucb --c 0.0625    # 1/16
${PYTHON} ${SCRIPT} --mode ucb --c 0.125     # 1/8

# 5) gradient
${PYTHON} ${SCRIPT} --mode gradient --alpha 0.0625    # 1/16
${PYTHON} ${SCRIPT} --mode gradient --alpha 0.125     # 1/8
${PYTHON} ${SCRIPT} --mode gradient --alpha 0.25      # 1/4
${PYTHON} ${SCRIPT} --mode gradient --alpha 0.5       # 1/2
