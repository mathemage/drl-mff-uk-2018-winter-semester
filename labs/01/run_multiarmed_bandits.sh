#!/usr/bin/env bash

PYTHON=python
SCRIPT=./multiarmed_bandits.py
OUTPUT_FILE="results.txt"

# reset output file
>${OUTPUT_FILE}

echo "1) greedy:" >>${OUTPUT_FILE}
${PYTHON} ${SCRIPT} --epsilon 0.015625  # 1/64
${PYTHON} ${SCRIPT} --epsilon 0.03125   # 1/32
${PYTHON} ${SCRIPT} --epsilon 0.0625    # 1/16
${PYTHON} ${SCRIPT} --epsilon 0.125     # 1/8
${PYTHON} ${SCRIPT} --epsilon 0.25      # 1/4

echo "2) greedy and alpha:" >>${OUTPUT_FILE}
${PYTHON} ${SCRIPT} --alpha 0.15 --epsilon 0.015625  # 1/64
${PYTHON} ${SCRIPT} --alpha 0.15 --epsilon 0.03125   # 1/32
${PYTHON} ${SCRIPT} --alpha 0.15 --epsilon 0.0625    # 1/16
${PYTHON} ${SCRIPT} --alpha 0.15 --epsilon 0.125     # 1/8
${PYTHON} ${SCRIPT} --alpha 0.15 --epsilon 0.25      # 1/4

echo "3) greedy and initial:" >>${OUTPUT_FILE}
${PYTHON} ${SCRIPT} --initial 1 --alpha 0.15 --epsilon 0.0078125 # 1/128
${PYTHON} ${SCRIPT} --initial 1 --alpha 0.15 --epsilon 0.015625  # 1/64
${PYTHON} ${SCRIPT} --initial 1 --alpha 0.15 --epsilon 0.03125   # 1/32
${PYTHON} ${SCRIPT} --initial 1 --alpha 0.15 --epsilon 0.0625    # 1/16

echo "4) UCB:" >>${OUTPUT_FILE}
${PYTHON} ${SCRIPT} --mode ucb --c 0.25 # 1/4
${PYTHON} ${SCRIPT} --mode ucb --c 0.5  # 1/2
${PYTHON} ${SCRIPT} --mode ucb --c 1    # 1
${PYTHON} ${SCRIPT} --mode ucb --c 2    # 2
${PYTHON} ${SCRIPT} --mode ucb --c 4    # 4

echo "5) gradient:" >>${OUTPUT_FILE}
${PYTHON} ${SCRIPT} --mode gradient --alpha 0.0625    # 1/16
${PYTHON} ${SCRIPT} --mode gradient --alpha 0.125     # 1/8
${PYTHON} ${SCRIPT} --mode gradient --alpha 0.25      # 1/4
${PYTHON} ${SCRIPT} --mode gradient --alpha 0.5       # 1/2
