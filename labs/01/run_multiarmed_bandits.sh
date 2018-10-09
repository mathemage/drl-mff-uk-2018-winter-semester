#!/usr/bin/env bash

PYTHON="/home/mathemage/anaconda3/envs/drl-mff-uk-2018-winter-semester/bin/python"
SCRIPT=./multiarmed_bandits.py

# 1) greedy
${PYTHON} ${SCRIPT} --epsilon 0.015625  # 1/64
${PYTHON} ${SCRIPT} --epsilon 0.03125   # 1/32
${PYTHON} ${SCRIPT} --epsilon 0.0625    # 1/16
${PYTHON} ${SCRIPT} --epsilon 0.125     # 1/8
${PYTHON} ${SCRIPT} --epsilon 0.25      # 1/4
