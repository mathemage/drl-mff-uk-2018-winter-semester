#!/usr/bin/env bash

PYTHON="/home/mathemage/anaconda3/envs/drl-mff-uk-2018-winter-semester/bin/python"
SCRIPT=./policy_iteration.py
OUTPUT_FILE="results_policy_iteration.txt"

# reset output file
>${OUTPUT_FILE}

echo "iterations:" >>${OUTPUT_FILE}
echo "##############################" >>${OUTPUT_FILE}
${PYTHON} ${SCRIPT} --iterations  1 >>${OUTPUT_FILE}
echo "##############################" >>${OUTPUT_FILE}
${PYTHON} ${SCRIPT} --iterations  2 >>${OUTPUT_FILE}
echo "##############################" >>${OUTPUT_FILE}
${PYTHON} ${SCRIPT} --iterations  4 >>${OUTPUT_FILE}
echo "##############################" >>${OUTPUT_FILE}
${PYTHON} ${SCRIPT} --iterations  8 >>${OUTPUT_FILE}
echo "##############################" >>${OUTPUT_FILE}
${PYTHON} ${SCRIPT} --iterations 16 >>${OUTPUT_FILE}
echo "##############################" >>${OUTPUT_FILE}
