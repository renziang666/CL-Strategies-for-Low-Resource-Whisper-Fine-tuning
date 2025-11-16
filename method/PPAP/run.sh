#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
echo "stage 1 start"
python /home/renziang/summer_holiday/task2_whisper/ckz/code/history/PPAP/finetune_PPAP_stepone.py

echo "stage 2 start"
python /home/renziang/summer_holiday/task2_whisper/ckz/code/history/PPAP/PAPP_steptwo.py
