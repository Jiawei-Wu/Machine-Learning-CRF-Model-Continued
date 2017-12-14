#!/bin/sh
# crf_learn --maxiter=10 -c 4.0 template data/train.data model > log_process/train_log.txt
crf_learn -c 4.0 template data/train.data model > log_process/train_log.txt
# crf_learn -c 4.0 template data/train.data model
rm model

