#!/bin/sh
# crf_learn --maxiter=10 -c 4.0 template data/train.data model > log_process/train_log.txt
crf_learn -c 4.0 template data/train.data tempmodel > log_process/train_log.txt
rm tempmodel
# crf_test -m model data/test.data > log_process/test_log.txt

#../../crf_learn -a MIRA template train.data model
#../../crf_test -m model test.data

#../../crf_learn -a CRF-L1 template train.data model
#../../crf_test -m model test.data

# rm -f model
