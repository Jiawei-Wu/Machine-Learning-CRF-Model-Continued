crf_learn --maxiter=5 -c 4.0 template data/train.data model
crf_test -m model data/test.data > log_process/test_log.txt
