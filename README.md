# token-estimator
Estimating Token Length of Eval Experiments

To get the token length for an experiment from one file.
```sh
python3 token_estimator.py -f example.log
```



To get the token length for an all experiments in a directory of log files.
```sh
python3 token_estimator.py -d log_dir/
```


To specify the model used.
```sh
python3 token_estimator.py -d log_dir/ -m gpt-4
```
