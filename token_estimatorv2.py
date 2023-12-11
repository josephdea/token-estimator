import numpy as np
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import os
from evals.utils import log_utils
from evals.registry import registry, is_chat_model
from pathlib import Path
import json
import seaborn as sns
import random
import tiktoken
import sys

def parse_spec(logpath):
    record = {}
    with logpath.open() as f:
        for line in f.readlines():
            line = line.strip()
            loaded_line = json.loads(line)
            if 'type' in loaded_line and loaded_line['type'] == 'token count':
                sid = loaded_line['sample_id']
                itc = loaded_line['data']['input']
                otc = loaded_line['data']['output']
                if sid not in record:
                    record[sid] = [0,0]
                record[sid][0] += itc
                record[sid][1] += otc
    input_count = 0
    output_count = 0
    for key in record:
        input_count += record[key][0]
        output_count += record[key][1]
    return input_count,output_count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir","-d",type=str,default="./")
    parser.add_argument("--file_path","-f",type=str,default=None)
    parser.add_argument("--model","-m",type=str,default=None)
    
    args = parser.parse_args()
    model = args.model
    input_arr = []
    output_arr = []
    if args.file_path == None:
        for path,_ in sorted(list(log_utils.get_final_results_from_dir(args.log_dir).items())):
            itc,otc = parse_spec(path)
            print(path)
            print("input token count:",itc)
            print("output token count:",otc,"\n")
            input_arr.append(itc)
            output_arr.append(otc)
    else:
        path = Path(args.file_path)
        itc,otc = parse_spec(path)
        print(path)
        print("input token count:",itc)
        print("output token count:",otc,"\n")
        input_arr.append(itc)
        output_arr.append(otc)
    print("Average Input Token Length:",np.mean(input_arr))
    print("Average Output Token Length:",np.mean(output_arr))
    print("Average Token Length:",np.mean(input_arr)+np.mean(output_arr))


if __name__ == "__main__":
    main()
