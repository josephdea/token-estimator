import numpy as np
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import os
from evals.utils import log_utils
from evals.registry import registry
from pathlib import Path
import json
import seaborn as sns
import random
import tiktoken

def dfs_add(loaded_line,arr):
    if(type(loaded_line) == dict):
        arr.append(loaded_line['content'])
    elif(type(loaded_line) == str):
        arr.append(loaded_line)
    elif(type(loaded_line) == list):
        for i in range(len(loaded_line)):
            dfs_add(loaded_line[i],arr)


def parse_spec(logpath):
    all_prompt_data = []
    all_sampled_data = []
    main_model = None
    extraction_model = None
    with logpath.open() as f:
        for line in f.readlines():
            line = line.strip()
            loaded_line = json.loads(line)
            if 'spec' in loaded_line:
                comp_fn_name = loaded_line['spec']['completion_fns'][0]
                completion_fn = registry.make_completion_fn(comp_fn_name)
                if hasattr(completion_fn,'cot_completion_fn') and hasattr(completion_fn,'extract_completion_fn'): ### cot completion fn
                    main_model = completion_fn.cot_completion_fn.model
                    extraction_model = completion_fn.extract_completion_fn.model
                elif hasattr(completion_fn,'completion_fn'):   ### regular completion fn
                    main_model = completion_fn.completion_fn.model
                else:
                    print("Setting Not Recognized")
                    return 0
            elif 'run_id' in loaded_line and 'data' in loaded_line:
                if 'prompt' in loaded_line['data']:
                    dfs_add(loaded_line['data']['prompt'],all_prompt_data)
                if 'sampled' in loaded_line['data']:
                    dfs_add(loaded_line['data']['sampled'],all_sampled_data)
    #print(len(all_prompt_data),len(all_sampled_data))
    token_count = 0
    if main_model != None:
        encoder = tiktoken.encoding_for_model(main_model)
        for msg in all_prompt_data:
            token_count += len(encoder.encode(msg))
        for msg in all_sampled_data:
            token_count += len(encoder.encode(msg))
    if extraction_model != None:
        encoder = tiktoken.encoding_for_model(extraction_model)
        for msg in all_sampled_data:
            token_count += len(encoder.encode(msg))
    return token_count
    #TODO: create tiktoken model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir","-d",type=str,default="./")
    parser.add_argument("--file_path","-f",type=str,default=None)
    args = parser.parse_args()
    arr = []
    if(args.file_path == None):
        for path,_ in sorted(list(log_utils.get_final_results_from_dir(args.log_dir).items())):
            val = parse_spec(path)
            arr.append(val)
            print(str(path).split("/")[-1],"token length:",val)
    else:
        path = Path(args.file_path)
        val = parse_spec(path)
        arr.append(val)
        print(str(path).split("/")[-1],"token length:",val)
    print("Average Token Length:",np.mean(arr))

if __name__ == "__main__":
    main()


