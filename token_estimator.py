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

pricing = {
    'input': {},
    'output': {},
}
pricing['input']['gpt-4'] = 0.03 / 1000
pricing['output']['gpt-4'] = 0.06 / 1000

pricing['input']['gpt-4-32k'] = 0.06 / 1000
pricing['output']['gpt-4-32k'] = 0.12 / 1000

pricing['input']['gpt-3.5-turbo'] = 0.0015 / 1000
pricing['output']['gpt-3.5-turbo'] = 0.002 / 1000

pricing['input']['gpt-3.5-turbo-16k'] = 0.003 / 1000
pricing['output']['gpt-3.5-turbo-16k'] = 0.004 / 1000

pricing['input']['davinci-002'] = 0.002 / 1000
pricing['output']['babbage-002'] = 0.0004 / 1000


def dfs_add(loaded_line,arr):
    if(type(loaded_line) == dict):
        arr.append(loaded_line['content'])
    elif(type(loaded_line) == str):
        arr.append(loaded_line)
    elif(type(loaded_line) == list):
        for i in range(len(loaded_line)):
            dfs_add(loaded_line[i],arr)

def parse_spec(logpath,main_model=None,extraction_model=None):
    input_main_model = []
    output_main_model = []
    with logpath.open() as f:
        for line in f.readlines():
            line = line.strip()
            loaded_line = json.loads(line)
            if 'spec' in loaded_line:
                comp_fn_name = loaded_line['spec']['completion_fns'][0]
                completion_fn = registry.make_completion_fn(comp_fn_name)
                if hasattr(completion_fn,'cot_completion_fn') and hasattr(completion_fn,'extract_completion_fn'): ### cot completion fn
                    if(main_model == None):
                        main_model = completion_fn.cot_completion_fn.model
                    extraction_model = completion_fn.extract_completion_fn.model
                elif hasattr(completion_fn,'completion_fn'):   ### regular completion fn
                    if(main_model == None):
                        main_model = completion_fn.completion_fn.model
                else:
                    print("Setting Not Recognized")
                    return 0
            elif 'run_id' in loaded_line and 'data' in loaded_line:
                if 'prompt' in loaded_line['data']:
                    dfs_add(loaded_line['data']['prompt'],input_main_model)
                if 'sampled' in loaded_line['data']:
                    dfs_add(loaded_line['data']['sampled'],output_main_model)

    mm_input_token_count = 0
    mm_output_token_count = 0
    em_input_token_count = 0

    if main_model != None:
        encoder = tiktoken.encoding_for_model(main_model)
        for msg in input_main_model:
            mm_input_token_count += len(encoder.encode(msg))
        for msg in output_main_model:
            mm_output_token_count += len(encoder.encode(msg))
    else:
        return 0

    if extraction_model != None:
        encoder = tiktoken.encoding_for_model(extraction_model)
        for msg in output_main_model:
            em_input_token_count += len(encoder.encode(msg))

    token_count = mm_input_token_count + mm_output_token_count + em_input_token_count
    estimated_cost = pricing['input'].get(main_model,0) * mm_input_token_count + pricing['output'].get(main_model,0) * mm_output_token_count + pricing['input'].get(extraction_model,0) * em_input_token_count
    return token_count,estimated_cost

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir","-d",type=str,default="./")
    parser.add_argument("--file_path","-f",type=str,default=None)
    parser.add_argument("--model","-m",type=str,default=None)
    
    args = parser.parse_args()
    model = args.model
    token_count_arr = []
    cost_arr = []
    if args.file_path == None:
        for path,_ in sorted(list(log_utils.get_final_results_from_dir(args.log_dir).items())):
            token_count,cost = parse_spec(path,main_model=model)
            token_count_arr.append(token_count)
            cost_arr.append(cost)
            print(str(path).split("/")[-1],"token length:",token_count,"estimated cost:",cost)
    else:
        path = Path(args.file_path)
        token_count,cost = parse_spec(path,main_model=model)
        token_count_arr.append(token_count)
        cost_arr.append(cost)
        print(str(path).split("/")[-1],"token length:",token_count,"estimated cost:",cost)
    print("Average Token Length:",np.mean(token_count_arr))
    print("Average Cost:",np.mean(cost_arr))


if __name__ == "__main__":
    main()


