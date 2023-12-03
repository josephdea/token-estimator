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
    prompt_data = []
    sampled_data = []
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
                if 'prompt' in loaded_line['data']: #this is input
                    dfs_add(loaded_line['data']['prompt'],prompt_data)
                if 'sampled' in loaded_line['data']: #this is output
                    dfs_add(loaded_line['data']['sampled'],sampled_data)

    main_model_token_count = 0
    if main_model != None:
        encoder = tiktoken.encoding_for_model(main_model)
        for msg in prompt_data:
            main_model_token_count += len(encoder.encode(msg))
        for msg in sampled_data:
            main_model_token_count += len(encoder.encode(msg))

    extraction_model_token_count = 0
    if extraction_model != None:
        encoder = tiktoken.encoding_for_model(extraction_model)
        for msg in prompt_data:
            extraction_model_token_count += len(encoder.encode(msg))
        for msg in sampled_data:
            extraction_model_token_count += len(encoder.encode(msg))

    input_token_count = 0
    output_token_count = 0
    if main_model != None: # main model looks at prompt as input tokens, and produced sampled as output tokens
        encoder = tiktoken.encoding_for_model(main_model)
        for msg in prompt_data:
            input_token_count += len(encoder.encode(msg))
        for msg in sampled_data:
            output_token_count += len(encoder.encode(msg))

    if extraction_model != None: #extraction model looks at sampled as input tokens, and produces the answer as output tokens(TODO)
        encoder = tiktoken.encoding_for_model(extraction_model)
        for msg in sampled_data:
            input_token_count += len(encoder.encode(msg))



    return main_model_token_count,extraction_model_token_count,input_token_count,output_token_count

def output_info(path,mm_count,em_count,arr,itc,otc):
    if(em_count != 0):
        arr.append((mm_count+em_count)/2)
        print(str(path).split("/")[-1])
        print("estimated token length range:","["+str(min(mm_count,em_count))+","+str(max(mm_count,em_count))+"]")
        print("input token length:",itc)
        print("output token length:",otc)
    else:
        arr.append(mm_count)
        print(str(path).split("/")[-1])
        print("estimated token length:",mm_count)
        print("input token length:",itc)
        print("output token length:",otc)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir","-d",type=str,default="./")
    parser.add_argument("--file_path","-f",type=str,default=None)
    parser.add_argument("--model","-m",type=str,default=None)
    
    args = parser.parse_args()
    model = args.model
    token_count_arr = []
    if args.file_path == None:
        for path,_ in sorted(list(log_utils.get_final_results_from_dir(args.log_dir).items())):
            mm_count,em_count,itc,otc = parse_spec(path,main_model=model)
            output_info(path,mm_count,em_count,token_count_arr,itc,otc)
    else:
        path = Path(args.file_path)
        mm_count,em_count,itc,otc = parse_spec(path,main_model=model)
        output_info(path,mm_count,em_count,token_count_arr,itc,otc)
    print("Average Token Length:",np.mean(token_count_arr))


if __name__ == "__main__":
    main()
