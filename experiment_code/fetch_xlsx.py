import openai
import math
import json
import pprint
import os
import pandas as pd
import time
import tqdm
import random
import csv
from scipy.stats import ttest_ind

with open("../instruct_excel_benchmark.json") as f:
    full_bench_no_data = json.load(f)

#  Fill the 'data_string' field in the benchmark data
for idx, ex in enumerate(tqdm.tqdm(full_bench_no_data)):
    if idx % 100 == 0:
        with open("full_bench_with_data.json", 'w') as f:
            json.dump(full_bench_no_data, f)
    if 'data_string' not in ex:
        file_url = ex['metadata']['filename']
        fname = file_url.split("/")[-1]
        print(fname, file_url)

        # Get the data from the file_url
        try:
            if file_url != "missing":
                os.system("curl %s --output temp.xlsx" % file_url)

                xl = pd.ExcelFile('temp.xlsx')

                datastr = ""
                for sheetname in xl.sheet_names:
                    df = xl.parse(sheetname)
                    df_str = df.to_string()
                    datastr += "SHEETNAME: " + sheetname + "\n\nSHEET:\n" + df_str + "\n\n"

                ex['data_string'] = datastr
            else:
                ex['data_string'] = "<DATA NOT AVAILABLE>"
        except:
            ex['data_string'] = "<DATA NOT AVAILABLE>"

with open("full_bench_with_data.json", 'rb') as f:
    full_bench = json.load(f)
    
print(len(full_bench))

full_bench_train = []
full_bench_dev = []
full_bench_test = []
    
for ex in full_bench:
    if len(full_bench_test) < 1000 and ex['data_string'] not in ["<DATA NOT AVAILABLE>", "<DATA NOT AVAILABLE"]:
        full_bench_test.append(ex)
    elif len(full_bench_dev) < 1000 and ex['data_string'] not in ["<DATA NOT AVAILABLE>", "<DATA NOT AVAILABLE"]:
        full_bench_dev.append(ex)
    elif ex['data_string'] not in ["<DATA NOT AVAILABLE>", "<DATA NOT AVAILABLE"]:
        full_bench_train.append(ex)
        
print(len(full_bench_train))
print(len(full_bench_dev))
print(len(full_bench_test))

with open("full_bench_train.json", 'w') as f:
    json.dump(full_bench_train, f)
    
with open("full_bench_dev.json", 'w') as f:
    json.dump(full_bench_dev, f)
    
with open("full_bench_test.json", 'w') as f:
    json.dump(full_bench_test, f)