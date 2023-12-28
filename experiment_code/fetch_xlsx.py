import json
import os
import pandas as pd
import tqdm
import hashlib

with open("../instruct_excel_benchmark.json") as f:
    full_bench_no_data = json.load(f)

if not os.path.exists("xlsx"):
    os.mkdir("xlsx")

#  Fill the 'data_string' field in the benchmark data
for idx, ex in enumerate(tqdm.tqdm(full_bench_no_data)):
    if 'data_string' not in ex:
        file_url = ex['metadata']['filename']
        fname = file_url.split("/")[-1]
        print(fname, file_url)
        file_hash = hashlib.md5(file_url.encode("utf-8")).hexdigest()
        # Get the data from the file_url
        try:
            if file_url != "missing":
                if not os.path.exists(f"xlsx/{file_hash}.xlsx"):
                    print(f"downloading {file_hash}")
                    os.system("curl --connect-timeout 10 -m 20 %s --output ./xlsx/%s.xlsx" % (file_url, file_hash))
            else:
                ex['data_string'] = "<DATA NOT AVAILABLE>"
        except:
            ex['data_string'] = "<DATA NOT AVAILABLE>"

for idx, ex in enumerate(tqdm.tqdm(full_bench_no_data)):
    if idx % 100 == 0:
        with open("full_bench_with_data.json", 'w') as f:
            json.dump(full_bench_no_data, f)
    if 'data_string' not in ex:
        file_url = ex['metadata']['filename']
        fname = file_url.split("/")[-1]
        file_hash = hashlib.md5(file_url.encode("utf-8")).hexdigest()
        # Get the data from the file_url
        try:
            if file_url != "missing":

                xl = pd.ExcelFile(f"./xlsx/{file_hash}.xlsx", "openpyxl")

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

with open("full_bench_with_data.json", 'w') as f:
    json.dump(full_bench_no_data, f)

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