import argparse
import json

from run_few_shot import cut_down_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_name = args.model
    tokenizer_name = model_name

    # Just load in the training set, dev set, and test set along with the data, generate the prompts,
    # and then also load in the augmented NL instructions and add those. Then write out the file in babel style.
    with open("full_bench_train.json", 'rb') as f:
        full_bench_train = json.load(f)
    with open("full_bench_dev.json", 'rb') as f:
        full_bench_dev = json.load(f)
    with open("full_bench_test.json", 'rb') as f:
        full_bench_test = json.load(f)

    # Create the prompts.
    train_prompts = []
    train_completions = []
    dev_prompts = []
    dev_completions = []
    test_prompts = []

    code_header = "function main(workbook: ExcelScript.Workbook) {  let selectedSheet = workbook.getActiveWorksheet(); // "

    def create_prompt(ex, tokenizer_name, code_header):
        prompt = 'Generate a function with excel script to execute the action given below in natural language'
        prompt += '\n' + 'Action:' + '\n' + str(ex['input'])
        stripped_data = cut_down_data([ex['data_string']],
                                      prompt + '\nData:\n' + '\nExcel Script Function:\n' + code_header, tokenizer_name,
                                      1)
        prompt += '\nData:\n' + str(stripped_data[0])
        prompt += '\nExcel Script Function:\n' + code_header
        return prompt

    print("Creating train prompts and completions")
    for ex in full_bench_train:
        prompt = create_prompt(ex, tokenizer_name, code_header)
        completion = str(ex['output'][103:])
        train_prompts.append(prompt)
        train_completions.append(completion)

    print("Creating dev prompts and completions")
    for ex in full_bench_dev:
        prompt = create_prompt(ex, tokenizer_name, code_header)

        completion = str(ex['output'][103:])
        dev_prompts.append(prompt)
        dev_completions.append(completion)

    print("Creating test prompts")
    for ex in full_bench_test:
        prompt = create_prompt(ex, tokenizer_name, code_header)

        test_prompts.append(prompt)

    # Ok, now write out the examples into a file named for just the base finetuning exp
    with open("instruct_excel_with_data_train.jsonl", 'w') as f:
        for idx in range(len(train_prompts)):
            to_write = {"prompt": train_prompts[idx], "completion": train_completions[idx]}
            f.write(json.dumps(to_write) + "\n")

    with open("instruct_excel_with_data_dev.jsonl", 'w') as f:
        for idx in range(len(dev_prompts)):
            to_write = {"prompt": dev_prompts[idx], "completion": dev_completions[idx]}
            f.write(json.dumps(to_write) + "\n")

    with open("instruct_excel_with_data_test.jsonl", 'w') as f:
        for idx in range(len(test_prompts)):
            to_write = {"prompt": test_prompts[idx]}
            f.write(json.dumps(to_write) + "\n")




