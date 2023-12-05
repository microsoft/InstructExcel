# InstructExcel: A Benchmark for Natural Language Instruction in Excel

This repository contains the InstructExcel benchmark, along with code for running the experiments in the [InstructExcel paper](https://arxiv.org/abs/2310.14495).

InstructExcel pairs natural language queries with [OfficeScripts code](https://learn.microsoft.com/en-us/office/dev/scripts/) achieving the requested task in Excel. These pairs also come with associated URL's for downloading the Excel file on which the query/code can be executed.

## Benchmark

The benchmark is included in `instruct_excel_benchmark.json`. Please see the paper for more information on the schema. Some Excel file URL's are currently missing, and are listed as "missing", but most of the items in the benchmark include publicly available links to download the Excel file associated with the NL-code pair.

## Instructions for reproducing experiments

Code for reproducing experiments is listed in the `experiment_code` folder. This code is set up to run from inside the `experiments_code` directory.

First install libraries in the `requirements.txt` file.

Before doing any experiments, you should also open the InjectDataToBenchmark python notebook and run it to save out train, dev, and test splits of the benchmark with data included from the Excel files downloaded from the web. 

### Zero-shot and Few-shot Experiments

The code for zero-shot and few-shot experiments with GPT is in `run_few_shot.py`.
There is a bash script `run_experiments.sh` that shows example calls of the `run_few_shot.py` script.  If you use the bash script, you should set the OPENAI_KEY environment variable before running it, and set FT_MODEL_NAME to the name of the model you're using to run the finetune experiments (using the name as listed in [OpenAI's documentation](https://platform.openai.com/docs/models)). 

You can run the 0-shot/few-shot experiments with the following syntax:

`python run_few_shot.py --model modelname --tokenizer_name tokenizername --setting zeroshot/fewshot/maxshot/maxshotinstructions --openai_key yourapikey`

You can turn on the behavior of including the API description in the prompt (for any setting) by adding `--api_in_prompt` to the arguments list. You can turn on dynamic prompting using F1 score by adding `--dynamic_prompt` to the arguments list.

### Finetuning

The file `construct_traindevtest_files.py` was used to collect the train, dev, and test sets into a format usable by our finetuning API, and may be useful for others as well. It generates jsonl files for each setting.

`run_experiments.sh` also shows an example call to `construct_traindevtest_files.py`.






## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.


## Bib
```
@inproceedings{payan2023instructexcel,
    title={{I}nstruct{E}xcel: {A} Benchmark for Natural Language Instruction in {E}xcel},
    author={Payan, Justin and Mishra, Swaroop and Singh, Mukul and Negreanu, Carina and Poelitz, Christian and Baral, Chitta and Roy, Subhro and Chakravarthy, Rasika and Van Durme, Benjamin and Nouri, Elnaz},
    booktitle={Findings of the Association for Computational Linguistics: EMNLP 2023},  
    year={2023}
}
```
