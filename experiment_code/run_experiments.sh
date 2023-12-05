#!/usr/bin/env bash

declare -a settings=("zeroshot" "fewshot" "maxshot" "maxshotinstructions")

for SETTING in ${settings}; do
  python run_few_shot.py --model gpt-3.5-turbo-16k --tokenizer_name gpt-3.5-turbo-16k --setting $SETTING --openai_key $OPENAI_KEY
  python run_few_shot.py --model gpt-4-32k --tokenizer_name gpt-3.5-turbo-16k --setting $SETTING --openai_key $OPENAI_KEY
  python run_few_shot.py --model gpt-4-32k --tokenizer_name gpt-4-32k --setting $SETTING --openai_key $OPENAI_KEY
done

python construct_traindevtest_files.py --model $FT_MODEL_NAME

