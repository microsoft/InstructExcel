# result on 60 samples of GPT-3.5-Turbo
## few-shot
```
{'sacrebleu': 26.01778675352435, 'rouge': 33.68887687431666, 'exact_match': 1.6666666666666667, 'f1': 15.854225181173923, 'sacrebleu_minus_params': 31.32635476273563, 'rouge_minus_params': 41.14314218127439, 'exact_match_minus_params': 1.6666666666666667, 'f1_minus_params': 17.754316691852758}
```

|sacrebleu|rouge|exact_match|f1|
|---------|-----|-----------|--|
|26.02|33.69|1.67|15.85|

## max-shot
```
{'sacrebleu': 34.217978107289795, 'rouge': 41.541124800869014, 'exact_match': 1.6666666666666667, 'f1': 23.14737745527758, 'sacrebleu_minus_params': 42.34938792835638, 'rouge_minus_params': 50.942274907561924, 'exact_match_minus_params': 6.666666666666667, 'f1_minus_params': 28.58744222874659}
```

|sacrebleu|rouge|exact_match|f1|
|---------|-----|-----------|--|
|34.22|41.54|1.67|23.15|

Reproduction is stopped due to shortage of API usage
