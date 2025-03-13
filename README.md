# SZFreqExtract

## Introduction

Seizure frequency is a critical factor in evaluating epilepsy treatment, ensuring patient safety, and mitigating the risk of Sudden Unexpected Death in Epilepsy (SUDEP). However, this information is often embedded in unstructured clinical narratives, making it challenging to extract in a structured format. SZFreqExtract addresses this challenge by providing an automated approach for extracting structured seizure frequency details from clinical text.

This tool is designed to perform two key tasks:
1. **Extracting phrases that describe seizure frequency**
2. **Extracting attributes related to seizure frequency**

To achieve this, based on our investigation of fine-tuning BERT-based models `bert-large-cased`, `biobert-large-cased`, and `Bio_ClinicalBERT` as well as generative large language models `GPT-4`, `GPT-3.5 Turbo`, and `Llama-2-70b-hf`, `GPT-4` demonstrated the best performance across both tasks. The final structured output integrates results from both tasks, showcasing the potential of fine-tuned generative models in extracting structured data from limited text strings.


## Requirements

Note that you need a Microsoft Azure OpenAI subscription to run this code.

Ensure that all required Python packages are installed by using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```


## How to Finetune

To finetune a model (either seizure frequency phrase extraction or attribute extraction), run `Extraction_with_GPT.py` with the following arguments:

```bash
python Extraction_with_GPT.py \
  --job_type finetune \
  --suffix_start <model_suffix> \
  --training_file_name <training_file.jsonl> \
  --validation_file_name <validation_file.jsonl> \
  --base_model <azure_openai_base_model> \
  --n_epochs <number_of_epochs> \
  --batch_size <batch_size> \
  --lr_multiplier <learning_rate_multiplier>
```

### Arguments:
- `--job_type` : Set to `finetune` for model fine-tuning.
- `--suffix_start` : Suffix to identify the model.
- `--training_file_name` : Training file in JSONL format.
- `--validation_file_name` : Validation file in JSONL format.
- `--base_model` : Base model name in Microsoft Azure OpenAI to be fine-tuned.
- `--n_epochs` : Number of epochs for training.
- `--batch_size` : Batch size for training.
- `--lr_multiplier` : Learning rate multiplier.

## How to Run a Finetuned Model on Test Data

To run a fine-tuned model on test data, use the following command:

```bash
python Extraction_with_GPT.py \
  --job_type responses \
  --freq_phrase <yes_or_no> \
  --dataset_json <test_data.jsonl> \
  --deployment_name <azure_openai_deployment_name> \
  --test_dataset_all_data_path <labeled_test_data.jsonl> \
  --output_excel_file <output_results.xlsx>
```

### Arguments:
- `--job_type` : Set to `responses` to generate model responses.
- `--freq_phrase` : Use `"yes"` for frequency phrase extraction models, `"no"` for attribute extraction models.
- `--dataset_json` : Test data in JSONL format where the fine-tuned model will be applied.
- `--deployment_name` : Microsoft Azure OpenAI deployment name of the fine-tuned model.
- `--test_dataset_all_data_path` : JSONL file containing the annotated text segments for the test set, under the column `"freq_str_labeled"`.
- `--output_excel_file` : Output Excel file where the results will be written.