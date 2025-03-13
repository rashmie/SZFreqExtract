import pandas as pd
import json
import tiktoken
import numpy as np
from collections import defaultdict
import re
import math
import os
import sys
from openai import AzureOpenAI
import time

import math
import argparse
from tqdm import tqdm

def fine_tune_gpt(client, base_model, training_file_id, validation_file_id, suffix, n_epochs, batch_size, learning_rate_multiplier):
    response = client.fine_tuning.jobs.create(
        suffix = suffix,
        training_file=training_file_id,
        validation_file=validation_file_id,
        model=base_model, # Enter base model name. Note that in Azure OpenAI the model name contains dashes and cannot contain dot/period characters. 
        hyperparameters = {
                            "n_epochs": n_epochs,
                            "batch_size": batch_size,
                            "learning_rate_multiplier":learning_rate_multiplier
        }
    )

    job_id = response.id

    # job ID can be used to monitor the status of the fine-tuning job.

    print("Job ID:", response.id)
    print("Status:", response.status)
    print(response.model_dump_json(indent=2))
    

def upload_fine_tune_files(client, training_file_name, validation_file_name):
    training_response = client.files.create(
        file=open(training_file_name, "rb"), purpose="fine-tune"
    )
    training_file_id = training_response.id
    training_file_status = training_response.status

    validation_response = client.files.create(
        file=open(validation_file_name, "rb"), purpose="fine-tune"
    )
    validation_file_id = validation_response.id
    validation_file_status = validation_response.status
    
    while (validation_file_status not in ["processed", "failed"]) or (training_file_status not in ["processed", "failed"]): # until both files are finished uploading
        validation_file_status = client.files.retrieve(validation_response.id).status
        training_file_status = client.files.retrieve(training_response.id).status
        time.sleep(5)

    print('Training and validation files uploaded successfully.')
    print("Training file ID:", training_file_id)
    print("Validation file ID:", validation_file_id)
    return training_file_id, validation_file_id
    
    
def fine_tune_job(suffix_start, client, training_file_name, validation_file_name, base_model, n_epochs, batch_size, learning_rate_multiplier):
    training_file_id, validation_file_id = upload_fine_tune_files(client, training_file_name, validation_file_name)
    suffix = suffix_start+'_batch'+str(batch_size)+'_lrMlt'+str(learning_rate_multiplier)+'_n_epochs'+str(n_epochs)
    print(suffix)
    
    fine_tune_gpt(client, base_model, training_file_id, validation_file_id, suffix, n_epochs, batch_size, learning_rate_multiplier)


# obtaining the responses of a deployed model on an instances of a dataset
# dataset_json should contain data in the requested GPT fine-tuning format
def obtain_api_response(client, dataset_json, deployment_name):
    with open(dataset_json, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]
    
    responses = []
    concent_management_policy_violations = []
    for instance in tqdm(dataset):
        try:
            response = client.chat.completions.create(
                model=deployment_name,
                messages = instance['messages']
                #temperature=1.9
            )
            #print(instance['messages'][1])
            responses.append(response.choices[0].message.content)
        except Exception as e:
            #print("Caught BadRequestError:", e)
            concent_management_policy_violations.append(instance['messages'][1])
            responses.append(instance['messages'][1]['content'])
    
    print('Number of content management policy violations: ', len(concent_management_policy_violations))
    
    return responses


def extract_entity_indexes(txt, label_code):
    indx_enttype = {}  #key= (start, end), value=ent_type
    for ent_type in label_code.values():
        start_string = '<' + ent_type + '>' 
        end_string = '<\\' + ent_type + '>'
        pattern = re.compile(f'{re.escape(start_string)}(.*?){re.escape(end_string)}')
        for match in re.finditer(pattern, txt):
            indx_enttype[(match.start(), match.end())] = ent_type

    indx_enttype_sorted = dict(sorted(indx_enttype.items(), key=lambda item: item[0][0]))
    offset = 0
    originalIndex_enttype_sorted = {}
    for indx, enttype in indx_enttype_sorted.items():
        start_string = '<' + enttype + '>' 
        end_string = '<\\' + enttype + '>'
        shift = len(start_string) + len(end_string)
        originalIndex_enttype_sorted[(indx[0]-offset), indx[1]-(shift+offset)] = enttype
        offset+=shift

    return originalIndex_enttype_sorted


def performance_calc(label, pred, ent_type, label_code):
    label_indx_enttype = extract_entity_indexes(label, label_code)
    pred_indx_enttype = extract_entity_indexes(pred, label_code)

    tp=0 
    fp=0
    fn=0
    for indx, enttype in label_indx_enttype.items():
        if (ent_type is not None) and (enttype != ent_type):
            continue
        if (indx in pred_indx_enttype) and (enttype==pred_indx_enttype[indx]):
            tp+=1
        else:
            fn+=1
            
    for indx, enttype in pred_indx_enttype.items():
        if (ent_type is not None) and (enttype != ent_type):
            continue
        if (indx not in label_indx_enttype) or ((indx in label_indx_enttype) and (enttype!=label_indx_enttype[indx])):
            fp+=1

    return tp, fp, fn


def precision_recall_f1(entity_type, predictions, labels, label_code):
    tp_count=0
    fp_count=0
    fn_count=0
    for lbl, pred in zip(labels, predictions):
        tp, fp, fn = performance_calc(lbl, pred, entity_type, label_code)
        tp_count+=tp
        fp_count+=fp
        fn_count+=fn

    if (tp_count+fp_count)>0:
        precision = tp_count/(tp_count+fp_count)
    else:
        precision = 0

    if (tp_count+fn_count)>0:
        recall = tp_count/(tp_count+fn_count)
    else:
        recall = 0
    
    if (precision+recall)>0:
        f1 = 2*(precision*recall)/(precision+recall)
    else:
        f1 = 0

    return {'true_positives':tp_count, 'false_positives':fp_count, 'false_negatives':fn_count, 'Precision':precision, 'Recall':recall, 'F-1 score':f1}


def compute_performance(test_dataset_all_data_path, responses, label_code):
    # obtain labels for the test dataset
    test_dataset_all_data = pd.read_json(test_dataset_all_data_path, lines=True)
    labels = list(test_dataset_all_data['freq_str_labeled'])
    
    print(precision_recall_f1(None, responses, labels, label_code))
    
    
def write_responses_to_file(responses, label_code, output_file):
    def tp_fp_fn(X):
        label= X['freq_str_labeled']
        pred = X['Predictions']
        
        label_indx_enttype = extract_entity_indexes(label, label_code)
        pred_indx_enttype = extract_entity_indexes(pred, label_code)

        tp=0 
        fp=0
        fn=0
        for indx, enttype in label_indx_enttype.items():
            if (indx in pred_indx_enttype) and (enttype==pred_indx_enttype[indx]):
                tp+=1
            else:
                fn+=1
                
        for indx, enttype in pred_indx_enttype.items():
            if (indx not in label_indx_enttype) or ((indx in label_indx_enttype) and (enttype!=label_indx_enttype[indx])):
                fp+=1

        return tp, fp, fn
    
    
    test_dataset_all_data = pd.read_json(test_dataset_all_data_path, lines=True)
    test_dataset_all_data['Predictions'] = responses
    
    test_dataset_all_data['tp?'] = test_dataset_all_data.apply(tp_fp_fn, axis=1).apply(lambda x: x[0])
    test_dataset_all_data['fp?'] = test_dataset_all_data.apply(tp_fp_fn, axis=1).apply(lambda x: x[1])
    test_dataset_all_data['fn?'] = test_dataset_all_data.apply(tp_fp_fn, axis=1).apply(lambda x: x[2])
    
    test_dataset_all_data.to_excel(output_file, index=False)
    
    
def response_job(client, dataset_json, deployment_name, label_code, output_excel_file):
    responses = obtain_api_response(client, dataset_json, deployment_name)
    #compute_performance(test_dataset_all_data_path, responses, label_code)
    write_responses_to_file(responses, label_code, output_excel_file)
    

def main():
    
    #python Extraction_with_GPT.py --job_type=responses --freq_phrase=no --dataset_json=GPT/test_dataset_freqAttribute_gpt.jsonl --deployment_name=freqAttribute_batch4_lrMlt5_n_epochs8_train370 --test_dataset_all_data_path=GPT/test_dataset_freqAttribute_allData.jsonl --output_excel_file=Results/test_dataset_allData_gpt-35-turbo-freqAttr_batch4_epochs8_lrWeight5_train370.xlsx
    
    print('Start..')
    
    api_key = input('What is the azure key?')
    azure_endpoint = input('What is the azure end point?')
    
    client = AzureOpenAI(
        azure_endpoint = azure_endpoint, 
        api_key=api_key,  
        api_version="2023-12-01-preview"  # This API version or later is required to access fine-tuning for turbo/babbage-002/davinci-002
    )
    
    label_code_phrase = {'Frequency':'FREQ'}
    label_code_attribute = {'Value':'VAL', 'Interval':'INT', 'Unit':'UNT', 'Date':'DT', 'Min value':'MINVAL', 'Max value': 'MAXVAL', 'Semiology':'EVNT', 'Min interval':'MININT', 
              'Max interval':'MAXINT', 'Min date':'MINDT', 'Max date':'MAXDT', 'Periodic':'PERD', 'Age':'AGE', 'Min age':'MINAGE', 'Max age':'MAXAGE',  'Relative time period':'RELPR', 'Relative time point':'RELPT'}


    
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_type', help='Is the job for finetuning or obtaining responses?',
                        choices = ['finetune', 'responses'], required=True)
    # parser.add_argument('--freq_phrase', help='Is the job for finetuning or obtaining responses?',
    #                     choices = ['yes', 'no'])
    
    
    
    #group_finetune = parser.add_mutually_exclusive_group()
    group_finetune = parser.add_argument_group('Finetune argument group')
    group_finetune.add_argument('--suffix_start', help='prefix of the suffix')
    group_finetune.add_argument('--training_file_name', help='training file name')
    group_finetune.add_argument('--validation_file_name', help='validation file name')
    group_finetune.add_argument('--base_model', help='base model name')
    group_finetune.add_argument('--n_epochs', type=int, help='Number of epochs')
    group_finetune.add_argument('--batch_size', type=int, help='Batch size')
    group_finetune.add_argument('--lr_multiplier', type=int, help='Learning rate multiplier')
    
    #group_predict = parser.add_mutually_exclusive_group()
    group_predict = parser.add_argument_group('Predict argument group')
    group_predict.add_argument('--freq_phrase', help='Is the job for finetuning or obtaining responses?',
                        choices = ['yes', 'no'])
    group_predict.add_argument('--dataset_json', help='json with the instances to predict')
    group_predict.add_argument('--deployment_name', help='json with the instances to predict')    
    group_predict.add_argument('--test_dataset_all_data_path', help='test dataset with additional data')
    group_predict.add_argument('--output_excel_file', help='output excel file with predictions')
    
    
    
    args = parser.parse_args()
    if args.job_type=='responses' and args.freq_phrase is None:
        parser.error("For responses job, you need to specify whether it is for frequency phrase extraction or attribute extraction.")
    
    if args.job_type=='finetune':
        print('Finetune')
        fine_tune_job(args.suffix_start, client, args.training_file_name, args.validation_file_name, args.base_model, args.n_epochs, args.batch_size, args.learning_rate_multiplier)
    elif args.job_type=='responses':
        if args.freq_phrase=='yes':
            print('Seizure phrase extraction - API calls')
            response_job(client, args.dataset_json, args.deployment_name, args.test_dataset_all_data_path, label_code_phrase, args.output_excel_file)
        elif args.freq_phrase=='no':
            print('Seizure attribute extraction - API calls')
            response_job(client, args.dataset_json, args.deployment_name, args.test_dataset_all_data_path, label_code_attribute, args.output_excel_file)
        
    
    
    # suffix_start = sys.argv[1]
    # training_file_name = sys.argv[2]
    # validation_file_name = sys.argv[3]
    # base_model = sys.argv[4] 
    # n_epochs = int(sys.argv[5])
    # batch_size = int(sys.argv[6])
    # learning_rate_multiplier = int(sys.argv[7])
    
    
    #fine_tune_job(suffix_start, client, training_file_name, validation_file_name, base_model, n_epochs, batch_size, learning_rate_multiplier)
    
    
    
    ## for Seizure Frequency Phrase Extraction
    # python3 Extraction_with_GPT.py freqPhrase GPT/train_dataset_370_freqPhrase_gpt.jsonl GPT/validation_dataset_gpt.jsonl gpt-35-turbo-0613 8 4 5
    # python3 Extraction_with_GPT.py freqPhrase GPT/train_dataset_270_freqPhrase_gpt.jsonl GPT/validation_dataset_gpt.jsonl gpt-35-turbo-0613 8 4 5
    # python3 Extraction_with_GPT.py freqPhrase GPT/train_dataset_170_freqPhrase_gpt.jsonl GPT/validation_dataset_gpt.jsonl gpt-35-turbo-0613 8 4 5
    # python3 Extraction_with_GPT.py freqPhrase GPT/train_dataset_70_freqPhrase_gpt.jsonl GPT/validation_dataset_gpt.jsonl gpt-35-turbo-0613 8 4 5
    
    ## for Seizure Frequency Attribute Extraction
    # python3 Extraction_with_GPT.py freqAttribute GPT/train_dataset_370_freqAttribute_gpt.jsonl GPT/validation_dataset_freqAttribute_gpt.jsonl gpt-35-turbo-0613 8 4 5
    # python3 Extraction_with_GPT.py freqAttribute GPT/train_dataset_270_freqAttribute_gpt.jsonl GPT/validation_dataset_freqAttribute_gpt.jsonl gpt-35-turbo-0613 8 4 5
    # python3 Extraction_with_GPT.py freqAttribute GPT/train_dataset_170_freqAttribute_gpt.jsonl GPT/validation_dataset_freqAttribute_gpt.jsonl gpt-35-turbo-0613 8 4 5
    # python3 Extraction_with_GPT.py freqAttribute GPT/train_dataset_70_freqAttribute_gpt.jsonl GPT/validation_dataset_freqAttribute_gpt.jsonl gpt-35-turbo-0613 8 4 5
            
    print('End.')
    
if __name__ == "__main__":
    main()