#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

import pandas as pd
import numpy as np
from datasets import Dataset, concatenate_datasets, load_metric, Features, load_dataset
import transformers
from transformers import AutoTokenizer, LlamaTokenizer, get_linear_schedule_with_warmup
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, LlamaForCausalLM, BitsAndBytesConfig, AutoModelForCausalLM
from transformers import DataCollatorForTokenClassification, DataCollatorWithPadding, DataCollatorForSeq2Seq, default_data_collator
from transformers import pipeline
import torch
import copy
import csv
#import random
from torch.utils.data import DataLoader
from tqdm import tqdm
import re
import sys

from trl import SFTTrainer
import json


#ran using all GPUs
from transformers import set_seed
seed = 379
set_seed(seed)

from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model


from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftType,
    PromptEncoderConfig,
    PromptTuningConfig,
    PromptTuningInit,
    AutoPeftModelForCausalLM
)

import evaluate


def load_annotation_file_to_dataframe(annotation_file):
    annotation_df = pd.read_json(path_or_buf=annotation_file, lines=True)
    annotation_df = annotation_df.rename(columns={'text':'original_text'})
    return annotation_df


def replace_element_by_indexes(input_str, freq_dicts, label_code):
    freq_dicts_sorted = sorted(freq_dicts, key=lambda d: d['start_offset']) 

    # Convert the input string to a list of characters
    str_list = list(input_str)
    shift = 0
    for freq_dict in freq_dicts_sorted:
        freq_start_delim = '<' + label_code[freq_dict['label']] + '>'
        freq_end_delim = '<\\' + label_code[freq_dict['label']] + '>'

        new_start_index = freq_dict['start_offset'] + shift # the addition of characters to surround a certain freq string changes the start/end indexes of the rest of the freq strings
        new_end_index = freq_dict['end_offset'] + shift

        shift += len(freq_start_delim) + len(freq_end_delim)
        
        replacement = freq_start_delim+input_str[freq_dict['start_offset']:freq_dict['end_offset']]+freq_end_delim
    
        # Replace the substring with the replacement string
        str_list[new_start_index:new_end_index] = list(replacement)

    return ''.join(str_list)


def replace_element(X, label_code, is_frequency_identification):
    elem_dicts = []
    for ent in X['entities']:
        if (not is_frequency_identification and ent['label'] != 'Frequency') or (is_frequency_identification and ent['label'] == 'Frequency'):
            elem_dicts.append(ent)
    if len(elem_dicts)>0:
        return replace_element_by_indexes(X['original_text'], elem_dicts, label_code)
    else:
        return X['original_text']
            

def generate_instruction_for_training(x):
	return f"""### Instruction:
Annotate the seizure frequencies in the input text.

### Input:
{x['original_text']}

### Response:
{x['freq_str_labeled']}
"""


def generate_instruction_for_prediction(x):
	return f"""### Instruction:
Annotate the seizure frequencies in the input text.

### Input:
{x['original_text']}

### Response:
"""

def load_and_preprocess_dataset_LLAMA2(annotation_file_path, label_code, is_frequency_identification):
    annotation_df = pd.read_json(path_or_buf=annotation_file_path, lines=True)
    annotation_df = annotation_df.rename(columns={'text':'original_text'})
    annotation_df['freq_str_labeled'] = annotation_df.apply(replace_element, label_code=label_code, is_frequency_identification=is_frequency_identification, axis=1)
    annotation_df['text'] = annotation_df.apply(generate_instruction_for_training, axis=1)
    all_data=Dataset.from_pandas(annotation_df[['original_text', 'entities', 'freq_str_labeled', 'text']])
    return all_data


# def load_and_preprocess_dataset_LLAMA2(annotation_df, label_code, is_frequency_identification):
#     annotation_df['freq_str_labeled'] = annotation_df.apply(replace_element, label_code=label_code, axis=1)
#     annotation_df['text'] = annotation_df.apply(generate_instruction_for_training, axis=1)
#     all_data=Dataset.from_pandas(annotation_df[['original_text', 'entities', 'freq_str_labeled', 'text']])
#     return all_data


def seperating_train_validation_test_LLAMA2(all_data, train_subset_size=None):
    
    def update_dataset_for_prediction(example):
        example['text'] = generate_instruction_for_prediction(example)
        return example
    
    train1 = all_data.select(range(400))
    augmented_samples_old = all_data.select(range(500, 538))

    test_dataset = concatenate_datasets([all_data.select(range(400, 500)), all_data.select(range(538, 638))])
    validation_dataset = all_data.select(range(638, 838))

    augmented_samples_new = all_data.select(range(838, 908))
    train_dataset = concatenate_datasets([train1, augmented_samples_new])
    
    if train_subset_size is not None:
        train_dataset = train_dataset.shuffle(seed=37).select(range(train_subset_size))
        #train_dataset = train_dataset.select(range(train_subset_size))
        
    test_dataset = test_dataset.map(update_dataset_for_prediction)
    validation_dataset = validation_dataset.map(update_dataset_for_prediction)
    return train_dataset, validation_dataset, test_dataset


# def seperating_train_validation_test_LLAMA2(all_data):
#     # removing original augmented samples (500-538), picking datasets based on the original order
#     # new augmented samples: 838 onwards
#     train1 = all_data.select(range(400))
#     augmented_samples_old = all_data.select(range(500, 538))

#     test_dataset = concatenate_datasets([all_data.select(range(400, 500)), all_data.select(range(538, 638))])
#     validation_dataset = all_data.select(range(638, 838))

#     augmented_samples_new = all_data.select(range(838, 908))
#     train_dataset = concatenate_datasets([train1, augmented_samples_new])
#     return train_dataset, validation_dataset, test_dataset


# def update_dataset_for_prediction(example):
#     example['text'] = generate_instruction_for_prediction(example)
#     return example


def run_training(model_checkpoint, checkpoints_path, tokenizer, train_dataset, seed, learning_rate, per_device_train_batch_size, num_train_epochs, per_device_eval_batch_size=1, model_save_path=None):
    
    def model_init():
        # LoRA config based on QLoRA paper
        peft_config = LoraConfig(
                lora_alpha = 32, #16,
                lora_dropout = 0.01, #0.1,
                r=64,
                bias="lora_only",
                task_type="CAUSAL_LM",
        )
        model = LlamaForCausalLM.from_pretrained(model_checkpoint, device_map='auto')
        #model = LlamaForCausalLM.from_pretrained(model_checkpoint)
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)
        return model

    args = TrainingArguments(
        checkpoints_path,
        #evaluation_strategy = "epoch",
        learning_rate= learning_rate, #0.001663848484454143, #0.00010840321077516575, #0.0001701984107966588,  #1e-4,
        optim="adamw_torch",
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs, # 8,
        #weight_decay= 1e-3, #0.01, #2.7516630551348117e-08,
        #report_to = "none",
        logging_steps = 50,
        #logging_strategy = 'epoch',
        #metric_for_best_model='overall_f1',
        #load_best_model_at_end=True,
        #save_total_limit = 2,
        #save_strategy = "epoch",
        report_to="none",
        seed = seed
    )
        
    max_seq_length = 100

    trainer = SFTTrainer(
        model = model_init(),
        #model=model,
        train_dataset=train_dataset,
        #eval_dataset=validation_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=args,
    )

    trainer.train()
    
    if model_save_path is not None:
        trainer.save_model(model_save_path)  
    return trainer


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
        

def extract_response(x):
    return x.split('### Response:')[1].split('###')[0].strip()

def get_overall_performance_LLAMA2(trained_model, dataset, tokenizer, label_code):
    
    def performance_calc(label, pred, ent_type):
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
    
    def precision_recall_f1(entity_type, predictions):
        tp_count=0
        fp_count=0
        fn_count=0
        for pred in predictions:
            pred_response = extract_response(pred['prediction'])
            tp, fp, fn = performance_calc(pred['freq_str_labeled'], pred_response, entity_type)
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
        
    predictions = []
    with torch.no_grad():
        for sample in tqdm(dataset):
            input_ids = tokenizer(sample['text'], return_tensors="pt", truncation=True).input_ids.cuda()
            outputs = trained_model.generate(input_ids=input_ids, max_new_tokens=100, temperature=0.01)
            output_str = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]#[len(prompt):]
            predictions.append({'original_text':sample['original_text'], 'freq_str_labeled':sample['freq_str_labeled'], 'text':sample['text'], 'prediction':output_str})
    
    return(precision_recall_f1(None, predictions), predictions)


def predict_and_write_to_file(predictions, code_to_label, output_file_name, label_code):
    res = []
    for pred in predictions:
        lbl_ents = []
        for (start, end), ent_type in extract_entity_indexes(pred['freq_str_labeled'], label_code).items():
            lbl_ents.append({'entity_group': code_to_label[ent_type],
                            'word': pred['original_text'][start:end],
                            'start': start,
                            'end': end})
        pred_ents = []
        for (start, end), ent_type in extract_entity_indexes(extract_response(pred['prediction']), label_code).items():
            pred_ents.append({'entity_group': code_to_label[ent_type],
                            'word': pred['original_text'][start:end],
                            'start': start,
                            'end': end})
        
        res.append({"text":pred['original_text'], "label":lbl_ents, "entities":pred_ents})

    res_df = pd.DataFrame(res)
    res_df.to_json(output_file_name)
    
    
def write_generations_to_file(predictions, ouput_file):
    pd.DataFrame(predictions).to_json(ouput_file)
    

def entity_extraction_LLAMA2(model_checkpoint, performance_output=None, train_size=None):
    print('Entity Extraction')
    
    label_code = {'Value':'VAL', 'Interval':'INT', 'Unit':'UNT', 'Date':'DT', 'Min value':'MINVAL', 'Max value': 'MAXVAL', 'Semiology':'EVNT', 'Min interval':'MININT', 
              'Max interval':'MAXINT', 'Min date':'MINDT', 'Max date':'MAXDT', 'Periodic':'PERD', 'Age':'AGE', 'Min age':'MINAGE', 'Max age':'MAXAGE',  'Relative time period':'RELPR', 'Relative time point':'RELPT'}

    code_to_label = {v:k for k,v in label_code.items()}
    
    #seed = 37
    transformer_cache = '/data/rabeysinghe/huggingface_transformers_cache'
    #model_checkpoint = "meta-llama/Llama-2-70b-hf"
    #model_checkpoint = "meta-llama/Llama-2-7b-hf"
    #model_identifier = model_checkpoint.replace('/', '_') + '_' + 'instructionTune_newDB'
    
    if train_size is None:
        model_identifier = model_checkpoint.replace('/', '_') + '_' + 'entityExtraction_seed'+str(seed)
    else:
        model_identifier = model_checkpoint.replace('/', '_') + '_' + 'entityExtraction_'+str(train_size)+'_seed'+str(seed)
    
    checkpoints_path = 'checkpoints/'+model_identifier
    tokenizer = LlamaTokenizer.from_pretrained(model_checkpoint, cache=transformer_cache)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    annotation_file = 'data/annotated/annotated_backup_838_Oct10_parsingFixed_annotationsFixed_newAnnoAdded_relTimePeriodAndPoint_final.jsonl'
    output_file_name = 'results/'+model_identifier+'.jsonl'
    performance_file_name = 'experiment_results/'+model_identifier+'_'+performance_output+'.txt'
    model_save_path = 'saved_models/'+model_identifier+'.model'
    
    learning_rate = 1e-3
    per_device_train_batch_size = 4
    num_train_epochs = 8
    
    is_frequency_identification = False
    
    
    all_data = load_and_preprocess_dataset_LLAMA2(annotation_file, label_code, is_frequency_identification)
    #print(all_data)
    train_dataset, validation_dataset, test_dataset = seperating_train_validation_test_LLAMA2(all_data, train_subset_size=train_size)
    # train_dataset = train_dataset.select(range(20))
    # test_dataset = test_dataset.select(range(20))
   
    
    trainer = run_training(model_checkpoint, checkpoints_path, tokenizer, train_dataset, seed, learning_rate, per_device_train_batch_size, num_train_epochs, per_device_eval_batch_size=1, model_save_path=model_save_path)
    #model_loaded = trainer.model.merge_and_unload()
    model_loaded = trainer.model
    model_loaded = model_loaded.merge_and_unload()
    tokenizer = trainer.tokenizer
    performance, predictions = get_overall_performance_LLAMA2(model_loaded, test_dataset, tokenizer, label_code)
    #print(performance)
    
    if performance_output is not None:
        performance_file_name = 'experiment_results/'+model_identifier+'.txt'
        with open(performance_file_name, 'w') as f:
            json.dump(performance, f)
        
    predict_and_write_to_file(predictions, code_to_label, output_file_name, label_code)
            
    generations_file_name = 'LLAMA2_generations/'+model_identifier+'_llama2_generations.jsonl'
    write_generations_to_file(predictions, generations_file_name)
    
    
# def entity_extraction_training_size_experiment(results_file):
#     label_code = {'Value':'VAL', 'Interval':'INT', 'Unit':'UNT', 'Date':'DT', 'Min value':'MINVAL', 'Max value': 'MAXVAL', 'Semiology':'EVNT', 'Min interval':'MININT', 
#               'Max interval':'MAXINT', 'Min date':'MINDT', 'Max date':'MAXDT', 'Periodic':'PERD', 'Age':'AGE', 'Min age':'MINAGE', 'Max age':'MAXAGE',  'Relative time period':'RELPR', 'Relative time point':'RELPT'}

#     code_to_label = {v:k for k,v in label_code.items()}
    
#     seed = 37
#     transformer_cache = '/data/rabeysinghe/huggingface_transformers_cache'
#     model_checkpoint = "meta-llama/Llama-2-7b-hf" #"meta-llama/Llama-2-70b-hf"
#     model_identifier = model_checkpoint.replace('/', '_') + '_' + 'instructionTune_newDB'
#     checkpoints_path = 'checkpoints/'+model_identifier
    
#     annotation_file = 'data/annotated/annotated_backup_838_Oct10_parsingFixed_annotationsFixed_newAnnoAdded_relTimePeriodAndPoint_final.jsonl'
#     output_file_name = 'results/'+model_identifier+'_2.jsonl'
    
#     learning_rate = 1e-3
#     per_device_train_batch_size = 4
#     num_train_epochs = 8
    
#     is_frequency_identification = True
   
#     trainSize_performance = {}
#     for train_size in range(470, 0, -200):
#         all_data = load_and_preprocess_dataset_LLAMA2(annotation_file, label_code, is_frequency_identification)
#         #all_data_copy = copy.deepcopy(all_data)
#         train_dataset, validation_dataset, test_dataset = seperating_train_validation_test_LLAMA2(all_data, train_subset_size=train_size)
#         train_dataset = train_dataset.select(range(30))
#         test_dataset = test_dataset.select(range(30))
        
#         tokenizer = LlamaTokenizer.from_pretrained(model_checkpoint, cache=transformer_cache)
#         tokenizer.pad_token_id = tokenizer.eos_token_id
        
#         trainer = run_training(model_checkpoint, checkpoints_path, tokenizer, train_dataset, seed, learning_rate, per_device_train_batch_size, num_train_epochs, per_device_eval_batch_size=1, model_save_path=None)
#         model_loaded = trainer.model
#         model_loaded = model_loaded.merge_and_unload()
#         tokenizer = trainer.tokenizer
#         performance, predictions = get_overall_performance_LLAMA2(model_loaded, test_dataset, tokenizer, label_code)
#         trainSize_performance[train_size] = performance
        
#     print(trainSize_performance)
#     pd.DataFrame.from_dict(trainSize_performance, orient='index').to_csv(results_file)
        
        
def frequency_identification_LLAMA2(model_checkpoint, performance_output=None, train_size=None):
    print('Frequency Identification')
    # label_list = ['O', 'B-Frequency', 'I-Frequency']
    # label_encoding_dict = {'O':0 ,'B-Frequency':1, 'I-Frequency':2}

    # id_to_label = {v:k for k,v in label_encoding_dict.items()}
    
    label_code = {'Frequency':'FREQ'}
    code_to_label = {v:k for k,v in label_code.items()}
    
    
    #seed = 379
    transformer_cache = '/data/rabeysinghe/huggingface_transformers_cache'
    #model_checkpoint = "meta-llama/Llama-2-7b-hf" #"meta-llama/Llama-2-70b-hf"
    #model_checkpoint = "meta-llama/Llama-2-70b-hf"
    if train_size is None:
        model_identifier = model_checkpoint.replace('/', '_') + '_' + 'frequencyIdentification_seed'+str(seed)
    else:
        model_identifier = model_checkpoint.replace('/', '_') + '_' + 'frequencyIdentification_'+str(train_size)+'_seed'+str(seed)
        
    checkpoints_path = 'checkpoints/'+model_identifier
    tokenizer = LlamaTokenizer.from_pretrained(model_checkpoint, cache=transformer_cache)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    annotation_file = 'data/annotated/annotated_backup_838_Oct10_parsingFixed_annotationsFixed_newAnnoAdded_relTimePeriodAndPoint_final.jsonl'
    output_file_name = 'results/'+model_identifier+'.jsonl'
    model_save_path = 'saved_models/'+model_identifier+'.model'
    
    learning_rate = 1e-3
    per_device_train_batch_size = 4
    num_train_epochs = 8
    
    is_frequency_identification = True
    
    
    all_data = load_and_preprocess_dataset_LLAMA2(annotation_file, label_code, is_frequency_identification)
    #print(all_data)
    #train_dataset, validation_dataset, test_dataset = seperating_train_validation_test_LLAMA2(all_data, train_subset_size=None)
    train_dataset, validation_dataset, test_dataset = seperating_train_validation_test_LLAMA2(all_data, train_subset_size=train_size)
    # train_dataset = train_dataset.select(range(20))
    # test_dataset = test_dataset.select(range(20))
   
    
    trainer = run_training(model_checkpoint, checkpoints_path, tokenizer, train_dataset, seed, learning_rate, per_device_train_batch_size, num_train_epochs, per_device_eval_batch_size=1, model_save_path=model_save_path)
    #model_loaded = trainer.model.merge_and_unload()
    model_loaded = trainer.model
    model_loaded = model_loaded.merge_and_unload()
    tokenizer = trainer.tokenizer
    performance, predictions = get_overall_performance_LLAMA2(model_loaded, test_dataset, tokenizer, label_code)
    #print(performance)
    
    predict_and_write_to_file(predictions, code_to_label, output_file_name, label_code)
    
    if performance_output is not None:
        performance_file_name = 'experiment_results/'+model_identifier+'.txt'
        with open(performance_file_name, 'w') as f:
            json.dump(performance, f)
            
    generations_file_name = 'LLAMA2_generations/'+model_identifier+'_llama2_generations.jsonl'
    write_generations_to_file(predictions, generations_file_name)
    

def main():
    
    print('Start..')
    
    if (len(sys.argv) > 1) and (sys.argv[1]!="470"):
        train_size = int(sys.argv[1])
        #print(sys.argv[1])
    else:
        train_size = None
    
    model_checkpoint_7b = "meta-llama/Llama-2-7b-hf"
    model_checkpoint_70b = "meta-llama/Llama-2-70b-hf"
    #entity_extraction_LLAMA2(sys.argv[1], train_size)
    #entity_extraction_training_size_experiment('experiment_results/meta-llama_Llama-2-7b-hf_LLAMA2_entityPred.csv')
    if sys.argv[2]=='1':
        print('Frequency identification with train size: ', train_size)
        frequency_identification_LLAMA2(model_checkpoint_7b, sys.argv[1], train_size)
    elif sys.argv[2]=='2':
        print('Entity extraction with train size: ', train_size)
        entity_extraction_LLAMA2(model_checkpoint_7b, sys.argv[1], train_size)
    elif sys.argv[2]=='3':
        print('Entity extraction with LLAMA 2 70b')
        entity_extraction_LLAMA2(model_checkpoint_70b, sys.argv[1], train_size)
    elif sys.argv[2]=='4':
        print('Frequency identification with LLAMA 2 70b: ', train_size)
        frequency_identification_LLAMA2(model_checkpoint_70b, sys.argv[1], train_size)
            
    print('End.')
    
if __name__ == "__main__":
    main()

