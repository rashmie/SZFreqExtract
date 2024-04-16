import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd
import numpy as np
from datasets import Dataset, concatenate_datasets, load_metric, Features
import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
import torch
import copy
import csv
from transformers import pipeline


def tokenization(example, tokenizer):
    tokens = tokenizer(example["text"], truncation=True, return_offsets_mapping=True)

    token_to_char_list=[]
    for i, tid in enumerate(tokens['input_ids']):
        rg = tokens.token_to_chars(i)
        if rg is None:
            token_to_char_list.append(None)
        else:
            token_to_char_list.append((rg.start, rg.end))
    
    tokens['token_to_char_list']=token_to_char_list
    return tokens


# assigns IOB2 tags to tokens
def ner_tags_for_tokens(example, label_encoding_dict, is_frequency_identification):
    ner_tags = []
    prev_token_to_chars = None
    for i, token_to_chars in enumerate(example['token_to_char_list']):
        flag=False
        if token_to_chars is not None:
            for s in example['entities']:
                if (s['start_offset']<=token_to_chars[0]) and (s['end_offset']>=token_to_chars[1]):
                    original_ner_label = s['label']
                    if (not is_frequency_identification and original_ner_label=='Frequency') or (is_frequency_identification and original_ner_label!='Frequency'):
                        continue
                    
                    # if the previous token's start and end characters were within the same annotation
                    if prev_token_to_chars is not None and (s['start_offset']<=prev_token_to_chars[0]) and (s['end_offset']>=prev_token_to_chars[1]):
                        IOB2_ner_label = 'I-'+s['label']
                    else:
                        IOB2_ner_label = 'B-'+s['label']
                        
                    ner_tags.append(label_encoding_dict[IOB2_ner_label])
                    flag=True
                    break
            if not flag:        
                ner_tags.append(0)

        else: # this happens when 101=CLS or 102=SEP
            ner_tags.append(-100)
        
        prev_token_to_chars = token_to_chars
            
    example['labels'] = ner_tags
    return example


def load_and_preprocess_dataset(annotation_file_path, tokenizer, is_frequency_identification, label_encoding_dict):
    annotation_df = pd.read_json(path_or_buf=annotation_file_path, lines=True)
    all_data=Dataset.from_pandas(annotation_df[['text', 'entities']])
    all_data_tokenized = all_data.map(tokenization, batched=False, fn_kwargs={"tokenizer": tokenizer})
    return all_data_tokenized.map(ner_tags_for_tokens, batched=False, fn_kwargs={"is_frequency_identification": is_frequency_identification, "label_encoding_dict":label_encoding_dict})


def seperating_train_validation_test(all_data_tokenized, train_subset_size=None):
    train1 = all_data_tokenized.select(range(400))
    augmented_samples_old = all_data_tokenized.select(range(500, 538))

    test_dataset = concatenate_datasets([all_data_tokenized.select(range(400, 500)), all_data_tokenized.select(range(538, 638))])
    validation_dataset = all_data_tokenized.select(range(638, 838))

    augmented_samples_new = all_data_tokenized.select(range(838, 908))
    train_dataset = concatenate_datasets([train1, augmented_samples_new])
    
    if train_subset_size is not None:
        train_dataset = train_dataset.shuffle(seed=37).select(range(train_subset_size))
        
    return train_dataset, validation_dataset, test_dataset


def run_training(model_checkpoint, checkpoints_path, tokenizer, transformer_cache, train_dataset, validation_dataset, seed, learning_rate, per_device_train_batch_size, num_train_epochs, weight_decay, id_to_label, label_encoding_dict, label_list, per_device_eval_batch_size=1, model_save_path=None):
    
    def model_init():
        return AutoModelForTokenClassification.from_pretrained(model_checkpoint, id2label=id_to_label, label2id=label_encoding_dict, cache_dir=transformer_cache)

    args = TrainingArguments(
        checkpoints_path,
        evaluation_strategy = "epoch", #"epoch",
        optim="adamw_torch",
        learning_rate= learning_rate,
        per_device_train_batch_size=per_device_train_batch_size, 
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        save_total_limit = 1,
        #logging_strategy = 'epoch',
        #metric_for_best_model='overall_f1',
        #load_best_model_at_end=True,
        #save_strategy = "epoch",
        report_to="none",
        seed = seed
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)
    metric = load_metric("seqeval")
    
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # need to convert back to IOB2 format so that 'seqeval' can automatically compute entity level performance.
        # https://huggingface.co/spaces/evaluate-metric/seqeval
        true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
        true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
        
        # https://huggingface.co/spaces/evaluate-metric/seqeval
        results = metric.compute(predictions=true_predictions, references=true_labels, scheme='IOB2', mode="strict")
        return results
    
    trainer = Trainer(
        model_init=model_init,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.evaluate()
    if model_save_path is not None:
        trainer.save_model(model_save_path)  
    return trainer 
    
    
def get_tags(smpl, model, tokenizer, label_list):
    predictions = model.forward(input_ids=torch.tensor(smpl['input_ids']).unsqueeze(0).to('cuda:0'), attention_mask=torch.tensor(smpl['attention_mask']).unsqueeze(0).to('cuda:0'))
    predictions = torch.argmax(predictions.logits.squeeze(), axis=1)
    predictions = [label_list[i] for i in predictions]
    words = tokenizer.batch_decode(smpl['input_ids'])
    
    words_mod = []
    iob2_tags = []
    
    for i in zip(predictions, words):
        words_mod.append(i[1])
        if i[1]=='[CLS]' or i[1]=='[SEP]':
            iob2_tags.append('O')
            continue
        
        iob2_tags.append(i[0])
    
    
    # Initialize variables to store spans
    spans = []
    current_start = None
    current_label = None
    prev_token_end = None
    prev_label = None
    
    # Iterate through tokens and IOB2 tags
    for i, (token, offset_mapping) in enumerate(zip(smpl["input_ids"], smpl["offset_mapping"])):
        #print(words_mod[i], iob2_tags[i], current_start, current_label)
        #print(tokenizer.decode(token), ' **** ', offset_mapping)
        ent_type = iob2_tags[i][2:]
        if iob2_tags[i] == "O":
            # If outside of a named entity, reset current_start and current_label
            if current_start is not None:
                spans.append({'word':smpl['text'][current_start:prev_token_end], "start": current_start, "end": prev_token_end, "entity_group": current_label})
                #print(prev_token_end)
                current_start = None
                current_label = None
        elif iob2_tags[i].startswith("B-"):
            if current_start is not None:
                spans.append({'word':smpl['text'][current_start:prev_token_end], "start": current_start, "end": prev_token_end, "entity_group": current_label})
            # If beginning of a named entity, update current_start and current_label
            #print(i, offset_mapping[0])
            #print(words_mod[i])
            current_start = offset_mapping[0]
            current_label = iob2_tags[i][2:]
        elif iob2_tags[i].startswith("I-") and iob2_tags[i][2:]!=prev_label:
            if current_start is not None:
                spans.append({'word':smpl['text'][current_start:prev_token_end], "start": current_start, "end": prev_token_end, "entity_group": current_label})
                current_start = None
                current_label = None
    
        prev_token_end = offset_mapping[1]
        prev_label = iob2_tags[i][2:]
        #print(prev_token_end)
    
    # Check if there's a remaining named entity at the end
    if current_start is not None:
        spans.append({'word':txt[current_start:prev_token_end], "start": current_start, "end": prev_token_end, "entity_group": current_label})

    return spans


def get_overall_performance(trainer, dataset):
    performance = trainer.evaluate(dataset)
    return {'overall_precision':performance['eval_overall_precision'], 'overall_recall':performance['eval_overall_recall'], 
            'overall_f1':performance['eval_overall_f1']}
    

def predict_and_write_to_file(tokenizer, model, output_file_name, dataset, label_list):
    res = []
    for smpl in dataset:
        entities = get_tags(smpl, model, tokenizer, label_list)
        res.append({"text":smpl['text'], "label":smpl['entities'], "entities":entities})
        
    res_df = pd.DataFrame(res)
    #output_file_name = 'results/'+model_identifier+'.jsonl'
    res_df.to_json(output_file_name)
    
    
def entity_extraction():
    #index of each element of label_list correspond to the value of it in label_encoding_dict. Therefore, we don't need a reverse dictionary.
    label_list = ['O', 'B-Value', 'I-Value', 'B-Interval', 'I-Interval', 
                        'B-Unit', 'I-Unit', 'B-Date', 'I-Date', 
                        'B-Min value', 'I-Min value', 'B-Max value', 'I-Max value',
                        'B-Semiology', 'I-Semiology', 'B-Min interval', 'I-Min interval',
                        'B-Max interval', 'I-Max interval', 'B-Min date', 'I-Min date', 
                        'B-Max date', 'I-Max date', 'B-Periodic', 'I-Periodic', 
                        'B-Age', 'I-Age', 'B-Min age', 'I-Min age', 'B-Max age', 'I-Max age',
                        'B-Relative time period','I-Relative time period',
                        'B-Relative time point','I-Relative time point']

    label_encoding_dict = {'O':0 , 'B-Value':1, 'I-Value':2, 'B-Interval':3, 'I-Interval':4, 
                        'B-Unit':5, 'I-Unit':6, 'B-Date':7, 'I-Date':8, 
                        'B-Min value':9, 'I-Min value':10, 'B-Max value':11, 'I-Max value':12,
                        'B-Semiology':13, 'I-Semiology':14, 'B-Min interval':15, 'I-Min interval':16,
                        'B-Max interval':17, 'I-Max interval':18, 'B-Min date':19, 'I-Min date':20, 
                        'B-Max date':21, 'I-Max date':22, 'B-Periodic':23, 'I-Periodic':24,
                        'B-Age':25, 'I-Age':26, 'B-Min age':27, 'I-Min age':28, 'B-Max age':29, 'I-Max age':30,
                        'B-Relative time period':31,'I-Relative time period':32, 'B-Relative time point':33, 'I-Relative time point':34}

    id_to_label = {v:k for k,v in label_encoding_dict.items()}
    
    seed = 379
    transformer_cache = '/data/rabeysinghe/huggingface_transformers_cache'
    model_checkpoint = "bert-large-cased"
    model_identifier = model_checkpoint+'_'+'freq_entity_extract'
    checkpoints_path = 'checkpoints/'+model_identifier
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, cache=transformer_cache)
    annotation_file = 'data/annotated/annotated_backup_838_Oct10_parsingFixed_annotationsFixed_newAnnoAdded_relTimePeriodAndPoint_final.jsonl'
    output_file_name = 'results/'+model_identifier+'_2.jsonl'
    
    learning_rate = 0.00011607079483796333
    per_device_train_batch_size = 32
    num_train_epochs = 9
    weight_decay = 1.2921786790381586e-06
    
    all_data_tokenized = load_and_preprocess_dataset(annotation_file, tokenizer, False, label_encoding_dict)
    train_dataset, validation_dataset, test_dataset = seperating_train_validation_test(all_data_tokenized, train_subset_size=None)
    
    trainer = run_training(model_checkpoint, checkpoints_path, tokenizer, transformer_cache, train_dataset, validation_dataset, seed, learning_rate, per_device_train_batch_size, num_train_epochs, weight_decay, id_to_label, label_encoding_dict, label_list, per_device_eval_batch_size=1, model_save_path=None)
    performance = get_overall_performance(trainer, test_dataset)
    print(performance)
    
    predict_and_write_to_file(tokenizer, trainer.model, output_file_name, test_dataset, label_list)
    

def entity_extraction_training_size_experiment():
    ## NEW SEED=37 HYPERPARAMETERS
    #distilbert-base-cased
    # model_checkpoint = 'distilbert-base-uncased'
    # learning_rate = 0.0001027917675229495
    # per_device_train_batch_size = 2
    # num_train_epochs = 5
    # weight_decay = 2.3550715924278655e-08
    
    # bert-large-cased
    # model_checkpoint = "bert-large-cased"
    # learning_rate = 0.00011607079483796333
    # per_device_train_batch_size = 32
    # num_train_epochs = 9
    # weight_decay = 1.2921786790381586e-06
    
    # dmis-lab/biobert-large-cased-v1.1
    # model_checkpoint = "dmis-lab/biobert-large-cased-v1.1"
    # learning_rate = 5.262178601368295e-05
    # per_device_train_batch_size = 8
    # num_train_epochs = 6
    # weight_decay = 1.4152956020973918e-06
    
    # emilyalsentzer/Bio_ClinicalBERT
    # model_checkpoint = "emilyalsentzer/Bio_ClinicalBERT"
    # learning_rate = 0.00030281871280768045
    # per_device_train_batch_size = 32
    # num_train_epochs = 6
    # weight_decay = 0.001971963668136106
    
    
    
    ## NEW SEED=379 HYPERPARAMETERS
    #distilbert-base-cased
    # model_checkpoint = 'distilbert-base-uncased'
    # learning_rate = 8.26154097351868e-05
    # per_device_train_batch_size = 8
    # num_train_epochs = 8
    # weight_decay = 7.388545494128008e-08
    
    # bert-large-cased
    # model_checkpoint = "bert-large-cased"
    # learning_rate = 2.056610050966157e-05
    # per_device_train_batch_size = 32
    # num_train_epochs = 18
    # weight_decay = 1.749510057969075e-08
    
    # dmis-lab/biobert-large-cased-v1.1
    # model_checkpoint = "dmis-lab/biobert-large-cased-v1.1"
    # learning_rate = 1.6351712436807695e-05
    # per_device_train_batch_size = 32
    # num_train_epochs = 16
    # weight_decay = 1.2832970965809684e-07
    
    # emilyalsentzer/Bio_ClinicalBERT
    # model_checkpoint = "emilyalsentzer/Bio_ClinicalBERT"
    # learning_rate = 0.00010365662805780822
    # per_device_train_batch_size = 8
    # num_train_epochs = 4
    # weight_decay = 1.1756808272788447e-05
    
    
    ## NEW SEED=379 HYPERPARAMETERS, new conda environment
    #distilbert-base-cased
    # model_checkpoint = 'distilbert-base-uncased'
    # learning_rate = 6.024455314300926e-05
    # per_device_train_batch_size = 4
    # num_train_epochs = 5
    # weight_decay = 5.4139426756079545e-08
    
    # bert-large-cased
    # model_checkpoint = "bert-large-cased"
    # learning_rate = 7.292674315011053e-05
    # per_device_train_batch_size = 32
    # num_train_epochs = 16
    # weight_decay = 2.0591104679996593e-06
    
    # dmis-lab/biobert-large-cased-v1.1
    # model_checkpoint = "dmis-lab/biobert-large-cased-v1.1"
    # learning_rate = 5.850448839095984e-05
    # per_device_train_batch_size = 16
    # num_train_epochs = 7
    # weight_decay = 0.0023057979606364858
    
    # emilyalsentzer/Bio_ClinicalBERT
    # model_checkpoint = "emilyalsentzer/Bio_ClinicalBERT"
    # learning_rate = 0.00011249114894981418
    # per_device_train_batch_size = 4
    # num_train_epochs = 5
    # weight_decay = 1.1557717651107671e-06
    
    
        
    #index of each element of label_list correspond to the value of it in label_encoding_dict. Therefore, we don't need a reverse dictionary.
    label_list = ['O', 'B-Value', 'I-Value', 'B-Interval', 'I-Interval', 
                        'B-Unit', 'I-Unit', 'B-Date', 'I-Date', 
                        'B-Min value', 'I-Min value', 'B-Max value', 'I-Max value',
                        'B-Semiology', 'I-Semiology', 'B-Min interval', 'I-Min interval',
                        'B-Max interval', 'I-Max interval', 'B-Min date', 'I-Min date', 
                        'B-Max date', 'I-Max date', 'B-Periodic', 'I-Periodic', 
                        'B-Age', 'I-Age', 'B-Min age', 'I-Min age', 'B-Max age', 'I-Max age',
                        'B-Relative time period','I-Relative time period',
                        'B-Relative time point','I-Relative time point']

    label_encoding_dict = {'O':0 , 'B-Value':1, 'I-Value':2, 'B-Interval':3, 'I-Interval':4, 
                        'B-Unit':5, 'I-Unit':6, 'B-Date':7, 'I-Date':8, 
                        'B-Min value':9, 'I-Min value':10, 'B-Max value':11, 'I-Max value':12,
                        'B-Semiology':13, 'I-Semiology':14, 'B-Min interval':15, 'I-Min interval':16,
                        'B-Max interval':17, 'I-Max interval':18, 'B-Min date':19, 'I-Min date':20, 
                        'B-Max date':21, 'I-Max date':22, 'B-Periodic':23, 'I-Periodic':24,
                        'B-Age':25, 'I-Age':26, 'B-Min age':27, 'I-Min age':28, 'B-Max age':29, 'I-Max age':30,
                        'B-Relative time period':31,'I-Relative time period':32, 'B-Relative time point':33, 'I-Relative time point':34}

    id_to_label = {v:k for k,v in label_encoding_dict.items()}
    
    seed = 379
    transformer_cache = '/data/rabeysinghe/huggingface_transformers_cache'
    #model_checkpoint = "dmis-lab/biobert-large-cased-v1.1"
    #model_identifier = model_checkpoint+'_'+'frequency_str_extract'
    model_identifier = model_checkpoint.replace('/', '_') + '_' + 'frequency_str_extract'
    checkpoints_path = 'checkpoints/'+model_identifier
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, cache=transformer_cache)
    annotation_file = 'data/annotated/annotated_backup_838_Oct10_parsingFixed_annotationsFixed_newAnnoAdded_relTimePeriodAndPoint_final.jsonl'
    output_file_name = 'results/'+model_identifier+'_2.jsonl'
    
    # learning_rate = 1.6473871479558415e-05
    # per_device_train_batch_size = 8
    # num_train_epochs = 7
    # weight_decay = 0.0038322670889910882
    
    is_frequency_identification = False
    
    all_data_tokenized = load_and_preprocess_dataset(annotation_file, tokenizer, is_frequency_identification, label_encoding_dict)
    
    trainSize_performance = {}
    for train_size in range(470, 0, -100):
        all_data_tokenized_copy = copy.deepcopy(all_data_tokenized)
        if train_size == 470:
            train_dataset, validation_dataset, test_dataset = seperating_train_validation_test(all_data_tokenized_copy, train_subset_size=None)
        else:
            train_dataset, validation_dataset, test_dataset = seperating_train_validation_test(all_data_tokenized_copy, train_subset_size=train_size)
        trainer = run_training(model_checkpoint, checkpoints_path, tokenizer, transformer_cache, train_dataset, validation_dataset, seed, learning_rate, per_device_train_batch_size, num_train_epochs, weight_decay, id_to_label, label_encoding_dict, label_list, per_device_eval_batch_size=1, model_save_path=None)
        performance = get_overall_performance(trainer, test_dataset)
        trainSize_performance[train_size] = performance
        
    print(trainSize_performance)
    

def frequency_identification():
    label_list = ['O', 'B-Frequency', 'I-Frequency']
    label_encoding_dict = {'O':0 ,'B-Frequency':1, 'I-Frequency':2}
    id_to_label = {v:k for k,v in label_encoding_dict.items()}
    
    seed = 379
    transformer_cache = '/data/rabeysinghe/huggingface_transformers_cache'
    model_checkpoint = "dmis-lab/biobert-large-cased-v1.1"
    #model_identifier = model_checkpoint+'_'+'frequency_str_extract'
    model_identifier = model_checkpoint.replace('/', '_') + '_' + 'frequency_str_extract'
    checkpoints_path = 'checkpoints/'+model_identifier
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, cache=transformer_cache)
    annotation_file = 'data/annotated/annotated_backup_838_Oct10_parsingFixed_annotationsFixed_newAnnoAdded_relTimePeriodAndPoint_final.jsonl'
    output_file_name = 'results/'+model_identifier+'_2.jsonl'
    
    learning_rate = 1.6473871479558415e-05
    per_device_train_batch_size = 8
    num_train_epochs = 7
    weight_decay = 0.0038322670889910882
    
    is_frequency_identification = False
    
    all_data_tokenized = load_and_preprocess_dataset(annotation_file, tokenizer, is_frequency_identification, label_encoding_dict)
    train_dataset, validation_dataset, test_dataset = seperating_train_validation_test(all_data_tokenized, train_subset_size=None)
    
    trainer = run_training(model_checkpoint, checkpoints_path, tokenizer, transformer_cache, train_dataset, validation_dataset, seed, learning_rate, per_device_train_batch_size, num_train_epochs, weight_decay, id_to_label, label_encoding_dict, label_list, per_device_eval_batch_size=1, model_save_path=None)
    performance = get_overall_performance(trainer, test_dataset)
    print(performance)
    
    predict_and_write_to_file(tokenizer, trainer.model, output_file_name, test_dataset, label_list)
    
    
def frequency_identification_training_size_experiment():
    
    # OLD SEED=37 HYPERPARAMETERS
    #distilbert-base-cased
    # model_checkpoint = 'distilbert-base-uncased'
    # learning_rate = 6.591001044664817e-05
    # per_device_train_batch_size = 4
    # num_train_epochs = 3
    # weight_decay = 0.0009327639270343508
    
    # bert-large-cased
    # model_checkpoint = "bert-large-cased"
    # learning_rate = 6.464997320186087e-05
    # per_device_train_batch_size = 8
    # num_train_epochs = 4
    # weight_decay = 2.4805526392712795e-05
    
    # dmis-lab/biobert-large-cased-v1.1
    # model_checkpoint = "dmis-lab/biobert-large-cased-v1.1"
    # learning_rate = 1.6473871479558415e-05
    # per_device_train_batch_size = 8
    # num_train_epochs = 7
    # weight_decay = 0.0038322670889910882
    
    # emilyalsentzer/Bio_ClinicalBERT
    # model_checkpoint = "emilyalsentzer/Bio_ClinicalBERT"
    # learning_rate = 0.00024611976987903205
    # per_device_train_batch_size = 8
    # num_train_epochs = 3
    # weight_decay = 0.0004943983993546622
    
    
    # NEW SEED=379 HYPERPARAMETERS
    #distilbert-base-cased
    # model_checkpoint = 'distilbert-base-uncased'
    # learning_rate = 0.00012678448939122547
    # per_device_train_batch_size = 16
    # num_train_epochs = 3
    # weight_decay = 8.928584367972738e-08
    
    # bert-large-cased
    # model_checkpoint = "bert-large-cased"
    # learning_rate = 0.00010174162734224253
    # per_device_train_batch_size = 32
    # num_train_epochs = 7
    # weight_decay = 0.0005074627263751504
    
    # dmis-lab/biobert-large-cased-v1.1
    # model_checkpoint = "dmis-lab/biobert-large-cased-v1.1"
    # learning_rate = 4.337281949888043e-05
    # per_device_train_batch_size = 8
    # num_train_epochs = 3
    # weight_decay = 0.0008051200815205763
    
    # emilyalsentzer/Bio_ClinicalBERT
    # model_checkpoint = "emilyalsentzer/Bio_ClinicalBERT"
    # learning_rate = 0.00019327168446193357
    # per_device_train_batch_size = 8
    # num_train_epochs = 4
    # weight_decay = 0.00043045088080879713
    
    
    # NEW SEED=379 HYPERPARAMETERS, new conda environment
    #distilbert-base-cased
    # model_checkpoint = 'distilbert-base-uncased'
    # learning_rate = 8.336582516229313e-05
    # per_device_train_batch_size = 16
    # num_train_epochs = 4
    # weight_decay = 5.01811145728112e-06
    
    # bert-large-cased
    # model_checkpoint = "bert-large-cased"
    # learning_rate = 0.00010962472970355563
    # per_device_train_batch_size = 8
    # num_train_epochs = 3
    # weight_decay = 6.175068452799282e-06
    
    # dmis-lab/biobert-large-cased-v1.1
    # model_checkpoint = "dmis-lab/biobert-large-cased-v1.1"
    # learning_rate = 3.732277916659287e-05
    # per_device_train_batch_size = 8
    # num_train_epochs = 4
    # weight_decay = 0.0019605681370994054
    
    # emilyalsentzer/Bio_ClinicalBERT
    # model_checkpoint = "emilyalsentzer/Bio_ClinicalBERT"
    # learning_rate = 8.288629791776871e-05
    # per_device_train_batch_size = 16
    # num_train_epochs = 4
    # weight_decay = 0.0027543913222117405
    
    
    label_list = ['O', 'B-Frequency', 'I-Frequency']
    label_encoding_dict = {'O':0 ,'B-Frequency':1, 'I-Frequency':2}
    id_to_label = {v:k for k,v in label_encoding_dict.items()}
    
    seed = 379
    transformer_cache = '/data/rabeysinghe/huggingface_transformers_cache'
    #model_checkpoint = "dmis-lab/biobert-large-cased-v1.1"
    #model_identifier = model_checkpoint+'_'+'frequency_str_extract'
    model_identifier = model_checkpoint.replace('/', '_') + '_' + 'frequency_str_extract'
    checkpoints_path = 'checkpoints/'+model_identifier
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, cache=transformer_cache)
    annotation_file = 'data/annotated/annotated_backup_838_Oct10_parsingFixed_annotationsFixed_newAnnoAdded_relTimePeriodAndPoint_final.jsonl'
    output_file_name = 'results/'+model_identifier+'_2.jsonl'
    
    # learning_rate = 1.6473871479558415e-05
    # per_device_train_batch_size = 8
    # num_train_epochs = 7
    # weight_decay = 0.0038322670889910882
    
    is_frequency_identification = True
    
    all_data_tokenized = load_and_preprocess_dataset(annotation_file, tokenizer, is_frequency_identification, label_encoding_dict)
    
    trainSize_performance = {}
    for train_size in range(470, 0, -100):
        all_data_tokenized_copy = copy.deepcopy(all_data_tokenized)
        if train_size == 470:
            train_dataset, validation_dataset, test_dataset = seperating_train_validation_test(all_data_tokenized_copy, train_subset_size=None)
        else:
            train_dataset, validation_dataset, test_dataset = seperating_train_validation_test(all_data_tokenized_copy, train_subset_size=train_size)
        trainer = run_training(model_checkpoint, checkpoints_path, tokenizer, transformer_cache, train_dataset, validation_dataset, seed, learning_rate, per_device_train_batch_size, num_train_epochs, weight_decay, id_to_label, label_encoding_dict, label_list, per_device_eval_batch_size=1, model_save_path=None)
        performance = get_overall_performance(trainer, test_dataset)
        trainSize_performance[train_size] = performance
        
    print(trainSize_performance)
        
        
def main():
    print('Start..')
    
    #entity_extraction()
    #frequency_identification()
    
    #frequency_identification_training_size_experiment()
    entity_extraction_training_size_experiment()
    
    print('End.')
    
if __name__ == "__main__":
    main()

