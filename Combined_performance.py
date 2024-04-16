import pandas as pd

def load_entity_extraction_output_to_dataframe(input_json_file):
    ent_df = pd.read_json(path_or_buf=input_json_file)
    

def load_frequency_identification_output_to_dataframe(inpu_json_file):
    freq_df = pd.read_json(path_or_buf=inpu_json_file)
    
def reformat_label(x, col):
    reformatted = set()
    text = x['text']
    for ent in x[col]:
        ent_grp = ent['label']
        word = text[ent['start_offset']: ent['end_offset']]
        start = ent['start_offset']
        end = ent['end_offset']
        reformatted.add((ent_grp, word, start, end))
    
    return reformatted

def reformat_prediction(x, col):
    reformatted = set()
    text = x['text']
    for ent in x[col]:
        ent_grp = ent['entity_group']
        start = ent['start']
        end = ent['end']
        word = text[int(start):int(end)] #ent['word']. Some uncased models may change upper case letters to lower case
        reformatted.add((ent_grp, word, start, end))
        
    return reformatted

def get_frequency_strings(x):
    freq_string_set = set()
    for a in x:
        #print(a)
        if a[0]=='Frequency':
        #if a['entity_group']=='Frequency':
            freq_string_set.add(a)            
    return freq_string_set

def get_entities(x):
    entity_set = set()
    for a in x:
        #print(a)
        if a[0]!='Frequency':
            entity_set.add(a) 
    return entity_set
            
def entities_overlap_with_freq_string(ent_set, freq_set):
    if len(freq_set)==0: # no frequency strings identified
        return
    #print('AAAA')
    seperated_ents = set()
    for frq in freq_set:
        ent_list = set()
        for ent in ent_set:
            if (ent[2]>=frq[2]) and (ent[3]<=frq[3]):
                ent_list.add(ent)
        if len(ent_list)>0:
            seperated_ents.add(frozenset(ent_list))
    
    if len(seperated_ents)==0:
        return
    return seperated_ents
                

def agrees_with_gold(X):
    pred_freqs = X['freq string']
    pred_entities = X['entities']
    label_freqs = get_frequency_strings(X['freq string label'])
    label_entities = get_entities(X['label'])
    
    pred_entities_overlapping_freq_strings = entities_overlap_with_freq_string(pred_entities, pred_freqs)
    label_entities_overlapping_freq_strings = entities_overlap_with_freq_string(label_entities, label_freqs)
        
    num_freq_gold = 0 if label_entities_overlapping_freq_strings is None else len(label_entities_overlapping_freq_strings)
    num_freq_pred = 0 if pred_entities_overlapping_freq_strings is None else len(pred_entities_overlapping_freq_strings)
    num_correct_freq_pred = 0 if (num_freq_gold==0 or num_freq_pred==0) else len(pred_entities_overlapping_freq_strings.intersection(label_entities_overlapping_freq_strings))
    
    return (num_freq_gold, num_freq_pred, num_correct_freq_pred)   


def agrees_with_gold_individual(X, pred_col, label_col, is_freq_phrase):
    if is_freq_phrase:
        labels = get_frequency_strings(X[label_col])
        preds = X[pred_col]
    else:
        labels = get_entities(X[label_col])
        preds = X[pred_col]
 
    num_freq_gold = 0 if labels is None else len(labels)
    num_freq_pred = 0 if preds is None else len(preds)
    num_correct_freq_pred = 0 if (num_freq_gold==0 or num_freq_pred==0) else len(preds.intersection(labels))
    return (num_freq_gold, num_freq_pred, num_correct_freq_pred)
    

def get_freqPhrase_precision_recall_f1(input_json_file, is_bert):
    #ent_df = pd.read_json(path_or_buf=input_json_file_entity_pred)
    freq_df = pd.read_json(path_or_buf=input_json_file)
    
    
    # ent_df['freq string label'] = freq_df['label']
    # ent_df['freq string'] = freq_df['entities']
    
    # if is_entity_pred_bert:
    #     ent_df['label'] = ent_df.apply(reformat_label, axis=1, col='label')
    
    # ent_df['entities'] = ent_df.apply(reformat_prediction, axis=1, col='entities')
    freq_df['entities'] = freq_df.apply(reformat_prediction, axis=1, col='entities')
    
    # if not is_entity_pred_bert:
    #     ent_df['label'] = ent_df.apply(reformat_prediction, axis=1, col='label') # for LLAMA 2 entity pred models
    
    if not is_bert:
        freq_df['label'] = freq_df.apply(reformat_prediction, axis=1, col='label') # for LLAMA 2 freq identification models
    else:
        freq_df['label'] = freq_df.apply(reformat_label, axis=1, col='label') # for BERT freq identification model
        
        
    freq_df['num_freq_gold/num_freq_pred/num_correct_freq_pred'] = freq_df.apply(agrees_with_gold_individual, pred_col='entities', label_col='label', is_freq_phrase=True, axis=1)
    num_total_freqs_gold = freq_df['num_freq_gold/num_freq_pred/num_correct_freq_pred'].apply(lambda x: x[0]).sum()
    num_total_freqs_pred = freq_df['num_freq_gold/num_freq_pred/num_correct_freq_pred'].apply(lambda x: x[1]).sum()
    num_correct_freqs_pred = freq_df['num_freq_gold/num_freq_pred/num_correct_freq_pred'].apply(lambda x: x[2]).sum()
    
    precision = num_correct_freqs_pred/(num_total_freqs_pred)
    recall = num_correct_freqs_pred/(num_total_freqs_gold)
    f1 = (2*precision*recall)/(precision+recall)
    
    #print(num_correct_freqs_pred, num_total_freqs_pred, num_total_freqs_gold)
    
    return precision, recall, f1


def get_freqAttribute_precision_recall_f1(input_json_file, is_bert):
    ent_df = pd.read_json(path_or_buf=input_json_file)
    #freq_df = pd.read_json(path_or_buf=input_json_file)
    
    # ent_df['freq string label'] = freq_df['label']
    # ent_df['freq string'] = freq_df['entities']
    
    if is_bert:
        ent_df['label'] = ent_df.apply(reformat_label, axis=1, col='label')
    
    ent_df['entities'] = ent_df.apply(reformat_prediction, axis=1, col='entities')
    # ent_df['freq string'] = ent_df.apply(reformat_prediction, axis=1, col='freq string')
    
    if not is_bert:
        ent_df['label'] = ent_df.apply(reformat_prediction, axis=1, col='label') # for LLAMA 2 entity pred models
    
    # if not is_bert:
    #     freq_df['label'] = freq_df.apply(reformat_prediction, axis=1, col='label') # for LLAMA 2 freq identification models
    # else:
    #     freq_df['label'] = freq_df.apply(reformat_label, axis=1, col='label') # for BERT freq identification model
        
    ent_df['num_freq_gold/num_freq_pred/num_correct_freq_pred'] = ent_df.apply(agrees_with_gold_individual, pred_col='entities', label_col='label', is_freq_phrase=False, axis=1)
    num_total_freqs_gold = ent_df['num_freq_gold/num_freq_pred/num_correct_freq_pred'].apply(lambda x: x[0]).sum()
    num_total_freqs_pred = ent_df['num_freq_gold/num_freq_pred/num_correct_freq_pred'].apply(lambda x: x[1]).sum()
    num_correct_freqs_pred = ent_df['num_freq_gold/num_freq_pred/num_correct_freq_pred'].apply(lambda x: x[2]).sum()
    
    precision = num_correct_freqs_pred/(num_total_freqs_pred)
    recall = num_correct_freqs_pred/(num_total_freqs_gold)
    f1 = (2*precision*recall)/(precision+recall)
    
    return precision, recall, f1


def get_combined_precision_recall_f1(input_json_file_entity_pred, input_json_file_frequency_identify, is_entity_pred_bert, is_freq_identify_bert, output_file=None):
    ent_df = pd.read_json(path_or_buf=input_json_file_entity_pred)
    freq_df = pd.read_json(path_or_buf=input_json_file_frequency_identify)
    
    ent_df['freq string label'] = freq_df['label']
    ent_df['freq string'] = freq_df['entities']
    
    if is_entity_pred_bert:
        ent_df['label'] = ent_df.apply(reformat_label, axis=1, col='label')
    
    ent_df['entities'] = ent_df.apply(reformat_prediction, axis=1, col='entities')
    ent_df['freq string'] = ent_df.apply(reformat_prediction, axis=1, col='freq string')
    
    if not is_entity_pred_bert:
        ent_df['label'] = ent_df.apply(reformat_prediction, axis=1, col='label') # for LLAMA 2 entity pred models
    
    if not is_freq_identify_bert:
        ent_df['freq string label'] = ent_df.apply(reformat_prediction, axis=1, col='freq string label') # for LLAMA 2 freq identification models
    else:
        ent_df['freq string label'] = ent_df.apply(reformat_label, axis=1, col='freq string label') # for BERT freq identification model
    
    ent_df['num_freq_gold/num_freq_pred/num_correct_freq_pred'] = ent_df.apply(agrees_with_gold, axis=1)
    num_total_freqs_gold = ent_df['num_freq_gold/num_freq_pred/num_correct_freq_pred'].apply(lambda x: x[0]).sum()
    num_total_freqs_pred = ent_df['num_freq_gold/num_freq_pred/num_correct_freq_pred'].apply(lambda x: x[1]).sum()
    num_correct_freqs_pred = ent_df['num_freq_gold/num_freq_pred/num_correct_freq_pred'].apply(lambda x: x[2]).sum()
    
    precision = num_correct_freqs_pred/(num_total_freqs_pred)
    recall = num_correct_freqs_pred/(num_total_freqs_gold)
    f1 = (2*precision*recall)/(precision+recall)
    
    if output_file is not None:
        ent_df['num_freq_gold'] = ent_df['num_freq_gold/num_freq_pred/num_correct_freq_pred'].apply(lambda x: x[0])
        ent_df['num_freq_pred'] = ent_df['num_freq_gold/num_freq_pred/num_correct_freq_pred'].apply(lambda x: x[1])
        ent_df['num_correct_freq_pred'] = ent_df['num_freq_gold/num_freq_pred/num_correct_freq_pred'].apply(lambda x: x[2])
        ent_df.to_excel(output_file, index=False)
    
    return precision, recall, f1


def main():
    # print('Start..')
    
    input_json_file_entity_pred = 'Results/test_set_predictions/test_dataset_allData_gpt-35-turbo-freqAttr_batch4_epochs8_lrWeight5.jsonl'
    input_json_file_frequency_identify = 'Results/test_set_predictions/meta-llama_Llama-2-70b-hf_frequencyIdentification_seed379.jsonl' #'Results/test_set_predictions/meta-llama_Llama-2-70b-hf_frequency_str_extract_instructionTune_2.jsonl'
    is_entity_pred_bert = False
    is_freq_identify_bert = False
    output_file = 'Results/test_set_predictions/Combined_gptAttribute_llama70bPhrase.xlsx' # None
    precision,recall, f1 = get_combined_precision_recall_f1(input_json_file_entity_pred, input_json_file_frequency_identify, is_entity_pred_bert, is_freq_identify_bert, output_file)
    print(precision,recall,f1)
    
    #precision,recall,f1 = get_freqPhrase_precision_recall_f1('GPT/Results/results_in_llama_format/test_dataset_allData_gpt-35-turbo-freqPhrase_batch4_epochs8_lrWeight5.jsonl', False)
    #print(precision,recall,f1)
    
    #precision,recall,f1 = get_freqAttribute_precision_recall_f1('Results/test_set_predictions/bert-large-cased_freq_entity_extract.jsonl', True)
    #print(precision,recall,f1)
    
    # precision,recall,f1 = get_combined_precision_recall_f1('Results/test_set_predictions/bert-large-cased_freq_entity_extract.jsonl', 'GPT/Results/results_in_llama_format/test_dataset_allData_gpt-35-turbo-freqPhrase_batch4_epochs8_lrWeight5.jsonl', True, False)
    # print(precision,recall,f1)
    
    #precision,recall,f1 = get_combined_precision_recall_f1('Results/test_set_predictions/bert-large-cased_freq_entity_extract.jsonl', 'GPT/Results/results_in_llama_format/test_dataset_allData_gpt-35-turbo-freqPhrase_batch4_epochs8_lrWeight5.jsonl', True, False)
    #print(precision,recall,f1)
    
    # pt1 = 'Results/test_set_predictions/meta-llama_Llama-2-70b-hf_frequency_str_extract__2.jsonl'
    # pt2 = 'Results/test_set_predictions/dmis-lab_biobert-large-cased-v1.1_frequency_str_extract_2.jsonl'
    # a = pd.read_json(path_or_buf = pt2)
    # print(a)
    
    # python Extraction_with_GPT.py --job_type=responses --freq_phrase=no --dataset_json=GPT/test_dataset_freqAttribute_gpt.jsonl --deployment_name=freqAttribute_batch4_lrMlt5_n_epochs8_train370 --test_dataset_all_data_path=GPT/test_dataset_freqAttribute_allData.jsonl --output_excel_file=Results/test_dataset_allData_gpt-35-turbo-freqAttr_batch4_epochs8_lrWeight5_train370.xlsx
    
    # key = file path, value = is_bert?
    entity_pred_file_info = {'Results/test_set_predictions/distilbert-base-cased_freq_entity_extract.jsonl': True,
                            'Results/test_set_predictions/bert-large-cased_freq_entity_extract.jsonl': True,
                            'Results/test_set_predictions/dmis-lab_biobert-large-cased-v1.1_freq_entity_extract.jsonl':True,
                            'Results/test_set_predictions/emilyalsentzer_Bio_ClinicalBERT_freq_entity_extract.jsonl':True,
                            'Results/test_set_predictions/meta-llama_Llama-2-7b-hf_entityExtraction_seed379.jsonl':False,
                            'Results/test_set_predictions/meta-llama_Llama-2-70b-hf_entityExtraction_seed379.jsonl':False,
                            'Results/test_set_predictions/test_dataset_allData_gpt-35-turbo-freqAttr_batch4_epochs8_lrWeight5.jsonl':False
    }
    
    # key = file path, value = is_bert?
    freq_identify_file_info = {'Results/test_set_predictions/distilbert-base-cased_frequency_str_extract.jsonl':True,
                               'Results/test_set_predictions/bert-large-cased_frequency_str_extract.jsonl':True,
                               'Results/test_set_predictions/dmis-lab_biobert-large-cased-v1.1_frequency_str_extract_2.jsonl':True,
                               'Results/test_set_predictions/emilyalsentzer_Bio_ClinicalBERT_frequency_str_extract.jsonl':True,
                               'Results/test_set_predictions/meta-llama_Llama-2-7b-hf_frequencyIdentification_seed379.jsonl':False,
                               'Results/test_set_predictions/meta-llama_Llama-2-70b-hf_frequencyIdentification_seed379.jsonl':False,
                               'Results/test_set_predictions/test_dataset_allData_gpt-35-turbo-freqPhrase_batch4_epochs8_lrWeight5.jsonl':False      
    }
    
    # comparisons_results_file = 'Results/test_set_predictions/comparisons_04_05_2024.csv'
    # comparisons = {} # key=ent_pred_file, value=(freq_identify_file, performance)
    # for ent_pred_file, is_entity_pred_bert in entity_pred_file_info.items():
    #     for freq_identify_file, is_freq_identify_bert in freq_identify_file_info.items():
    #         print(ent_pred_file, freq_identify_file)
    #         performance = get_combined_precision_recall_f1(ent_pred_file, freq_identify_file, is_entity_pred_bert, is_freq_identify_bert)
    #         ent_pred_name = ent_pred_file.split('Results/test_set_predictions/')[1].split('.jsonl')[0]
    #         freq_identify_name = freq_identify_file.split('Results/test_set_predictions/')[1].split('.jsonl')[0]
    #         print(ent_pred_name, freq_identify_name, performance)
    #         if ent_pred_name not in comparisons:
    #             comparisons[ent_pred_name] = {}
    #         comparisons[ent_pred_name][freq_identify_name] = performance
            
    
    # print(comparisons['bert-large-cased_frequency_str_extract']['test_dataset_allData_gpt-35-turbo-freqAttr_batch4_epochs8_lrWeight5'])
    # pd.DataFrame.from_dict(comparisons, orient='index').to_csv(comparisons_results_file)
    print('End.')
    
if __name__ == "__main__":
    main()