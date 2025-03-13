## Data Files  

The datasets used for the experiments are available in the following files:  

- **train_set.xlsx**  
- **test_set.xlsx**  
- **validation_set.xlsx**  
- **augmented_set.xlsx**  

**Note:** `augmented_set.xlsx` includes augmented instances derived from `train_set.xlsx`, which were also utilized during training.


## File Structure  

Each dataset contains the following columns:  

- **deidentified_string** – The original text snippet with de-identified information.  
- **deidentified_frequency_phrase_annotation** – Annotated seizure frequency phrases.  
- **deidentified_frequency_attribute_annotation** – Annotated seizure frequency attributes.  
- **deidentified_structured_freqs** – The structured seizure frequencies. Note that if a frequency phrase annotation does not exists, then a value of "None" will be included here.  



## Abbreviations  

In these files, **abbreviations** were used for tagging seizure frequency phrases and attributes. Below is a reference list mapping these abbreviations to their full names.  

| Abbreviation | Full Name              |
|-------------|------------------------|
| FREQ        | Frequency              |
| EVNT        | Event                  |
| UNT         | Temporal unit          |
| PERD        | Periodic               |
| AGE         | Age                    |
| QNT         | Quantity               |
| MINQNT      | Minimum quantity       |
| MAXQNT      | Maximum quantity       |
| INTST       | Interval start         |
| INTED       | Interval end           |
| TIME        | Time                   |
| RELT        | Relative time          |
| RELTPR      | Relative time period   |
| DUR         | Duration               |
| MINDUR      | Minimum duration       |
| MAXDUR      | Maximum duration       |
| AGEST       | Age start              |		
| AGEED       | Age end                | 