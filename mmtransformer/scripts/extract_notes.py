from scipy import stats
import os
import pandas as pd
"""
Preprocess PubMed abstracts or MIMIC-III reports
Combine all the clinical notes in a same HAMD_ID sorted by CHAETTIME. 
    The HAMD_ID comes from MERGE on 'NOTEEVENTS.csv' and 'train/test split from MIMIC-III-Benchmarks packages'.
    Named by SUBJECT_ID, if a SUBJECT_ID has multiple HAMD_ID, then list as _1, _2, ...

"""
import re
import json

from nltk import sent_tokenize, word_tokenize

SECTION_TITLES = re.compile(
    r'('
    r'ABDOMEN AND PELVIS|CLINICAL HISTORY|CLINICAL INDICATION|COMPARISON|COMPARISON STUDY DATE'
    r'|EXAM|EXAMINATION|FINDINGS|HISTORY|IMPRESSION|INDICATION'
    r'|MEDICAL CONDITION|PROCEDURE|REASON FOR EXAM|REASON FOR STUDY|REASON FOR THIS EXAMINATION'
    r'|TECHNIQUE'
    r'):|FINAL REPORT',
    re.I | re.M)


def pattern_repl(matchobj):
    """
    Return a replacement string to be used for match object
    """
    return ' '.rjust(len(matchobj.group(0)))


def find_end(text):
    """Find the end of the report."""
    ends = [len(text)]
    patterns = [
        re.compile(r'BY ELECTRONICALLY SIGNING THIS REPORT', re.I),
        re.compile(r'\n {3,}DR.', re.I),
        re.compile(r'[ ]{1,}RADLINE ', re.I),
        re.compile(r'.*electronically signed on', re.I),
        re.compile(r'M\[0KM\[0KM')
    ]
    for pattern in patterns:
        matchobj = pattern.search(text)
        if matchobj:
            ends.append(matchobj.start())
    return min(ends)


def split_heading(text):
    """Split the report into sections"""
    start = 0
    for matcher in SECTION_TITLES.finditer(text):
        # add last
        end = matcher.start()
        if end != start:
            section = text[start:end].strip()
            if section:
                yield section

        # add title
        start = end
        end = matcher.end()
        if end != start:
            section = text[start:end].strip()
            if section:
                yield section

        start = end

    # add last piece
    end = len(text)
    if start < end:
        section = text[start:end].strip()
        if section:
            yield section


def clean_text(text):
    """
    Clean text
    """

    # Replace [**Patterns**] with spaces.
    text = re.sub(r'\[\*\*.*?\*\*\]', pattern_repl, text)
    # Replace `_` with spaces.
    text = re.sub(r'_', ' ', text)

    start = 0
    end = find_end(text)
    new_text = ''
    if start > 0:
        new_text += ' ' * start
    new_text = text[start:end]

    # make sure the new text has the same length of old text.
    if len(text) - end > 0:
        new_text += ' ' * (len(text) - end)
    return new_text


def preprocess_mimic(text):
    """
    Preprocess reports in MIMIC-III.
    1. remove [**Patterns**] and signature
    2. split the report into sections
    3. tokenize sentences and words
    4. lowercase
    """
    for sec in split_heading(clean_text(text)):
        for sent in sent_tokenize(sec):
            text = ' '.join(word_tokenize(sent))
            yield text.lower()


df = pd.read_csv('./PATH-TO-ORIGINAL-MIMIC-DATA/physionet.org/mimiciii/1.4/NOTEEVENTS.csv')
df.CHARTDATE = pd.to_datetime(df.CHARTDATE)
df.CHARTTIME = pd.to_datetime(df.CHARTTIME)
df.STORETIME = pd.to_datetime(df.STORETIME)

df2 = df[df.SUBJECT_ID.notnull()]
df2 = df2[df2.HADM_ID.notnull()]
df2 = df2[df2.CHARTTIME.notnull()]
df2 = df2[df2.TEXT.notnull()]

df2 = df2[['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'TEXT']]

del df

def filter_for_first_hrs(dataframe, _days=2):
    min_time = dataframe.CHARTTIME.min()
    return dataframe[dataframe.CHARTTIME < min_time + pd.Timedelta(days=_days)]


def getText(t):
    return " ".join(list(preprocess_mimic(t)))


def getSentences(t):
    return list(preprocess_mimic(t))


print(df2.groupby('HADM_ID').count().describe())
'''
        SUBJECT_ID       CHARTTIME           TEXT
count  55926.000000  55926.000000  55926.000000
mean      28.957283     28.957283     28.957283
std       59.891679     59.891679     59.891679
min        1.000000      1.000000      1.000000
25%        5.000000      5.000000      5.000000
50%       11.000000     11.000000     11.000000
75%       27.000000     27.000000     27.000000
max     1214.000000   1214.000000   1214.000000
'''

# dataset_path = 'PATH-TO-CODE/Multimodal_Transformer/data-mimic3/root/test/'
dataset_path = 'PATH-TO-CODE/Multimodal_Transformer/data-mimic3/root/train/'
all_files = os.listdir(dataset_path)
all_folders = list(filter(lambda x: x.isdigit(), all_files))

# output_folder = 'PATH-TO-CODE/Multimodal_Transformer/data-mimic3/root/test_text_fixed/'
output_folder = 'PATH-TO-CODE/Multimodal_Transformer/data-mimic3/root/text_fixed/'

suceed = 0
failed = 0
failed_exception = 0

all_folders = all_folders

sentence_lens = []
hadm_id2index = {}

for folder in all_folders:
    try:
        patient_id = int(folder)
        sliced = df2[df2.SUBJECT_ID == patient_id]
        if sliced.shape[0] == 0:
            print("No notes for PATIENT_ID : {}".format(patient_id))
            failed += 1
            continue
        sliced.sort_values(by='CHARTTIME')

        # get the HADM_IDs from the stays.csv.
        stays_path = os.path.join(dataset_path, folder, 'stays.csv')
        stays_df = pd.read_csv(stays_path)
        hadm_ids = list(stays_df.HADM_ID.values)

        for ind, hid in enumerate(hadm_ids):
            hadm_id2index[str(hid)] = str(ind)
            '''
            #TODO: Different from the Khadanga's version!! Will replace the second HAMD_ID in a same SUBJECT_ID
            ori version:
                sliced = sliced[sliced.HADM_ID == hid]
            cur_version:
                sliced_hid = sliced[sliced.HADM_ID == hid]
            '''

            sliced_hid = sliced[sliced.HADM_ID == hid]
            #text = sliced.TEXT.str.cat(sep=' ')
            #text = "*****".join(list(preprocess_mimic(text)))
            data_json = {}
            for index, row in sliced_hid.iterrows():
                #f.write("%s\t%s\n" % (row['CHARTTIME'], getText(row['TEXT'])))
                data_json["{}".format(row['CHARTTIME'])
                          ] = getSentences(row['TEXT'])

            with open(os.path.join(output_folder, folder + '_' + str(ind+1)), 'w') as f:
                json.dump(data_json, f)

        suceed += 1
    except:
        import traceback
        traceback.print_exc()
        print("Failed with Exception FOR Patient ID: %s", folder)
        failed_exception += 1

print("Sucessfully Completed: %d/%d" % (suceed, len(all_folders)))
print("No Notes for Patients: %d/%d" % (failed, len(all_folders)))
print("Failed with Exception: %d/%d" % (failed_exception, len(all_folders)))


with open(os.path.join(output_folder, 'test_hadm_id2index'), 'w') as f:
    json.dump(hadm_id2index, f)

'''
train:
Sucessfully Completed: 27625/28728
No Notes for Patients: 1103/28728
Failed with Exception: 0/28728

test:
Sucessfully Completed: 4874/5070
No Notes for Patients: 196/5070
Failed with Exception: 0/5070

'''