import pandas as pd
import pickle
import numpy as np


class ICD9Reader():
    def __init__(self, patient2hadmid_picklepath):
        df = pd.read_csv('../mimic3/PROCEDURES_ICD.csv')
        df = df[df.HADM_ID.notnull()]
        df['HADM_ID'] = df['HADM_ID'].astype(int)

        counts = df.ICD9_CODE.value_counts()
        counts = counts[counts > 5]
        self.icd9s = {item: i for i, item in enumerate(counts.index)}
        with open(patient2hadmid_picklepath, 'rb') as f:
            patient2hadmid = pickle.load(f)

        self.patient2icd9 = {}
        for patient in patient2hadmid.keys():
            hadmid = patient2hadmid[patient]
            icd9_hadmid = list(df[df.HADM_ID == hadmid].ICD9_CODE)
            icd9_hadmid = list(filter(lambda x: x in self.icd9s, icd9_hadmid))
            self.patient2icd9[int(patient)] = icd9_hadmid

        del patient2hadmid
        print("Finished building object for ICD9Reader: ", len(self.icd9s))

    def get_ic9_onehot(self, fname):
        def _get_patient_id_from_filename(fname):
            return int(fname.split('_')[0])
        a = [0]*len(self.icd9s)
        l = self.patient2icd9[_get_patient_id_from_filename(fname)]
        for item in l:
            if item in self.icd9s:
                a[self.icd9s[item]] = 1
        return np.array(a)
