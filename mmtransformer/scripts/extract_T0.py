'''
pick the first charttime which is positive or > -eps (1e-6)

'''

import os
import pandas as pd
import numpy as np
import pickle
path = 'PATH-TO-CODE/Multimodal_Transformer/data-mimic3/root/train/'
test_starttime_path = 'PATH-TO-CODE/Multimodal_Transformer/data-mimic3/root/T0/train_starttime.pkl'


# path = 'PATH-TO-CODE/Multimodal_Transformer/data-mimic3/root/test/'
# test_starttime_path = 'PATH-TO-CODE/Multimodal_Transformer/data-mimic3/root/T0/test_starttime.pkl'
episodeToStartTimeMapping = {}


def diff(time1, time2):
    # compute time2-time1
    # return difference in hours
    a = np.datetime64(time1)
    b = np.datetime64(time2)
    return (a-b).astype('timedelta64[h]').astype(int)


for findex, folder in enumerate(os.listdir(path)):
    events_path = os.path.join(path, folder, 'events.csv')
    events = pd.read_csv(events_path)

    stays_path = os.path.join(path, folder, 'stays.csv')
    stays_df = pd.read_csv(stays_path)
    hadm_ids = list(stays_df.HADM_ID.values)
    intimes = stays_df.INTIME.values

    for ind, hid in enumerate(hadm_ids):
        sliced = events[events.HADM_ID == hid]
        chart_times = sliced['CHARTTIME']
        chart_times = chart_times.sort_values()
        intime = intimes[ind]
        # remove intime from charttime
        result = -1
        # pick the first charttime which is positive or > -eps (1e-6)
        for t in chart_times:
            # compute t-intime in hours
            if diff(t, intime) > 1e-6:
                result = t
                break
        name = folder + '_' + str(ind+1)
        episodeToStartTimeMapping[name] = result

    if findex % 100 == 0:
        print("Processed %d" % (findex + 1))

with open(test_starttime_path, 'wb') as f:
    pickle.dump(episodeToStartTimeMapping, f, pickle.HIGHEST_PROTOCOL)
