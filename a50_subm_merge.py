# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

from a00_common_functions import *
from scipy.stats import pearsonr


INPUT_PATH = "../input/"
OUTPUT_PATH = "../subm/"

def ensemble(subm_list):
    sl = []
    for s in subm_list:
        sl.append(pd.read_csv(s))

    t1 = []
    t2 = []
    t3 = []
    ids = []
    for s in sl:
        t1.append(s['Type_1'].values)
        t2.append(s['Type_2'].values)
        t3.append(s['Type_3'].values)
    ids = sl[0]['image_name'].values

    print('Corr class 1:', pearsonr(sl[0]['Type_1'].values, sl[1]['Type_1'].values))
    print('Corr class 2:', pearsonr(sl[0]['Type_2'].values, sl[1]['Type_2'].values))
    print('Corr class 3:', pearsonr(sl[0]['Type_3'].values, sl[1]['Type_3'].values))

    t1o = np.zeros(len(t1[0]))
    t2o = np.zeros(len(t2[0]))
    t3o = np.zeros(len(t3[0]))
    for i in range(len(subm_list)):
        t1o += np.array(t1[i])
        t2o += np.array(t2[i])
        t3o += np.array(t3[i])

    t1o /= len(subm_list)
    t2o /= len(subm_list)
    t3o /= len(subm_list)

    out = open(OUTPUT_PATH + "merge.csv", "w")
    out.write("image_name,Type_1,Type_2,Type_3\n")
    for i in range(len(t1o)):
        out.write(str(ids[i]))
        out.write(',' + str(t1o[i]))
        out.write(',' + str(t2o[i]))
        out.write(',' + str(t3o[i]))
        out.write('\n')
    out.close()

    # Check score
    print('Score on LB: {}'.format(check_score(OUTPUT_PATH + "merge.csv")))
