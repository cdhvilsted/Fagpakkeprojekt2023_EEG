
import scipy
from data_load_cec import data_As, data_Ans, data_AVcns, data_AVcs, data_AVicns, data_AVics, data_Vns, data_Vs, all_datans, all_datas
import mne
import numpy as np

# using scipy: t-test used compares the means of two independent samples of scores
# N component
times = np.arange(-0.1,1,step=1/128)
# speech
test_As_in = scipy.stats.ttest_ind(data_As, data_AVics) # auditiv + incongruent
test_As_con = scipy.stats.ttest_ind(data_As, data_AVcs) # auditiv + congruent
#print('T-test for A and incongruent AV in speech mode gives: ' + str(test_As_in))
print('T-test for A and congruent AV in speech mode gives: ' + str(test_As_con))

# non-speech
test_Ans_in = scipy.stats.ttest_ind(data_Ans, data_AVicns) # auditiv + incongruent
test_Ans_con = scipy.stats.ttest_ind(data_Ans, data_AVcns) # auditiv + congruent
#print('T-test for A and incongruent AV in non-speech mode gives: ' + str(test_Ans_in))
#print('T-test for A and congruent AV in non-speech mode gives: ' + str(test_Ans_con))

# using mne
#n_permutations = 50000
#T0, p_values, H0 = mne.stats.permutation_t_test(np.ndarray(data_As, all_datas), n_permutations, n_jobs=None)
#print(p_values)

#significant_sensors = picks[p_values <= 0.05]
#significant_sensors_names = [raw.ch_names[k] for k in significant_sensors]

#print("Number of significant sensors : %d" % len(significant_sensors))
#print("Sensors names : %s" % significant_sensors_names)
# speech
#speech_ttest = mne.stats.permutation_t_test(np.ndarray(all_datas, 4), n_permutations=10000, n_jobs=None)
#print(speech_ttest)