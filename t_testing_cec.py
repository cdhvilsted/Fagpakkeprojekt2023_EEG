
import scipy
from data_load_cec import data_As, data_Ans, data_AVcns, data_AVcs, data_AVicns, data_AVics, data_Vns, data_Vs, all_datans, all_datas
import mne
import numpy as np

# using scipy: t-test used compares the means of two related samples of scores
# N component
times = np.arange(-0.1,1,step=1/128)
N_comp_min = (np.where(times <= 0.05))[-1][-1]
N_comp_max = (np.where(times >= 0.15))[0][0]
#print(times)
print(N_comp_min, N_comp_max)
print(len(data_As))

# speech
test_As_in = scipy.stats.ttest_rel(data_As[N_comp_min:N_comp_max], data_AVics[N_comp_min:N_comp_max]) # auditiv + incongruent
test_As_con = scipy.stats.ttest_rel(data_As[N_comp_min:N_comp_max], data_AVcs[N_comp_min:N_comp_max]) # auditiv + congruent
print('T-test for A and incongruent AV in speech mode gives (N): ' + str(test_As_in))
print('T-test for A and congruent AV in speech mode gives (N): ' + str(test_As_con))

# non-speech
test_Ans_in = scipy.stats.ttest_rel(data_Ans[N_comp_min:N_comp_max], data_AVicns[N_comp_min:N_comp_max]) # auditiv + incongruent
test_Ans_con = scipy.stats.ttest_rel(data_Ans[N_comp_min:N_comp_max], data_AVcns[N_comp_min:N_comp_max]) # auditiv + congruent
print('T-test for A and incongruent AV in non-speech mode gives (N): ' + str(test_Ans_in))
print('T-test for A and congruent AV in non-speech mode gives (N): ' + str(test_Ans_con))


# P2 component
P_comp_min = (np.where(times <= 0.15))[-1][-1]
P_comp_max = (np.where(times >= 0.25))[0][0]

# speech
As_in = scipy.stats.ttest_rel(data_As[P_comp_min:P_comp_max], data_AVics[P_comp_min:P_comp_max]) # auditiv + incongruent
As_con = scipy.stats.ttest_rel(data_As[P_comp_min:P_comp_max], data_AVcs[P_comp_min:P_comp_max]) # auditiv + congruent
print('T-test for A and incongruent AV in speech mode gives (P): ' + str(As_in))
print('T-test for A and congruent AV in speech mode gives (P): ' + str(As_con))

# non-speech
Ans_in = scipy.stats.ttest_rel(data_Ans[P_comp_min:P_comp_max], data_AVicns[P_comp_min:P_comp_max]) # auditiv + incongruent
Ans_con = scipy.stats.ttest_rel(data_Ans[P_comp_min:P_comp_max], data_AVcns[P_comp_min:P_comp_max]) # auditiv + congruent
print('T-test for A and incongruent AV in non-speech mode gives (P): ' + str(Ans_in))
print('T-test for A and congruent AV in non-speech mode gives (P): ' + str(Ans_con))

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