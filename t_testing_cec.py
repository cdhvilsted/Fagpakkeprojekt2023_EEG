
import scipy
from data_load_cec import *
import mne
import numpy as np

# using scipy: t-test used compares the means of two related samples of scores
# N1 component
times = np.arange(-0.1,1,step=1/128)
N_comp_min = (np.where(times <= 0.05))[-1][-1] # 50 ms

N_comp_max = (np.where(times >= 0.15))[0][0] # 150 ms
print('N indexes: ' + str(N_comp_min) + ',' + str(N_comp_max))

# P2 component
P_comp_min = (np.where(times <= 0.15))[-1][-1] # 150 ms
P_comp_max = (np.where(times >= 0.25))[0][0] # 250 ms
print('P indexes: ' + str(P_comp_min) + ',' + str(P_comp_max))

# amplitude samples for same stimuli
N1_As = []
N1_ics = []
N1_cs = []

P2_As = []
P2_ics = []
P2_cs = []

for i in range(len(data_As_ind)):
    N1_As.append(np.min(data_As_ind[i][0][N_comp_min:N_comp_max]))
    N1_ics.append(np.min(data_AVics_ind[i][0][N_comp_min:N_comp_max]))
    N1_cs.append(np.min(data_AVcs_ind[i][0][N_comp_min:N_comp_max]))
    
    P2_As.append(np.max(data_As_ind[i][0][P_comp_min:P_comp_max]))
    P2_ics.append(np.max(data_AVics_ind[i][0][P_comp_min:P_comp_max]))
    P2_cs.append(np.max(data_AVcs_ind[i][0][P_comp_min:P_comp_max]))


diff_P_A_ic = [abs((P2_As[i] - P2_ics[i])*(10**6)) for i in range(len(N1_As))]
total_diff_Paic = np.mean(diff_P_A_ic)

diff_P_A_c = [abs((P2_As[i] - P2_cs[i])*(10**6)) for i in range(len(N1_As))]
total_diff_Pac = np.mean(diff_P_A_c)

print('Difference in A and incongruent speech: ' + str(total_diff_Paic))
print('Difference in A and congruent speech: ' + str(total_diff_Pac))

test_t = scipy.stats.ttest_1samp(diff_P_A_c, popmean=0) # auditiv + congruent
print('Difference in A and congruent speech t test: ' + str(test_t))

test_t2 = scipy.stats.ttest_1samp(diff_P_A_ic, popmean=0) # auditiv + congruent
print('Difference in A and incongruent speech t test: ' + str(test_t2))

N1_Ans = []
N1_icns = []
N1_cns = []

P2_Ans = []
P2_icns = []
P2_cns = []

for i in range(len(data_Ans_ind)):
    N1_Ans.append(np.min(data_Ans_ind[i][0][N_comp_min:N_comp_max]))
    N1_icns.append(np.min(data_AVicns_ind[i][0][N_comp_min:N_comp_max]))
    N1_cns.append(np.min(data_AVcns_ind[i][0][N_comp_min:N_comp_max]))
    
    P2_Ans.append(np.max(data_Ans_ind[i][0][P_comp_min:P_comp_max]))
    P2_icns.append(np.max(data_AVicns_ind[i][0][P_comp_min:P_comp_max]))
    P2_cns.append(np.max(data_AVcns_ind[i][0][P_comp_min:P_comp_max]))

#print(len(data_As_ind))
#print(data_As_ind[1][0][N_comp_min:N_comp_max])
print(len((data_As_ind[0][0])))

# N1
# speech
test_As_in = scipy.stats.ttest_rel(N1_As, N1_ics) # auditiv + incongruent
test_As_con = scipy.stats.ttest_rel(N1_As, N1_cs) # auditiv + congruent
print('T-test for A and incongruent AV in speech mode gives (N): ' + str(test_As_in))
print('T-test for A and congruent AV in speech mode gives (N): ' + str(test_As_con))

# non-speech
test_Ans_in = scipy.stats.ttest_rel(N1_Ans, N1_icns) # auditiv + incongruent
test_Ans_con = scipy.stats.ttest_rel(N1_Ans, N1_cns) # auditiv + congruent
print('T-test for A and incongruent AV in non-speech mode gives (N): ' + str(test_Ans_in))
print('T-test for A and congruent AV in non-speech mode gives (N): ' + str(test_Ans_con))



# P2
# speech
As_in = scipy.stats.ttest_rel(P2_As, P2_ics) # auditiv + incongruent
As_con = scipy.stats.ttest_rel(P2_As, P2_cs) # auditiv + congruent
print('T-test for A and incongruent AV in speech mode gives (P): ' + str(As_in))
print('T-test for A and congruent AV in speech mode gives (P): ' + str(As_con))

# non-speech
Ans_in = scipy.stats.ttest_rel(P2_Ans, P2_icns) # auditiv + incongruent
Ans_con = scipy.stats.ttest_rel(P2_Ans, P2_cns) # auditiv + congruent
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
