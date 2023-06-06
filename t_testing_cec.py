
import scipy
#from data_load_cec import *
from ICA_dataImport import common, data_Vs, data_As, data_AVcs, data_AVics, data_AVc, data_A, data_V, data_AVic
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
N1_Vs = []

P2_As = []
P2_ics = []
P2_cs = []
P2_Vs = []

N1_Ans = []
N1_icns = []
N1_cns = []
N1_Vns = []

P2_Ans = []
P2_icns = []
P2_cns = []
P2_Vns = []

print(data_A.shape)
for i in range(14):
    t_data_A = data_A[i,12,:].reshape(97,141)
    t_data_V = data_V[i,12,:].reshape(97,141)
    t_data_AVc = data_AVc[i,12,:].reshape(97,141)
    t_data_AVic = data_AVic[i,12,:].reshape(97,141)
    t_data_As = data_As[i,12,:].reshape(97,141)
    t_data_Vs = data_Vs[i,12,:].reshape(97,141)
    t_data_AVcs = data_AVcs[i,12,:].reshape(97,141)
    t_data_AVics = data_AVics[i,12,:].reshape(97,141)
    for j in range(97):
        #non-speech 
        N1_Ans.append(np.min(t_data_A[j][N_comp_min:N_comp_max]))
        N1_Vns.append(np.min(t_data_V[j][N_comp_min:N_comp_max]))
        N1_cns.append(np.min(t_data_AVc[j][N_comp_min:N_comp_max]))
        N1_icns.append(np.min(t_data_AVic[j][N_comp_min:N_comp_max]))
        P2_Ans.append(np.max(t_data_A[j][P_comp_min:P_comp_max]))
        P2_Vns.append(np.max(t_data_V[j][P_comp_min:P_comp_max]))
        P2_cns.append(np.max(t_data_AVc[j][P_comp_min:P_comp_max]))
        P2_icns.append(np.max(t_data_AVic[j][P_comp_min:P_comp_max]))

        #speech
        N1_As.append(np.min(t_data_As[j][N_comp_min:N_comp_max]))
        N1_Vs.append(np.min(t_data_Vs[j][N_comp_min:N_comp_max]))
        N1_cs.append(np.min(t_data_AVcs[j][N_comp_min:N_comp_max]))
        N1_ics.append(np.min(t_data_AVics[j][N_comp_min:N_comp_max]))
        P2_As.append(np.max(t_data_As[j][P_comp_min:P_comp_max]))
        P2_Vs.append(np.max(t_data_Vs[j][P_comp_min:P_comp_max]))
        P2_cs.append(np.max(t_data_AVcs[j][P_comp_min:P_comp_max]))
        P2_ics.append(np.max(t_data_AVics[j][P_comp_min:P_comp_max]))

"""
for i in range(len(data_As_ind)):
    N1_As.append(np.min(data_As_ind[i][12][N_comp_min:N_comp_max]))
    N1_ics.append(np.min(data_AVics_ind[i][12][N_comp_min:N_comp_max]))
    N1_cs.append(np.min(data_AVcs_ind[i][12][N_comp_min:N_comp_max]))
    
    P2_As.append(np.max(data_As_ind[i][12][P_comp_min:P_comp_max]))
    P2_ics.append(np.max(data_AVics_ind[i][12][P_comp_min:P_comp_max]))
    P2_cs.append(np.max(data_AVcs_ind[i][12][P_comp_min:P_comp_max]))

"""
#subtracting the visual from audiovisual
diff_P_A_ic = [abs((P2_As[i] - (P2_ics[i]-P2_Vs))*(10**6)) for i in range(len(N1_As))]
total_diff_Paic = np.mean(diff_P_A_ic)

diff_P_A_c = [abs((P2_As[i] - (P2_cs[i]-P2_Vs))*(10**6)) for i in range(len(N1_As))]
total_diff_Pac = np.mean(diff_P_A_c)

print('Difference in A and incongruent speech: ' + str(total_diff_Paic))
print('Difference in A and congruent speech: ' + str(total_diff_Pac))

test_t = scipy.stats.ttest_1samp(diff_P_A_c, popmean=0) # auditiv + congruent
print('Difference in A and congruent speech t test: ' + str(test_t))

test_t2 = scipy.stats.ttest_1samp(diff_P_A_ic, popmean=0) # auditiv + congruent
print('Difference in A and incongruent speech t test: ' + str(test_t2))

"""
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

"""
# N1
# speech
print('length of N1 and P2 lists:')
print(len(N1_As), len(N1_ics), len(N1_cs),len(N1_Vs))
print(len(N1_Ans), len(N1_icns), len(N1_cns), len(N1_Vns))
print(len(P2_As), len(P2_ics), len(P2_cs), len(P2_Vs))
print(len(P2_Ans), len(P2_icns), len(P2_cns), len(P2_Vns))

N1_ics_nv = [N1_ics[i]-N1_Vs[i] for i in range(len(N1_ics))]
N1_cs_nv = [N1_cs[i]-N1_Vs[i] for i in range(len(N1_cs))]
test_As_in = scipy.stats.ttest_rel(N1_As, N1_ics_nv) # auditiv + incongruent
test_As_con = scipy.stats.ttest_rel(N1_As, N1_cs_nv) # auditiv + congruent
print('T-test for A and incongruent AV in speech mode gives (N): ' + str(test_As_in))
print('T-test for A and congruent AV in speech mode gives (N): ' + str(test_As_con))

# non-speech
N1_icns_nv = [N1_icns[i]-N1_Vns[i] for i in range(len(N1_icns))]
N1_cns_nv = [N1_cns[i]-N1_Vns[i] for i in range(len(N1_cns))]
test_Ans_in = scipy.stats.ttest_rel(N1_Ans, N1_icns_nv ) # auditiv + incongruent
test_Ans_con = scipy.stats.ttest_rel(N1_Ans, N1_cns_nv) # auditiv + congruent
print('T-test for A and incongruent AV in non-speech mode gives (N): ' + str(test_Ans_in))
print('T-test for A and congruent AV in non-speech mode gives (N): ' + str(test_Ans_con))



# P2
# speech
P2_ics_nv = [P2_ics[i]-P2_Vs[i] for i in range(len(P2_ics))]
P2_cs_nv = [P2_cs[i]-P2_Vs[i] for i in range(len(P2_cs))]
As_in = scipy.stats.ttest_rel(P2_As, P2_ics_nv) # auditiv + incongruent
As_con = scipy.stats.ttest_rel(P2_As, P2_cs_nv) # auditiv + congruent
print('T-test for A and incongruent AV in speech mode gives (P): ' + str(As_in))
print('T-test for A and congruent AV in speech mode gives (P): ' + str(As_con))

# non-speech
P2_icns_nv = [P2_icns[i]-P2_Vns[i] for i in range(len(P2_icns))]
P2_cns_nv = [P2_cns[i]-P2_Vns[i] for i in range(len(P2_cns))]
Ans_in = scipy.stats.ttest_rel(P2_Ans, P2_icns_nv) # auditiv + incongruent
Ans_con = scipy.stats.ttest_rel(P2_Ans, P2_cns_nv) # auditiv + congruent
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
