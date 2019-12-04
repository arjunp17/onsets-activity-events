# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 16:53:41 2019

@author: arjun
"""

import numpy as np

# reference onset (binary)
reference_onset_frame = np.load('../test_label_ONSET.npy')
# predicted onset prob
estimated_onset_prob = np.load('../pred_label_ONSET.npy')

def prob_to_onehot(model_prediction, threshold):
   onehot_pred =[]
   for i in range(len(model_prediction)):
      present_sample = model_prediction[i]
      present_sample[present_sample >= threshold] = 1
      present_sample[present_sample < threshold] = 0
      onehot_pred.append(present_sample)
   return onehot_pred


estimated_onset_frame = prob_to_onehot(estimated_onset_prob, 0.2) # threshold for onset detection is 0.2
estimated_onset_frame = np.array(estimated_onset_frame) 


def onset_metrics(reference_onset_frame,estimated_onset_frame,fps,t_collar):
    ntp = []
    nfp = []
    nfn = []
    true_onset = []
    for i in range(len(reference_onset_frame)):
        present_reference = reference_onset_frame[i]
        present_prediction = estimated_onset_frame[i]
        for j in range(len(present_reference)):
            if (present_reference[j] == 1):
                true_onset.append(1)
                
            if (present_reference[j] == 1 and (np.amax(present_prediction[max(0,((j-1)-int((float(t_collar)/2)*fps))):(j+1)+int((float(t_collar)/2)*fps)]) == 1)):
                ntp.append(1)
            else:
                if (present_reference[j] == 1 and (np.amax(present_prediction[max(0,((j-1)-int((float(t_collar)/2)*fps))):(j+1)+int((float(t_collar)/2)*fps)]) == 0)):
                    nfn.append(1)
                else:
                    if (present_prediction[j] == 1 and (np.amax(present_reference[max(0,((j-1)-int((float(t_collar)/2)*fps))):(j+1)+int((float(t_collar)/2)*fps)]) == 0)):
                        nfp.append(1)
                
                        
    ntp = np.array(ntp)
    nfp = np.array(nfp)
    nfn = np.array(nfn)
    true_onset = np.array(true_onset)
    print 'total no. of reference onset:', len(true_onset)
    print 'no. of TPs:',len(ntp)
    print 'no. of FPs:',len(nfp)
    print 'no. of FNs:',len(nfn)
    P = float(len(ntp))/float(len(ntp)+len(nfp))
    R = float(len(ntp))/float(len(ntp)+len(nfn))
    F = float(2*P*R)/float(P+R)
    return P, R, F
                      



#fps - frames per second = 50
#t_collar - 250ms
precision, recall, f_score = onset_metrics(reference_onset_frame,estimated_onset_frame,50,0.5)
print 'precision:', precision
print 'recall:', recall
print 'f_score:', f_score
##########################################################################################


