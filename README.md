# ONSETS, ACTIVITY, AND EVENTS: A MULTI-TASK APPROACH FOR POLYPHONICSOUND EVENT MODELLING

State of the art polyphonic sound event detection (SED) systems function as frame-level multi-label classification models. In the
context of dynamic polyphony levels at each frame, sound events interfere with each other which degrade a classifier’s ability to learn the exact frequency profile of individual sound events. Frame-level localized classifiers also fail to explicitly model the long-term temporal structure of sound events. Consequently, the event-wise detection performance is less than the segment-wise detection. We define ‘temporally precise polyphonic sound event detection’ as the subtask of detecting sound event instances with the correct onset. Here, we investigate the effectiveness of sound activity detection (SAD) and onset detection as auxiliary tasks to improve temporal precision in polyphonic SED using multi-task learning. SAD helps to differentiate event activity frames from noisy and silence frames and helps to avoid missed detections at each frame. Onset predictions ensure the start of each event which in turn are used to condition predictions of both SAD and SED. Our experiments on the URBAN-SED dataset show that by conditioning SED with onset detection and SAD, there is over a three-fold relative improvement in event-based F -score.

         
# Description

         /feature_extration - this folder contains code and associated files for feature extraction
         /training - baseline models for SED, SAD, and ONSET detection, conditional models for SED
         /testing - code for model prediction
         /evaluation - codes for SED and ONSET evaluation
         /best_models - best models for conditional SED

 

# Publication

[Pankajakshan A, Bear H, Benetos E. Onsets, activity, and events: a multi-task approach for polyphonic sound event modelling 4th Workshop on Detection and Classification of Acoustic Scenes and Events (DCASE 2019), New York, USA, 25 Oct 2019 - 26 Oct 2019](http://dcase.community/documents/workshop2019/proceedings/DCASE2019Workshop_Pankajakshan_43.pdf)
                
                
                
# References

[1]  J. Salamon, D. MacConnell, M. Cartwright, P. Li, and J. P.Bello, “Scaper: A library for soundscape synthesis and aug-mentation,” in IEEE Workshop on Applications of Signal Processing  to  Audio  and  Acoustics  (WASPAA),  2017,  pp.  344–348.


[2] A. Mesaros, T. Heittola, and T. Virtanen, “Metrics for polyphonic sound event detection, ”Applied Sciences, vol. 6, no. 6,p. 162, 2016.

[3] C. Hawthorne, E. Elsen, J. Song, A. Roberts, I. Simon, C. Raffel, J. Engel, S. Oore, and D. Eck, “Onsets and frames: Dual objective  piano  transcription, ”International Conference of Music Information retrieval (ISMIR), 2018.
                
                
                
                
