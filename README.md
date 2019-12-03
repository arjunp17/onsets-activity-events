# ONSETS, ACTIVITY, AND EVENTS: A MULTI-TASK APPROACH FOR POLYPHONICSOUND EVENT MODELLING

State of the art polyphonic sound event detection (SED) systems function as frame-level multi-label classification models. In the
context of dynamic polyphony levels at each frame, sound events interfere with each other which degrade a classifier’s ability to learn the exact frequency profile of individual sound events. Frame-level localized classifiers also fail to explicitly model the long-term temporal structure of sound events. Consequently, the event-wise detection performance is less than the segment-wise detection. We define ‘temporally precise polyphonic sound event detection’ as the subtask of detecting sound event instances with the correct onset. Here, we investigate the effectiveness of sound activity detection (SAD) and onset detection as auxiliary tasks to improve temporal precision in polyphonic SED using multi-task learning. SAD helps to differentiate event activity frames from noisy and silence frames and helps to avoid missed detections at each frame. Onset predictions ensure the start of each event which in turn are used to condition predictions of both SAD and SED. Our experiments on the URBAN-SED dataset show that by conditioning SED with onset detection and SAD, there is over a three-fold relative improvement in event-based F -score.

         
# Description

1. /feature_extration - this folder contains code and associated files for feature extraction
2. /annotations - this folder contains ground truth anotations for SED, SAD, and ONSET
3. /training - baseline models for SED, SAD, and ONSET detection, conditional models for SED
4. /testing - code for model prediction
5. /evaluation - codes for SED and ONSET evaluation
6. /best_models - best models for conditional SED

 

                
                
                
                
                
                
