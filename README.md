# ONSETS, ACTIVITY, AND EVENTS: A MULTI-TASK APPROACH FOR POLYPHONICSOUND EVENT MODELLING

State of the art polyphonic sound event detection (SED) systems function as frame-level multi-label classification models. In the
context of dynamic polyphony levels at each frame, sound events interfere with each other which degrade a classifier’s ability to learn the exact frequency profile of individual sound events. Frame-level localized classifiers also fail to explicitly model the long-term temporal structure of sound events. Consequently, the event-wise detection performance is less than the segment-wise detection. We define ‘temporally precise polyphonic sound event detection’ as the subtask of detecting sound event instances with the correct onset. Here, we investigate the effectiveness of sound activity detection (SAD) and onset detection as auxiliary tasks to improve temporal precision in polyphonic SED using multi-task learning. SAD helps to differentiate event activity frames from noisy and silence frames and helps to avoid missed detections at each frame. Onset predictions ensure the start of each event which in turn are used to condition predictions of both SAD and SED. Our experiments on the URBAN-SED dataset show that by conditioning SED with onset detection and SAD, there is over a three-fold relative improvement in event-based F -score.

         
      Here, I have added some extra materials related to our paper "ONSETS, ACTIVITY, AND EVENTS: A MULTI-TASK APPROACH FOR                      
      
      POLYPHONIC SOUND EVENT MODELLING", which is now under review for WASPAA 2019 conference. The class-wise metrics for different    
      
      models described in the paper are attached here.

 

                
                
                
                
                
                
