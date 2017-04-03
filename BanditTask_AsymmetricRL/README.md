# Code for a 2-arm bandit task, using asymmetric and symmetric (RW) RL models.

Three models are included in this folder: 
1) a Rescorla-Wagner (symmetric) model which assumes that learners weight positive and negative events using the same learning rate parameter
2) an Asymmetric Prediction Error model which assumes that learners have two distinct learning rates depending on the valence of the prediction error of each trial. This allows for trials with outcomes that were better than or worse than expected to be weighted differently.
3) an Asymmetric Outcome model which assumes that learners have two distinct learning rates depending on the valence of the outcome, itself. Thsi allows for trials with gains and losses to be weighted differently. 
