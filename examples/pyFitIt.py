# this file is needed for convenient import of functions from library before pyFitIt deploy

import sys
sys.path.insert(0, '../lib')

import experiment
LoadExperiment = experiment.LoadExperiment

import sampling
sample = sampling.sample