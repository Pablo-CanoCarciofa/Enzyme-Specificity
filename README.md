# Description
This repository contains the code to create the Enzyme Specificity Predictor model (ESP). It covers the creation of a dataset of active enzyme-substrate pairs, the data augmentation used to sample inactive enzyme-substrate pairs, exploratory data analysis, ESP and other ML models used to predict enzyme specificity, and out-of-sample example predictions.

It contains 6 main files:
1. Data.ipynb - this is where the dataset of active enzyme-substrate pairs is created, and enzymes and substrates are featurised with ESM-1b and ESM-2, and the ECFP algorithm
2. EDA.ipynb - this is some exploratory data analysis with a few plots
3. Prediction.ipynb - this is where negative (inactive) enzyme-substrate pairs are created with the novel data augmentation technique, and models are trained and tested, including ESP
4. Examples.ipynb - this is where ESP is tested with some out-of-sample testing from a few salient enzyme examples from various industries
5. objects.py - this is where classes are defined for use in Prediction.ipynb, this makes the code much easier to understand and much more reproducible
6. extract.py - this is code taken from Kroll et al. (2023) used to feed amino acid sequences into ESM models to create enzyme representations
