# DeepNet
## Data
1. gnn_sim.R is used for creating simulated data sets.
2. preprocess.R and variable selection. R are used to re-organize the original data sets and get the data that we need in our project.
3. variant visits.R is used to select out the patients we want, and build data sets can be directly used in the model (e.g build graph for each sequence).

## Model
### Python codes in the three folders cnn_gnn, gnn_sim and three_outcomes has the same file names. cnn_gnn is for real data analysis (binary outcomes), gnn_sim is for simulation, three_outcomes is for real data analysis (three outcomes), whole_image is for real data analysis (adas score as outcome).
1. Transformer.py is used for building transformer encoder.
2. ffn.py is used for building feed-forward networks.
3. gnn.py is used for building graph neural networks.
4. rnn.py is used for building recurrent neural networks.
5. get_data.py is used for building python data sets (e.g dictionaries and pandas) using the exisiting data sets.
6. featues extraction.py is used for building CNN on original data (PET images) and dimension reduction.

## Registration
### The raw PET images directly downloaded from ADNI website are registered to a template. This is done in a Ubuntu 2.0 Window subsystem.
1. reg.R is used to register PET images to an AAL template and extract regions of interest.
