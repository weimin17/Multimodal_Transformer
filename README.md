# Multimodal Transformer
The repository for the paper "A Multimodal Transformer: Fusing Clinical Notes With Structured EHR Data for Interpretable In-Hospital Mortality Prediction" submitted to AMIA'22 Annual Symposium.

# Setup
The codes are tested on CUDA 11.4 with 24GB RAM GPU. For environment setup, please follow the install instruction in Section 'Clinical Data Processing'. 

# Clinical Data Processing
## Structured Clinical Variables Processing
Clone https://github.com/YerevaNN/mimic3-benchmarks (Harutyunyan et al.) to 'Multimodal_Transformer/mimic3-benchmarks' folder. Setup the environment, and run all data generation steps to generate training data without text features.

create folder 'data-mimic3' under 'Multimodal_Transformer' folder, and all the MIMIC-III processed data will be stored in 'data-mimi3' folder.

## Unstructured Clinical Notes Processing
Clinical Notes processing is based on repository in https://github.com/kaggarwal/ClinicalNotesICU. 

### Requirenments
setup the environment for notes processing and model training. Install environment:

~~~~
pip install -r requrements.txt
~~~~

Update all paths and configuration in 'mmtransformer/config.py'. 


### Notes Processing

+ Run 'mmtransformer/scripts/extract_notes.py', the folder 'data-mimic3/root/test_text_fixed/', and 'data-mimic3/root/text_fixed/' will be generated.
+ Run 'mmtransformer/scripts/extract_T0.py' file.

# Train and Test

For our well-trained model, you can download from [GoogleDrive](https://drive.google.com/file/d/1Wch0pEgQ8PeWE9p77B6rdNuo9l28CZNv/view?usp=sharing). Unzip the file and put them in './Multimodal_Transformer/mmtransformer/models/Checkpoints' and './Multimodal_Transformer/mmtransformer/models/Data' accordingly. Or you can generate the files yourself.

## Test

For model with only clinical notes (mbert), run

~~~~
python mbert.py --gpu_id 1
~~~~

For multimodal transformer, run

~~~~
python IHM_mmtransformer.py --mode test --model_type both --model_name BioBert --TSModel Transformer --checkpoint_path Multimodal_Transformer --MaxLen 512 --NumOfNotes 0 --TextModelCheckpoint BioClinicalBERT_FT --freeze_model 1 --number_epoch 5 --batch_size 5 --load_model 1 --gpu_id 1
~~~~

## Train

For multimodal transformer training, run

~~~~
python IHM_mmtransformer.py --mode train --model_type both --model_name BioBert --TSModel Transformer --checkpoint_path Multimodal_Transformer --MaxLen 512 --NumOfNotes 0 --TextModelCheckpoint BioClinicalBERT_FT --freeze_model 1 --number_epoch 5 --batch_size 5 --load_model 0 --gpu_id 1
~~~~


# Visualization
The output of all analysis are in 'Analysis' folder. For important clinical words analysis and visualization in clinical notes, 

1. Run 'notes_analysis.py' to get the IG value with associated words, stored in file 'Analysis/bert_analysis_pred_all2.pkl'

2. Run 'notes_analysis3.py' to get the word list with frequency, stored in 'pred_tokenlist_top10_l0_2.txt'. We further filtered the list to remove the irrelavent words and tokens, which is stored in 'filter_pred_tokenlist_top10_l0_2.txt'.

It will also generate the word cloud 'filter_pred_tokenlist_top10_l0_2.png'.


# Credits
The code is based on repository by Khadanga et al. given in https://github.com/kaggarwal/ClinicalNotesICU, and by Deznabi et al. given in https://github.com/Information-Fusion-Lab-Umass/ClinicalNotes_TimeSeries for experimental setup.


The MIMIC-III clinical variables pre-processing is clone from repository by Harutyunyan et al. given in https://github.com/YerevaNN/mimic3-benchmarks
