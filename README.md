# Human-AI collaboration for cartridge case classification

This codebase is complementary code to the AI master thesis of Sarah de Boer. 

Cartridge case classification, as is meant in this thesis, refers to forensic investigators investigating cartridge cases and trying to predict from which make and model firearm the cartridge case is discharged from. 

The proposed human-AI collaboration system consists of two parts, the model stage and the trust score stage. In the model stage, an adaptation of the MMBT model (Kiela et al., 2019), MMBT-CC, is used to integrate user knowledge into a deep learning model that classifies images of the bottom of the cartridge cases. 
In the trust score stage, a trust score, as defined by Jiang et al. (2018), is used as a way to guide forensic investigators on whether to trust the model prediction and use it in further investigation or not to trust the model prediction and only use their own judgment of the evidence. 

In this codebase, two other repositories are used. The MMBT repository license and the Trust Score repository license can be found in the Licences folder. Copyright notices from those repositories are present in the files. Adaptations made to those files fall under the MIT license. 

MMBT repository: https://github.com/facebookresearch/mmbt 
Paper Link: https://arxiv.org/abs/1909.02950

Trust Score repository: https://github.com/google/TrustScore
Paper Link: https://arxiv.org/abs/1805.11783

### Requirements
The thesis_gpu.yml file contains the necessary packages to run the code. 

### Train procedure
First run train.py with the chosen arguments, e.g., model and save_dir. 
After that you can run create_embeddings.py to make and save the embeddings to be used in the trust score analysis. 
Then you can run get_shapley_values.py to make and save the shapley values of the embeddings for the trust score analysis. 

Then locally, you can run the Jupyter notebook, trustscore_evaluation.ipynb. You need to carefully fill in the arguments, and run the cells belonging to the chosen model (image or mmbt). 
