# Semantic Annotation of Software Requirements with Language Frame

## Dataset
Dataset for this research is [M.Dollmann et al. (2016)](http://www.aclweb.org/anthology/D16-1186)'s contribution because they manually constructed it (you can download it in [here](https://drive.google.com/open?id=1dabiJGg96PrXJX0KsLRGvJNeMILG8rRt)). We converted .ann format (original) to a single .json file for modification and converted .json to .txt file (conll-format) for training models. Actually, such conversions were very tedious because there were many exceptions (cf. jupyer notebooks in dataset folder).




## Contribution
* Text preprocessing (rule list) for specific dataset (especially for improving the performance of parsers)
* Improve the model performance of previous research model, REaCT (Dollmann et al., 2016)
* Design new feature set modeled on Language Frame
* 

## Summary
* stemming is better than lemmatization in our problem (cf. [jupyter notebook](https://github.com/gritmind/semantic-annotation/blob/master/jupyter-notebook/stemming_vs_lemmatization.ipynb))
