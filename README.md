# Improving Classifier for Semantic Annotation of Software Requirements with Elaborate Syntactic Structure

## Dataset
Dataset for this research is [M.Dollmann et al. (2016)](http://www.aclweb.org/anthology/D16-1186)'s contribution because they manually constructed it (you can download it in [here](https://drive.google.com/open?id=1dabiJGg96PrXJX0KsLRGvJNeMILG8rRt)). We converted .ann format (original) to a single .json file for modification and converted .json to .txt file (conll-format) for training models. Actually, such conversions were very tedious because there were many exceptions (cf. jupyer notebooks in dataset folder).




## Contribution
* Text preprocessing (rule list) for specific dataset (especially for improving the performance of parsers)
* Design elaborate syntactic features with constituency and dependency parsers (Dependency parser is more sensitive to represent the type of word and clause)
* Improve the model performance of previous research model, REaCT (Dollmann et al., 2016)

## Summary
* Stemming is better than lemmatization in our problem (cf. [jupyter notebook](https://github.com/gritmind/semantic-annotation/blob/master/jupyter-notebook/stemming_vs_lemmatization.ipynb))
* Data cleaning/preprocessing, feature engineering are closely related to each other for both input and output to parsers.
* Elaborate syntactic features are based on tree sturcture (Constituency parser) and dependent type between two words (Dependency parser). We can distinguish all kinds of clauses by using dependent type (Dependency parser is more sensitive than Constituency in terms of grouping words).
* Optimization with validation set is only to find the best hyperparameter set via knowing overfitting point. Best features are not optimized with validation, but just selected (among representative sets) after calculating test error. This is similar to model selection based on test error. 
* Dependency relations are more elaborate than tree structure to represent word type (object of preposition or direct object), clause type (relative caluse or adverbial clause) (check: [De.Marneffe et al., 2008](https://nlp.stanford.edu/software/dependencies_manual.pdf))
* Dimension reduction techniques (PCA, univariate selection) were ineffective in our problem, meaning that even with high and sparse vector, all signal is important.
