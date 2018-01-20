# Improving Classifier for Semantic Annotation of Software Requirements with Elaborate Syntactic Structure

It is hard for developers to clearly understand software requirements becuase of ambiguous and incomplete expressions. We solve this problem using Natural Language Processing (NLP) or Semantic Annotation (SA) with pre-defined semantic categories. We expect that after solving the problem, the readability can be improved even with ambiguities and give feedback to users when incomplete sentences. With small specific dataset, text preprocessing and feature engineering with elaborate syntactic structure were constructed for our classifier. We improve the performance of previous model in both averaged score and each category score. 



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
* Dimension reduction techniques (PCA, univariate selection) were ineffective in our problem, meaning that even with high and sparse vector, all signals were important.
* With syntactic features elaborately designed by parsers (and bag of n-grams), logistic regression as linear model also got high performances even with complex sequential problem. (Power of feature engineering!)
