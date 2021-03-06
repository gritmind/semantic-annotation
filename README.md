# Semantic Annotation of Software Requirement

It is hard for developers to clearly understand software requirements becuase of ambiguous and incomplete expressions. To solve this problem, we propose an automatic classifier for semantic annotation with pre-defined semantic categories. We expect that after obtaining the output of the classifier, the readability can be improved even with ambiguities and feedback is given to users when incomplete sentences. With small specific dataset, text preprocessing and feature engineering with elaborate syntactic structure by using parsers, were constructed for our classifier. We improve the performance of previous model in both averaged score and each category score. 

![](/assets/model.PNG)

The paper for this project has been published in its final form in the "International Journal of Advanced Science and Technology" with ISSN 2005-4238.
[Improving Classifier for Semantic Annotation of Software Requirements with Elaborate Syntactic Structure](https://1drv.ms/b/s!AllPqyV9kKUrwx7uQT8j9DsEtAqY), Yeongsu Kim, Seungwoo Lee, Markus Dollmann and Michaela Geierhos (pp. 122-136)

Presentation for this research is in [here](https://1drv.ms/p/s!AllPqyV9kKUrsin5YbvQ5cs0GBPY).


## Prerequisites
* nltk 3.2.5
* pandas 0.20.3
* scikit-learn 0.19.1
* keras 2.0.9
* tensorflow 1.1.0
* spacy 0.101.0
* stanford parser 3.7.0


## Dataset
Dataset for this research is [M.Dollmann et al. (2016)](http://www.aclweb.org/anthology/D16-1186)'s contribution because they manually constructed it (you can download it in [here](https://drive.google.com/open?id=1dabiJGg96PrXJX0KsLRGvJNeMILG8rRt)). We converted .ann format (original) to a single .json file for modification and converted .json to .txt file (conll-format) for training models. Actually, such conversions were very tedious because there were many exceptions (cf. jupyer notebooks in dataset folder).

## Usage

0. Hyperparameter Setting: fill hyperparameters, path info, etc. in `parameters.ini`

1. Data Preparation

```
python 1_data_preparation.py --stemming
```
 * arg set: choose only one among {--stemming, --lemmatization}

![](/assets/img/img1.PNG)

2. Feature Extraction

```
python 2_feature_extraction.py --use_1gram --use_2gram --use_stanford_pos --use_stanford_parser --use_rule --use_spacy_pos --use_spacy_chunk --use_spacy_parser
``` 
 * arg set: choose at least one among {--use_1gram, --use_2gram, --use_3gram, --use_5gram, --use_classes_1gram, --use_stanford_pos, --use_stanford_parser, --use_rule, --use_spacy_pos, --use_spacy_chunk, --use_spacy_parser}

![](/assets/img/img2.PNG)

3. Train and Evaluate Model

```
python 3_train_and_test.py --use_fa --use_fb --use_fc --use_fd --model_svm
``` 
 * arg set1: choose at least one among {--use_1gram, --use_2gram, --use_3gram, --use_5gram }
 * arg set2: choose only one among {--model_lr, --model_pa, --model_nb, --model_knn, --model_dt, --model_svm, --model_et, --model_rf, --model_vc, --model_fnn, --model_cnn, --model_rnn}

![](/assets/img/img3.PNG)


## 주요내용
* 애매한 사용자의 소프트웨어 요구사항의 이해를 위한 의미 주석 실시
  - 사용자가 작성하는 소프트웨어 요구사항은 중의성과 불명확성이 존재
  - 사전에 정의된 의미 카테고리와 기계학습 모델을 사용한 의미 주석
* 기존 모델 대비 성능 향상을 위해 구문 분석기를 활용한 정교한 구문적 자질 설계
  - 구문 분석기 결과값의 신뢰성을 높이기 위한 데이터 전처리 실시
  - (구-구조 파서) 트리 구조와 구 정보를 활용한 위치 자질 설계
  - (의존 파서) 의존도 및 다양한 의존 타입들을 활용한 다양한 크기 및 특징의 그룹 자질 설계
  - 마치 영어 언어학자가 문장을 분석한 결과를 컴퓨터가 이해할 수 있도록 유도 
* 추가로 일반적인 전처리, 적합한 모델 선택, bag of n-그램과 같은 통계 기반 자질 설계


## Contribution
* Text preprocessing (rule list) for specific dataset (especially for improving the performance of parsers)
* Design elaborate syntactic features with constituency and dependency parsers (Dependency parser is more sensitive to represent the type of word and clause)
* Improve the model performance of previous research model, REaCT (Dollmann et al., 2016)

## Summary
* Stemming is better than lemmatization in our problem (cf. [jupyter notebook](https://github.com/gritmind/semantic-annotation/blob/master/jupyter-notebook/stemming_vs_lemmatization.ipynb))
* Data cleaning/preprocessing, feature engineering are closely related to each other for both input and output to parsers.
* Elaborate syntactic features are based on tree sturcture (Constituency parser) and dependent type between two words (Dependency parser). We can distinguish all kinds of clauses by using dependent type (Dependency parser is more sensitive than Constituency in terms of grouping words (e.g. *SBAR vs. advcl, acl, relcl, ..*))
* Dependency relations are more elaborate than tree structure to represent word type (object of preposition or direct object), clause type (relative caluse or adverbial clause) (cf. [De.Marneffe et al., 2008](https://nlp.stanford.edu/software/dependencies_manual.pdf))
* Optimization with validation set is only to find the best hyperparameter set via knowing overfitting point. Best features are not optimized with validation, but just selected (among representative sets) after calculating test error. This is similar to model selection based on test error. 
* Dimension reduction techniques (PCA, univariate selection) were ineffective in our problem, meaning that even with high and sparse vector, most signals are considered as important (or maybe PCA is inherently not suitable to our dataset).
* With syntactic features elaborately designed by parsers (and bag of n-grams), logistic regression as linear model also got high performances even with complex sequential problem. (Power of feature engineering!). This is related to (Joulin et al 2016) and (Wang and Manning 2012) that mentioned if the right features are used, linear classifiers often obtain state-of-the-art performances.
* SVM and Voting classfier were the best model in Micro-averaged F1 and Macro-averaged F1, respectively.
* Non-linear models are more sensitive to hyperparameter setting than linear models. (even random seed to split dataset)

## Acknowledgement
Korea Institute of Science and Technology Information (KISTI) <br>
University of Science and Technology (UST), Korea <br>
University of Paderborn, Germany <br>
2016.12 ~ 2017.05
