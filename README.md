# Bangla_Grammatical_Error_Detection

[Return_Zero_Bangla_GED_IEEE_Conf_Paper.pdf](https://github.com/ridwanultanvir/Bangla_Grammatical_Error_Detection/blob/main/Return_Zero_Bangla_GED_IEEE_Conf_Paper.pdf)


## Project Directory Structure:
### BanglaNLG: BUET CSE NLP Generative model
seq2seq translation 

### data
DataSetFold1_u.csv: gold dataset

pred: preprocessing, postprocessing

### data_old: 
data for old data, do not use or mixup with data

### dcspell data/model
Basically, bangla spell detection+correction dataset

* corpus: 1 million common spell error list (bad words)
* dictionary: BengaliWordList, 310 bangla corpus: merged
* bn_wiki_words: wikipedia title list
* **bad_words_notd_wiki**: words in corpus that are not in dictionary, not in wiki-title-dump
* model: ner, pos


### External_datasets
* sazzed: youtube comments sentiment dataset used for model blind validation

### ninth_place_tsd: code for model
* configs: ymls for training (train, dataset) and validation (eval)
* utils: combine_pred required for ensemble
* train.py: training
* eval.py: for validation and prediction (with_ground or not respectively)
* run.sh: commands
* batch.sh: evaluation in loop

### punctuation-restoration
* run.sh



## Literature

[Judge a Sentence by Its Content to Generate Grammatical Errors](https://arxiv.org/pdf/2208.09693.pdf)

[BSpell: A CNN-blended BERT Based Bengali Spell Checker](https://arxiv.org/pdf/2208.09709.pdf)

### Grammarly
[Under the Hood at Grammarly: Leveraging Transformer Language Models for Grammatical Error Correction](https://www.grammarly.com/blog/engineering/under-the-hood-at-grammarly-leveraging-transformer-language-models-for-grammatical-error-correction/)

[The Unreasonable Effectiveness of Transformer Language Models in
Grammatical Error Correction](https://arxiv.org/pdf/1906.01733.pdf)

[Adversarial Grammatical Error Correction](https://arxiv.org/pdf/2010.02407.pdf)

[MuCGEC: a Multi-Reference Multi-Source Evaluation Dataset for
Chinese Grammatical Error Correction](https://arxiv.org/pdf/2204.10994.pdf)
