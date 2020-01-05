# AICUP - TASK1
## team: SDML_SimpleBaseline (5th place)

### Requirements
***
+ torch：1.2.0
+ transformers：2.0.0

### Model
***
+ Sentence Encoder
    + SciBERT (scivocab, uncased)
    + Only get `[CLS]` embeddings as representation of sentences in abstract
        + add `[CLS]` token at the begin of each sentence
+ Classifier
    + Just vanilla Linear (768 -> 6)
        + simplest the best!
+ Training Procedure
    + Optimizer: Adam with linear warmup
    + Learning Rate: 1e-5
    + Batch Size: 32
    + Epoch: 3
    + Train : Test = 0.9 : 0.1
    + Loss Function: Binary Cross Entropy
    + Positive weight = [2.0, 2.25, 2.0, 2.87, 4.0, 8.7]

### Keys for improvement
***
+ Quality of training data
    + Just change the random seed
    + Vary dramatically with different training data
+ Thresholds for classifying 1 and 0
    + Easily overfit local validation set
    + As for how to tune them we leave it to `post-process` part
+ Since local validation score is not reliable, we upload to TBrain to test and keep the predictions with good performance
    + Finally we collect 95 predictions

### Post-process
***
+ Straightforward approach
    + Tune thresholds for each category
        + Best public F1 score w/o ensembling: **0.731**
        + Thresholds: **[0.56, 0.55, 0.45, 0.59, 0.59, 0.73]**
        + F1 score w/ ensembling: **0.736**
+ Advanced works for post-processing
    + Tune single threshold for every category, which is slightly higher than thresholds of above approach. Then flip prediction of 0s if predicted logits are greatest among the 6 categories but not pass the threshold
        + Best public F1 score w/o ensembling: **0.735**
        + Threshold: **0.7**
        + F1 score w/ ensembling: **0.737**
    + Different from task2, categories are not independent obviously in this task. For instance two adjacent categories such as RESULTS and CONCLUSIONS are more likely to be 1 simultaneously, we thus tune second thresholds for each category, which is less than the first threshold above. For each sentence predicted one-sentence and **not OTHERS** we flip the adjacent category to be 1 if the predicted logit is greater then its second threshold.
        + Best public F1 score w/o ensembling: **0.738**
        + Thresholds: **[(0, 1): 0.59, (1, 0): 0.59, (1, 2): 0.6, (2, 1): 0.6, (2, 3): 0.6, (3, 2), 0.6, (3, 4): 0.62, (4, 3): 0.62]**
            + where "$(a, b): t$" means $a$th category passes first threshold and $b$th category's second threshold will be $t$
        + F1 score w/ ensembling: **0.740**
    + Since OTHERS is hardest to predict, we found that in local validation number of TPs is far fewer than that of FPs and FNs. Therefore, we flip OTHERS to be 0 if a sentence is predicted multiple categories with OTHERS.
        + Best public F1 score w/o ensembling: **0.739**
        + F1 score w/ ensembling: **0.741**

### Failed Attempts
***
+ Training procedure
    + Different form of training data
        + Separate every sentence to be different input
            + Split directly: **0.701**
            + Accumulate sentences: **0.731**
            + Deprecated reason: too time-consuming and not better off
        + Replace `[CLS]` token with `[SEP]` token: **0.735**
            + Deprecated reason: slightly worse than using `[CLS]`
    + Different loss
        + Cross Entropy Loss
        + Multiclass-like training but predicting multilabel by giving another lower threshold
        + F1 score: **0.734**
        + Deprecated reason: also not better off
    + SciBERT + CRF
        + CRF is a statistical modeling method often applied in pattern recognition and structured prediction, which sounds super suitable for this task. However, the result was not quite well and it might be because of the strong enough strength of BERT embedding
        + F1 score: **0.727**
        + Deprecated reason: also not better off
+ Post-process
    + Train another classifier to learn how to label from logits
    + The two adjacent sentences are also dependent with the following prior transition probability
        ![](https://i.imgur.com/J7Ty9Q0.png)
        + No matter how we include this feature into post-process procedure, the F1 score can never be better. Thus we calculated the co-occurrence from prediction and found that the distribution predicted by our model is almost the same as the transition matrix. We can say that our model had learned to the limit with only embeddings from SciBERT!

### Reproducibility
***
+ train
    + `python bert_finetune_task1.py [seed] [gpu_id]`
        + seed: for both data spliting and torch
+ predict
    + `python predict_task1.py [seed] [gpu_id]`
        + seed: load model trained by this seed
+ ensemble
    + run all cells directly in `vote_task1.ipynb`
        + which will vote with all predictions