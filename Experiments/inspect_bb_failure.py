import numpy as np
import pandas as pd

train = pd.read_csv('Datasets/train.csv')
test = pd.read_csv('Datasets/test.csv')

sub_live = pd.read_csv('Submission/submission_tri_ensemble_svm_xlm_muril_0.7269.csv')
sub_bb = pd.read_csv('Experiments/oof/submission_banglishbert_0.6781.csv')
sub_xlm = pd.read_csv('Experiments/oof/submission_xlm_roberta_large_0.6696.csv')

disagree = sub_bb['label'] != sub_live['label']
print('Total disagreements between BB and Live #1 Champion:', disagree.sum())

diff_df = pd.DataFrame({
    'id': test['id'][disagree],
    'text': test['text'][disagree],
    'live_pred': sub_live['label'][disagree],
    'bb_pred': sub_bb['label'][disagree]
})

print('\nDisagreement frequency by (live_pred -> bb_pred):')
print(diff_df.groupby(['live_pred', 'bb_pred']).size())

print('\nSample inspection of disagreements (first 10):')
for idx, row in diff_df.head(10).iterrows():
    t = str(row['text'])[:75].replace('\n', ' ')
    print(f"[{row['live_pred']} -> {row['bb_pred']}] {t}")

