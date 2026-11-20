import numpy as np
import pandas as pd

test = pd.read_csv('Datasets/test.csv')
classes = ['14AC78', '19E1A1', '6C0197', '72A2C8']
idx2label = {i: lbl for i, lbl in enumerate(classes)}

test_svm = np.load('Experiments/models/test_linearsvc_calibrated_probs.npy')
test_xlm = np.load('Experiments/oof/test_xlm_roberta_large.npy')
test_mur = np.load('Experiments/transformer_output/test_muril_base_probs.npy')

sub_live = pd.read_csv('Submission/submission_tri_ensemble_svm_xlm_muril_0.7269.csv')
preds_live = sub_live['label'].values

preds_70 = [idx2label[p] for p in np.argmax(0.70 * test_svm + 0.15 * test_xlm + 0.15 * test_mur, axis=1)]

diff_idx = np.where(preds_live != preds_70)[0]
print(f'Total diffs: {len(diff_idx)}')
diff_df = pd.DataFrame({
    'id': test['id'].iloc[diff_idx],
    'text': test['text'].iloc[diff_idx],
    'live_073041': preds_live[diff_idx],
    'fine_tuned_70': [preds_70[i] for i in diff_idx]
})

print(diff_df.groupby(['live_073041', 'fine_tuned_70']).size())

print('\nFirst 8 transitions:')
for _, r in diff_df.head(8).iterrows():
    t = str(r['text'])[:75].replace('\n', ' ')
    print(f"[{r['live_073041']} -> {r['fine_tuned_70']}] {t}")

