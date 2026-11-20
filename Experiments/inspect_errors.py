import json
import re
from collections import Counter

with open('Experiments/reports/misclassified_samples_19_vs_72.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

s_19_to_72 = data['samples_19E1A1_pred_as_72A2C8']
s_72_to_19 = data['samples_72A2C8_pred_as_19E1A1']

print("=" * 80)
print(f"ANALYZING 50 SAMPLES: True 19E1A1 predicted as 72A2C8")
print("=" * 80)
for i, s in enumerate(s_19_to_72[:10], 1):
    print(f"[{i:02d}] ID: {s['id']} | WordLen: {s['word_len']:2d} | P(19)={s['prob_19']:.2f}, P(72)={s['prob_72']:.2f}")
    print(f"     \"{s['text']}\"\n")

print("=" * 80)
print(f"ANALYZING 50 SAMPLES: True 72A2C8 predicted as 19E1A1")
print("=" * 80)
for i, s in enumerate(s_72_to_19[:10], 1):
    print(f"[{i:02d}] ID: {s['id']} | WordLen: {s['word_len']:2d} | P(19)={s['prob_19']:.2f}, P(72)={s['prob_72']:.2f}")
    print(f"     \"{s['text']}\"\n")

# Extract vocabulary / lexical patterns in both groups
def get_words(samples):
    words = []
    for s in samples:
        tokens = re.findall(r'\b[a-zA-Z-]+\b', s['text'].lower())
        words.extend(tokens)
    return Counter(words)

c_19_to_72 = get_words(s_19_to_72)
c_72_to_19 = get_words(s_72_to_19)

print("\n--- Top Distinctive Words in 19E1A1 -> 72A2C8 Errors ---")
for w, cnt in c_19_to_72.most_common(20):
    print(f"  {w:<15}: {cnt}")

print("\n--- Top Distinctive Words in 72A2C8 -> 19E1A1 Errors ---")
for w, cnt in c_72_to_19.most_common(20):
    print(f"  {w:<15}: {cnt}")

