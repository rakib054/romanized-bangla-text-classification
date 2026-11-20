import json

with open('Experiments/reports/cleanlab_label_issues.json', 'r', encoding='utf-8') as f:
    report = json.load(f)

print('=== Cleanlab Summary ===')
for k, v in report['summary'].items():
    print(f'  {k}: {v}')

print('\n=== Consensus Label Issues by Class ===')
for k, v in report['consensus_by_class'].items():
    print(f'  {k}: {v}')

print('\n=== Top 5 Examples of Confident Label Issues in Dataset ===')
for ex in report['top_flagged_examples'][:5]:
    print(f"ID: {ex['id']} | Given: {ex['given_label']} -> Predicted: {ex['pred_linearsvc']} (P_pred={ex['prob_svm_pred']:.2f}, P_given={ex['prob_svm_given']:.2f})")
    print(f"Text: \"{ex['text']}\"\n")

