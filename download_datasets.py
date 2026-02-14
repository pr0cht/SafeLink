import os
from datasets import load_dataset

os.makedirs('datasets', exist_ok=True)

print('uci download')
uci = load_dataset('ucirvine/sms_spam')
uci['train'].to_pandas().to_csv('datasets/sms_spam.csv', index=False)
print('dataset download - uci done')

print('wildguard download')
wildguard = load_dataset('walledai/WildGuardTest')
wildguard['train'].to_pandas().to_csv('datasets/wildguard.csv', index=False)
print('dataset download - wildguard done')


