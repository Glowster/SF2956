# loader.py
#source /Users/theodorbjork/SF2956/.venv/bin/activate
import json, numpy as np

meta = json.load(open('stft_np/meta.json'))
A = np.memmap('stft_np/'+meta['A_memmap'], dtype=np.float16, mode='r', shape=tuple(meta['A_shape']))
B = np.memmap('stft_np/'+meta['B_memmap'], dtype=np.float16, mode='r', shape=tuple(meta['B_shape']))
freqs = np.load('stft_np/freqs.npy')
print(A.shape, B.shape, freqs.shape)