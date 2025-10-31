# make_stft_arrays.py
# Usage: python make_stft_arrays.py
import os, glob, json
import numpy as np
from scipy import signal
from scipy.signal import resample_poly
import soundfile as sf  # pip install soundfile

# === Config (adjust if your folders differ) ===
A_DIR = 'fma/clips/A'
B_DIR = 'fma/clips/B'
OUT_DIR = 'stft_np'

SR        = 48_000           # target sample rate
WIN_SEC   = 0.01             # 10 ms windows
WIN_SAMP  = int(SR * WIN_SEC) # 480 samples
N_FFT     = 512              # ~93.75 Hz bins at 48k; use 1024 for ~46.9 Hz
HOP_SAMP  = WIN_SAMP         # no overlap (one 10 ms frame step)
DUR_SEC   = 10.0             # clips are 10 s
NSAMP_OUT = int(SR * DUR_SEC)  # 480000 samples
N_FRAMES  = 1 + (NSAMP_OUT - WIN_SAMP) // HOP_SAMP  # -> 1000
WINDOW    = 'hann'
DTYPE_OUT = np.float16       # compact storage (dB magnitudes)

os.makedirs(OUT_DIR, exist_ok=True)

def load_wav_mono(path, target_sr=SR, nsamp=NSAMP_OUT):
    x, sr = sf.read(path, always_2d=False)
    # to mono
    if x.ndim == 2:
        x = x.mean(axis=1)
    # resample if needed (high-quality polyphase)
    if sr != target_sr:
        # resample_poly up=target_sr, down=sr
        x = resample_poly(x, target_sr, sr)
    # trim/pad to exactly nsamp
    if x.shape[0] < nsamp:
        xp = np.zeros(nsamp, dtype=np.float32)
        xp[:x.shape[0]] = x.astype(np.float32, copy=False)
        x = xp
    elif x.shape[0] > nsamp:
        x = x[:nsamp]
    else:
        x = x.astype(np.float32, copy=False)
    return x

def stft_mag_db(x):
    # STFT (no centering/padding): boundary=None, noverlap=0
    freqs, times, Z = signal.stft(
        x, fs=SR, window=WINDOW, nperseg=WIN_SAMP, noverlap=0,
        nfft=N_FFT, boundary=None, padded=False, return_onesided=True
    )
    # |Z| -> dB (clip floor to -80 dB for compact float16)
    mag = np.abs(Z).T  # (frames, bins)
    db = 20.0 * np.log10(np.maximum(mag, 1e-6))
    db = np.clip(db, -80.0, 0.0)
    return db.astype(np.float32, copy=False), freqs

def collect_files(folder):
    files = sorted(glob.glob(os.path.join(folder, '*.wav')))
    if not files:
        raise RuntimeError(f'No WAV files found in {folder}')
    return files

def write_memmap(files, out_path, freqs_ref=None):
    # pre-create memmap: (num_files, N_FRAMES, N_BINS)
    # we’ll discover N_BINS from first file
    x0 = load_wav_mono(files[0])
    db0, freqs = stft_mag_db(x0)
    if db0.shape[0] != N_FRAMES:
        # ensure exact number of frames by pad/truncate in time if needed
        if db0.shape[0] < N_FRAMES:
            pad = np.full((N_FRAMES - db0.shape[0], db0.shape[1]), -80.0, dtype=np.float32)
            db0 = np.vstack([db0, pad])
        else:
            db0 = db0[:N_FRAMES, :]

    if freqs_ref is not None:
        # sanity: same bins
        if len(freqs) != len(freqs_ref) or not np.allclose(freqs, freqs_ref):
            raise RuntimeError("Frequency bins mismatch across files")
    N_BINS = db0.shape[1]
    mm = np.memmap(out_path, dtype=DTYPE_OUT, mode='w+', shape=(len(files), N_FRAMES, N_BINS))

    # write first
    mm[0] = db0.astype(DTYPE_OUT)
    # remaining
    for i, fp in enumerate(files[1:], start=1):
        x = load_wav_mono(fp)
        db, _ = stft_mag_db(x)
        if db.shape[0] < N_FRAMES:
            pad = np.full((N_FRAMES - db.shape[0], N_BINS), -80.0, dtype=np.float32)
            db = np.vstack([db, pad])
        elif db.shape[0] > N_FRAMES:
            db = db[:N_FRAMES, :]
        mm[i] = db.astype(DTYPE_OUT)
        if (i+1) % 100 == 0:
            print(f'{os.path.basename(out_path)}: {i+1}/{len(files)} done')
    del mm  # flush to disk
    return freqs

def main():
    files_A = collect_files(A_DIR)
    files_B = collect_files(B_DIR)

    print(f'Found {len(files_A)} files in {A_DIR}, {len(files_B)} in {B_DIR}')
    # A
    freqs = write_memmap(files_A, os.path.join(OUT_DIR, 'A_stft.float16.memmap'))
    # B (check same bins)
    _ = write_memmap(files_B, os.path.join(OUT_DIR, 'B_stft.float16.memmap'), freqs_ref=freqs)

    # Save frequencies and metadata
    np.save(os.path.join(OUT_DIR, 'freqs.npy'), freqs.astype(np.float32))
    meta = {
        'sr': SR,
        'win_sec': WIN_SEC,
        'win_samp': WIN_SAMP,
        'hop_samp': HOP_SAMP,
        'n_fft': N_FFT,
        'window': WINDOW,
        'frames_per_clip': N_FRAMES,
        'dtype': str(DTYPE_OUT),
        'A_shape': [len(files_A), N_FRAMES, len(freqs)],
        'B_shape': [len(files_B), N_FRAMES, len(freqs)],
        'A_memmap': 'A_stft.float16.memmap',
        'B_memmap': 'B_stft.float16.memmap',
        'freqs_npy': 'freqs.npy',
        'paths_A_txt': 'paths_A.txt',
        'paths_B_txt': 'paths_B.txt',
    }
    with open(os.path.join(OUT_DIR, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    # Save file index lists
    with open(os.path.join(OUT_DIR, 'paths_A.txt'), 'w') as f:
        f.write('\n'.join(files_A))
    with open(os.path.join(OUT_DIR, 'paths_B.txt'), 'w') as f:
        f.write('\n'.join(files_B))

    print('Done.')
    print(f"A array: {meta['A_shape']} → {os.path.join(OUT_DIR, meta['A_memmap'])}")
    print(f"B array: {meta['B_shape']} → {os.path.join(OUT_DIR, meta['B_memmap'])}")
    print(f"Freq bins: {len(freqs)} saved to {os.path.join(OUT_DIR, 'freqs.npy')}")

if __name__ == '__main__':
    main()