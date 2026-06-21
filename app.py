"""
EE200 Course Project, Q3B: 'Zapptain America'
"""

import io
import os
import glob
import pickle
import gzip
import tempfile
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import librosa
import librosa.display
import librosa.effects
import scipy.ndimage as ndimage


SONGS_DIR = "songs"                 
DB_PAIR_PATH = "song_database_pairs.pkl.gz"
DB_SINGLE_PATH = "song_database_single.pkl.gz"

WINDOW = 4096
HOP = WINDOW // 2
NEIGHBORHOOD_SIZE = 20
MIN_TIME_DELTA = 0.1
MAX_TIME_DELTA = 5.0
FAN_OUT = 3

st.set_page_config(page_title="Sonic Signatures — Song Identifier", layout="wide")


def extract_peaks(y, sr, window=WINDOW):
    hop = window // 2
    D = librosa.stft(y, n_fft=window, hop_length=hop)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    frequencies = librosa.fft_frequencies(sr=sr, n_fft=window)
    times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr, hop_length=hop)

    local_max_filter = ndimage.maximum_filter(S_db, size=NEIGHBORHOOD_SIZE)
    is_local_max = (S_db == local_max_filter)

    volume_threshold = np.mean(S_db)
    is_loud_enough = (S_db > volume_threshold)

    valid_peaks = is_local_max & is_loud_enough
    freq_indices, time_indices = np.where(valid_peaks)

    peak_times = times[time_indices]
    peak_freqs = frequencies[freq_indices]

    return S_db, frequencies, times, peak_times, peak_freqs


def hash_generator(peak_times, peak_freqs):
    hashes = []
    num_peaks = len(peak_times)

    sort_indices = np.argsort(peak_times)
    sorted_times = peak_times[sort_indices]
    sorted_freqs = peak_freqs[sort_indices]

    for i in range(num_peaks):
        anchor_time = sorted_times[i]
        anchor_freq = sorted_freqs[i]

        pairs_made = 0
        for j in range(i + 1, num_peaks):
            target_time = sorted_times[j]
            target_freq = sorted_freqs[j]

            time_delta = target_time - anchor_time
            if time_delta < MIN_TIME_DELTA:
                continue
            if time_delta > MAX_TIME_DELTA or pairs_made >= FAN_OUT:
                break

            hash_signature = (
                round(anchor_freq, 1),
                round(target_freq, 1),
                round(time_delta, 2),
            )
            hashes.append({'hash': hash_signature, 'offset': anchor_time})
            pairs_made += 1

    return hashes


def hash_generator_single(peak_times, peak_freqs):
    hashes = []
    sort_indices = np.argsort(peak_times)
    sorted_times = peak_times[sort_indices]
    sorted_freqs = peak_freqs[sort_indices]

    for i in range(len(sorted_times)):
        hash_signature = round(sorted_freqs[i], 1)
        hashes.append({'hash': hash_signature, 'offset': sorted_times[i]})

    return hashes


def index_song(song_name, song_hashes, database):
    for h in song_hashes:
        database.setdefault(h['hash'], []).append((song_name, h['offset']))


def offset_histogram(query_hashes, database, target_song):
    offsets = []
    for q_hash in query_hashes:
        sig = q_hash['hash']
        if sig in database:
            for db_song, db_time in database[sig]:
                if db_song == target_song:
                    offsets.append(round(db_time - q_hash['offset'], 1))
    return Counter(offsets)


def predict_song(query_hashes, database):
    matches = defaultdict(Counter)
    for q_hash in query_hashes:
        sig = q_hash['hash']
        if sig in database:
            for db_song, db_time in database[sig]:
                delta_time = round(db_time - q_hash['offset'], 1)
                matches[db_song][delta_time] += 1

    best_song, best_score = "No Match Found", 0
    for song, offs in matches.items():
        if offs:
            score = max(offs.values())
            if score > best_score:
                best_score = score
                best_song = song

    return best_song, best_score


def add_random_noise(audio_data, noise_multiplier):
    noise = np.random.randn(len(audio_data))
    return audio_data + (noise_multiplier * noise)


def apply_time_stretch(audio_data, rate):
    return librosa.effects.time_stretch(audio_data, rate=rate)

# Static database

@st.cache_resource(show_spinner="Indexing the static song database (one-time)...")
def build_databases():
    

    if os.path.exists(DB_PAIR_PATH) and os.path.exists(DB_SINGLE_PATH):
        with gzip.open(DB_PAIR_PATH, 'rb') as f:
            db_pairs = pickle.load(f)
        with gzip.open(DB_SINGLE_PATH, 'rb') as f:
            db_single = pickle.load(f)
        song_list = sorted({s for v in db_pairs.values() for s, _ in v})
        return db_pairs, db_single, song_list

    db_pairs, db_single = {}, {}
    song_list = []

    audio_files = sorted(
        glob.glob(os.path.join(SONGS_DIR, '*.mp3'))
        + glob.glob(os.path.join(SONGS_DIR, '*.wav'))
    )

    for path in audio_files:
        song_name = os.path.splitext(os.path.basename(path))[0]
        y, sr = librosa.load(path, sr=None)
        _, _, _, peak_times, peak_freqs = extract_peaks(y, sr)

        index_song(song_name, hash_generator(peak_times, peak_freqs), db_pairs)
        index_song(song_name, hash_generator_single(peak_times, peak_freqs), db_single)
        song_list.append(song_name)

    if song_list:
        with open(DB_PAIR_PATH, 'wb') as f:
            pickle.dump(db_pairs, f)
        with open(DB_SINGLE_PATH, 'wb') as f:
            pickle.dump(db_single, f)

    return db_pairs, db_single, song_list

# Plotting Intermediate

def plot_waveform(y, sr):
    fig, ax = plt.subplots(figsize=(6, 2.6))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_dft_magnitude(y, sr):
    """Full-clip DFT magnitude — shows WHICH frequencies are present but
    not WHEN, motivating the move to a spectrogram."""
    fft_result = np.fft.fft(y)
    magnitude = np.abs(fft_result)
    n = len(y)
    frequencies = np.fft.fftfreq(n, d=1.0 / sr)
    half_point = n // 2

    fig, ax = plt.subplots(figsize=(6, 2.6))
    ax.plot(frequencies[:half_point], magnitude[:half_point], linewidth=0.6)
    ax.set_title("DFT Magnitude Spectrum (whole clip)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_spectrogram_with_window(y, sr, window):
    hop = window // 2
    D = librosa.stft(y, n_fft=window, hop_length=hop)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    fig, ax = plt.subplots(figsize=(6, 4))
    img = librosa.display.specshow(
        S_db, sr=sr, hop_length=hop, x_axis='time', y_axis='hz', cmap='magma', ax=ax
    )
    ax.set_ylim(0, 5000)
    ax.set_title(f"Spectrogram — window = {window} samples")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    return fig


def plot_spectrogram(S_db, sr, hop):
    fig, ax = plt.subplots(figsize=(6, 4))
    img = librosa.display.specshow(
        S_db, sr=sr, hop_length=hop, x_axis='time', y_axis='hz', cmap='magma', ax=ax
    )
    ax.set_ylim(0, 5000)
    ax.set_title("Spectrogram")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    return fig


def plot_constellation(S_db, sr, hop, peak_times, peak_freqs):
    fig, ax = plt.subplots(figsize=(6,4))
    img = librosa.display.specshow(
        S_db, sr=sr, hop_length=hop, x_axis='time', y_axis='hz', cmap='magma', ax=ax
    )
    ax.scatter(
        peak_times, peak_freqs, s=5, marker='o',
        facecolors='none', edgecolors='yellow', linewidths=1.2
    )
    ax.set_ylim(0, 5000)
    ax.set_title("Constellation Map")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    return fig


def plot_offset_histogram(hist, song_name):
    fig, ax = plt.subplots(figsize=(6, 3))
    if hist:
        offsets = list(hist.keys())
        counts = list(hist.values())
        ax.bar(offsets, counts, width=0.1, color='steelblue', edgecolor='navy')
    else:
        ax.text(0.5, 0.5, "No matching hashes", ha='center', va='center', transform=ax.transAxes)
    ax.set_title(f"Offset Histogram — {song_name}")
    ax.set_xlabel("Time Offset (s)")
    ax.set_ylabel("Matching Hashes")
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    return fig


def load_uploaded_audio(uploaded_file):
    """Writes an uploaded Streamlit file to a temp path and loads it with librosa."""
    suffix = os.path.splitext(uploaded_file.name)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    try:
        y, sr = librosa.load(tmp_path, sr=None)
    finally:
        os.remove(tmp_path)
    return y, sr

# App layout

st.title("🎵 Sonic Signatures — Audio Fingerprint Identifier")

db_pairs, db_single, song_list = build_databases()

with st.sidebar:
    st.header("Database")
    if song_list:
        st.success(f"{len(song_list)} song(s) indexed")
        with st.expander("Show indexed songs"):
            for s in song_list:
                st.write("•", s)
    else:
        st.error(
            f"No audio files found in '{SONGS_DIR}/'.\n\n"
            "Add the provided song database's .mp3 / .wav files to that "
            "folder and restart the app."
        )

    st.divider()
    mode = st.radio("Mode", ["Single Clip", "Batch Mode"])

    st.divider()
    st.caption(
        "Note: pairing two peaks into one hash (Δfreq, Δtime) makes a true "
        "match line up at one offset very strongly, while single-peak "
        "hashes are far more common across songs and give noisier, less "
        "decisive matches."
    )


# ---------------------------- Single-clip mode ----------------------------

if mode == "Single Clip":
    st.subheader("Single-clip identification")

    uploaded = st.file_uploader(
        "Upload a query clip (a short excerpt is enough)",
        type=["mp3", "wav", "ogg", "m4a", "flac"],
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        add_noise = st.checkbox("Add noise")
        noise_level = st.slider(
            "Noise multiplier", 0.0, 0.05, 0.01, 0.005, disabled=not add_noise
        )
    with col2:
        add_stretch = st.checkbox("Time-stretch (≈ pitch test)")
        stretch_rate = st.slider(
            "Stretch rate (1.0 = unchanged)", 0.8, 1.2, 1.0, 0.01, disabled=not add_stretch
        )
    with col3:
        use_single_peaks = st.checkbox(
            "Use single-peak hashing instead of paired hashing",
            help="For comparing against the default, more decisive pair-based hashing.",
        )

    run = st.button("Identify Song", type="primary", disabled=uploaded is None)

    if run and uploaded is not None:
        if not song_list:
            st.error("The database is empty — add songs to the 'songs/' folder first.")
        else:
            y, sr = load_uploaded_audio(uploaded)

            if add_noise:
                y = add_random_noise(y, noise_level)
            if add_stretch and stretch_rate != 1.0:
                y = apply_time_stretch(y, stretch_rate)

            S_db, frequencies, times, peak_times, peak_freqs = extract_peaks(y, sr)

            if use_single_peaks:
                query_hashes = hash_generator_single(peak_times, peak_freqs)
                database = db_single
            else:
                query_hashes = hash_generator(peak_times, peak_freqs)
                database = db_pairs

            best_song, best_score = predict_song(query_hashes, database)

            st.markdown("### Result")
            if best_song != "No Match Found":
                st.success(f"🎶 Identified as **{best_song}**  (confidence score: {best_score})")
            else:
                st.warning("No match found.")

            st.markdown("### Intermediate Steps")

            tab_wave, tab_dft, tab_spec, tab_const, tab_hist = st.tabs(
                ["Waveform", "DFT Spectrum", "Spectrogram", "Constellation", "Offset Histogram"]
            )

            with tab_wave:
                st.pyplot(plot_waveform(y, sr), use_container_width=True)
                st.caption("The raw time-domain signal that was fingerprinted (after any noise/stretch applied above).")

            with tab_dft:
                st.pyplot(plot_dft_magnitude(y, sr), use_container_width=True)
                st.caption(
                    "A single DFT over the whole clip shows *which* frequencies are present, "
                    "but all timing information is gone — this is exactly why we need a "
                    "spectrogram instead of a single Fourier transform."
                )

            with tab_spec:
                st.pyplot(plot_spectrogram(S_db, sr, HOP), use_container_width=True)
                st.caption("STFT magnitude (dB) over time — frequency content as it evolves through the clip.")

            with tab_const:
                st.pyplot(plot_constellation(S_db, sr, HOP, peak_times, peak_freqs), use_container_width=True)
                st.caption("Only the strongest, locally-maximal time-frequency points are kept as the clip's fingerprint.")

            with tab_hist:
                if best_song != "No Match Found":
                    hist = offset_histogram(query_hashes, database, best_song)
                    st.pyplot(plot_offset_histogram(hist, best_song), use_container_width=True)
                    st.caption(
                        "A sharp, tall peak at one offset means the query's hashes line up "
                        "consistently with the matched song; a flat, scattered histogram "
                        "means the match is coincidental."
                    )
                else:
                    st.info("No matched song, so there is no offset histogram to show.")

            with st.expander("Bonus: short window vs. long window (time–frequency resolution trade-off)"):
                wc1, wc2 = st.columns(2)
                with wc1:
                    st.pyplot(plot_spectrogram_with_window(y, sr, 1024), use_container_width=True)
                    st.caption("Short window → blurry frequency).")
                with wc2:
                    st.pyplot(plot_spectrogram_with_window(y, sr, 16384), use_container_width=True)
                    st.caption("Long window → precise frequency, blurry timing.")


# ------------------------------- Batch mode -------------------------------

else:
    st.subheader("Batch identification")
    st.write(
        "Upload several query clips at once. The app fingerprints each one "
        "(using the default, paired-hash method) against the static "
        "database and produces a `results.csv` with exactly two columns: "
        "`filename, prediction`."
    )

    uploaded_files = st.file_uploader(
        "Upload query clips",
        type=["mp3", "wav", "ogg", "m4a", "flac"],
        accept_multiple_files=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        add_noise_b = st.checkbox("Add noise to all clips (robustness test)", key="b_noise")
        noise_level_b = st.slider(
            "Noise multiplier", 0.0, 0.05, 0.01, 0.005, key="b_noise_lvl", disabled=not add_noise_b
        )
    with col2:
        add_stretch_b = st.checkbox("Time-stretch all clips (robustness test)", key="b_stretch")
        stretch_rate_b = st.slider(
            "Stretch rate", 0.8, 1.2, 1.0, 0.01, key="b_stretch_rate", disabled=not add_stretch_b
        )

    run_batch = st.button(
        "Run Batch Identification", type="primary",
        disabled=not uploaded_files or not song_list,
    )

    if run_batch:
        results = []
        progress = st.progress(0.0)
        status = st.empty()

        for i, f in enumerate(uploaded_files):
            status.write(f"Processing `{f.name}` ...")
            y, sr = load_uploaded_audio(f)

            if add_noise_b:
                y = add_random_noise(y, noise_level_b)
            if add_stretch_b and stretch_rate_b != 1.0:
                y = apply_time_stretch(y, stretch_rate_b)

            _, _, _, peak_times, peak_freqs = extract_peaks(y, sr)
            query_hashes = hash_generator(peak_times, peak_freqs)
            best_song, _ = predict_song(query_hashes, db_pairs)
            prediction = best_song if best_song != "No Match Found" else ""

            results.append({"filename": f.name, "prediction": prediction})
            progress.progress((i + 1) / len(uploaded_files))

        status.write("Done.")
        df = pd.DataFrame(results, columns=["filename", "prediction"])
        st.dataframe(df, use_container_width=True)

        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        st.download_button(
            "Download results.csv",
            data=csv_buf.getvalue(),
            file_name="results.csv",
            mime="text/csv",
        )