from pathlib import Path

import numpy as np
import pandas as pd

from mpd_df.dataset import (
    Alignment,
    SubjectFiles,
    build_alignment,
    build_window_index,
    iter_preprocessed_windows,
)
from mpd_df.preprocessing import PreprocessingConfig


def test_window_index_never_crosses_annotation_blocks() -> None:
    files = SubjectFiles("01", Path("dummy.edf"), Path("dummy.txt"))
    labels = np.array([0] * 30 + [1] * 30, dtype=np.int8)
    blocks = np.array([1] * 30 + [2] * 30, dtype=np.int32)
    alignment = Alignment("01", 100, 60, 0, labels, blocks)
    index = build_window_index(files, alignment, window_sec=10, stride_sec=5)
    assert len(index) == 10
    assert not any(
        row.window_start_sec < 30 < row.window_start_sec + row.window_sec
        for row in index.itertuples()
    )
    assert set(index["group_id"]) == {"01:1", "01:2"}
    assert index.loc[
        index["window_start_sec"].isin([20, 30]),
        "distance_to_transition_sec",
    ].eq(0).all()


def test_alignment_can_use_portable_manifest(monkeypatch, tmp_path) -> None:
    annotation = tmp_path / "annotation.txt"
    annotation.write_text("12:00:00,0,0\n12:00:30,1,1\n")
    manifest = tmp_path / "alignment_manifest.csv"
    pd.DataFrame(
        [
            {
                "subject": "01",
                "eeg_start_clock_sec": 43199,
                "aligned_start_clock_sec": 43200,
                "duration_sec": 59,
            }
        ]
    ).to_csv(manifest, index=False)
    files = SubjectFiles(
        subject="01",
        eeg=tmp_path / "eeg.edf",
        annotation=annotation,
        alignment_manifest=manifest,
    )
    monkeypatch.setattr(
        "mpd_df.dataset.read_edf_header",
        lambda _: {
            "start_clock_sec": 43199,
            "duration_sec": 61.0,
            "sfreq": 500.0,
            "n_channels": 32,
            "ch_names": (),
            "n_samples": 30500,
        },
    )

    alignment = build_alignment(files)

    assert alignment.eeg_offset_sec == 1
    assert alignment.duration_sec == 59
    assert np.bincount(alignment.labels).tolist() == [30, 29]


def test_group_bounded_filter_does_not_see_adjacent_group(monkeypatch) -> None:
    sfreq = 100.0
    signal = np.zeros((1, 6000))
    signal[0, 3000] = 1e6

    class FakeRaw:
        info = {"sfreq": sfreq}
        ch_names = ["Cz"]
        n_times = signal.shape[1]

        def pick(self, _):
            return self

        def get_data(self, start, stop):
            return signal[:, start:stop]

    monkeypatch.setattr("mpd_df.dataset.mne.io.read_raw_edf", lambda *_, **__: FakeRaw())
    files = SubjectFiles("01", Path("dummy.edf"), Path("dummy.txt"))
    alignment = Alignment(
        "01",
        0,
        60,
        0,
        np.zeros(60, dtype=np.int8),
        np.repeat([0, 1], 30).astype(np.int32),
    )
    index = pd.DataFrame(
        [
            {
                "subject": "01",
                "window_start_sec": 29,
                "eeg_start_sec": 29,
                "window_sec": 1,
                "label": 0,
                "block_id": 0,
                "group_id": "01:0",
            },
            {
                "subject": "01",
                "window_start_sec": 30,
                "eeg_start_sec": 30,
                "window_sec": 1,
                "label": 0,
                "block_id": 1,
                "group_id": "01:1",
            },
        ]
    )
    config = PreprocessingConfig(
        name="test",
        l_freq=1.0,
        h_freq=35.0,
        notch_freq=None,
        target_sfreq=None,
        demean=False,
        normalization="none",
    )

    bounded = list(
        iter_preprocessed_windows(
            files,
            alignment,
            index,
            ["Cz"],
            config,
            filter_scope="group_bounded",
        )
    )
    continuous = next(
        iter_preprocessed_windows(
            files,
            alignment,
            index,
            ["Cz"],
            config,
            filter_scope="subject_continuous",
        )
    )[0]

    assert np.max(np.abs(bounded[0][0])) == 0.0
    assert np.max(np.abs(continuous[0])) > 0.0
