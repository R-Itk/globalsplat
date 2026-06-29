"""Tests for the exact mid-epoch resume support (``ResumableDataLoader``).

Covers (a) that Lightning recognizes it as stateful (so the "not resumable"
warning is suppressed and its state is checkpointed/restored), and (b) that
``load_state_dict`` + the next ``__iter__`` fast-forward to the exact position.
"""
import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import Dataset

from globalsplat.dataset.data_module import ResumableDataLoader


class _Range(Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return i


def _loader(n=10, **kw):
    # deterministic order: no shuffle, single worker -> batch i yields value i
    return ResumableDataLoader(_Range(n), batch_size=1, shuffle=False, num_workers=0, **kw)


def test_recognized_as_stateful_by_lightning():
    # This is exactly the isinstance check Lightning uses to decide a loader is
    # resumable (and to suppress the mid-epoch warning).
    _Stateful = pytest.importorskip("lightning_fabric.utilities.types")._Stateful
    assert isinstance(_loader(), _Stateful)


def test_state_dict_counts_batches_consumed():
    dl = _loader(10)
    it = iter(dl)
    next(it); next(it); next(it)
    assert dl.state_dict() == {"batches_seen": 3}


def test_load_state_dict_fast_forwards_to_exact_position():
    # Consume 4 batches, snapshot, then a fresh loader restored from that snapshot
    # must resume at value 4 (i.e. skip the 4 already-seen) and yield 4..9.
    src = _loader(10)
    it = iter(src)
    for _ in range(4):
        next(it)
    snap = src.state_dict()
    assert snap == {"batches_seen": 4}

    resumed = _loader(10)
    resumed.load_state_dict(snap)
    got = [int(b.item()) for b in resumed]
    assert got == [4, 5, 6, 7, 8, 9]
    # after the epoch, the cumulative count includes the skipped batches
    assert resumed.state_dict() == {"batches_seen": 10}


def test_skip_happens_only_once_then_full_epochs():
    resumed = _loader(5)
    resumed.load_state_dict({"batches_seen": 2})
    first = [int(b.item()) for b in resumed]   # skips 2 -> 2,3,4
    assert first == [2, 3, 4]
    second = [int(b.item()) for b in resumed]  # fresh epoch -> full 0..4, no skip
    assert second == [0, 1, 2, 3, 4]
    assert resumed.state_dict() == {"batches_seen": 5}


def test_skip_beyond_length_is_safe():
    resumed = _loader(3)
    resumed.load_state_dict({"batches_seen": 99})  # more than the epoch has
    assert [int(b.item()) for b in resumed] == []  # exhausted during fast-forward
    # next epoch is normal
    assert [int(b.item()) for b in resumed] == [0, 1, 2]


def test_no_skip_behaves_like_plain_loader():
    dl = _loader(4)
    assert [int(b.item()) for b in dl] == [0, 1, 2, 3]
    assert dl.state_dict() == {"batches_seen": 4}


if __name__ == "__main__":
    test_state_dict_counts_batches_consumed()
    test_load_state_dict_fast_forwards_to_exact_position()
    test_skip_happens_only_once_then_full_epochs()
    test_skip_beyond_length_is_safe()
    test_no_skip_behaves_like_plain_loader()
    print("test_resumable_loader: OK")
