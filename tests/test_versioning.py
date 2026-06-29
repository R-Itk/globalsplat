"""Tests for versioned checkpoint dirs + resume resolution
(``globalsplat.checkpointing``). Pure stdlib logic exercised against tmp dirs.
"""
import os

from globalsplat.checkpointing import (
    list_versions,
    parse_version_from_path,
    next_version,
    latest_last_ckpt,
    resolve_run,
)


def _mk_version(base, n, with_last=False):
    d = os.path.join(base, f"version_{n}")
    os.makedirs(d, exist_ok=True)
    if with_last:
        open(os.path.join(d, "last.ckpt"), "w").close()
    return d


def test_list_and_next_version(tmp_path):
    base = str(tmp_path / "exp")
    assert list_versions(base) == []
    assert next_version(base) == 0
    _mk_version(base, 0)
    _mk_version(base, 1)
    _mk_version(base, 3)
    assert list_versions(base) == [0, 1, 3]
    assert next_version(base) == 4  # max + 1, even with a gap


def test_parse_version_from_path():
    assert parse_version_from_path("/a/exp/version_2/last.ckpt") == 2
    assert parse_version_from_path("/a/version_0/version_5/step.ckpt") == 5  # innermost
    assert parse_version_from_path("/a/checkpoints/last.ckpt") is None
    assert parse_version_from_path(None) is None


def test_fresh_run(tmp_path):
    base = str(tmp_path / "exp")
    run = resolve_run(load=None, resume=False, auto_resume=True, exp_ckpt_base=base)
    assert run.version == 0
    assert run.resume_ckpt is None and run.weights_only is None
    assert run.ckpt_dir == os.path.join(base, "version_0")


def test_auto_resume_picks_latest_last(tmp_path):
    base = str(tmp_path / "exp")
    _mk_version(base, 0, with_last=True)
    _mk_version(base, 1, with_last=True)
    _mk_version(base, 2, with_last=False)  # newer but no last.ckpt
    found = latest_last_ckpt(base)
    assert found is not None and found[0] == 1

    run = resolve_run(load=None, resume=False, auto_resume=True, exp_ckpt_base=base)
    assert run.version == 1
    assert run.resume_ckpt == os.path.join(base, "version_1", "last.ckpt")
    assert run.ckpt_dir == os.path.join(base, "version_1")


def test_auto_resume_disabled_starts_fresh(tmp_path):
    base = str(tmp_path / "exp")
    _mk_version(base, 0, with_last=True)
    run = resolve_run(load=None, resume=False, auto_resume=False, exp_ckpt_base=base)
    assert run.resume_ckpt is None
    assert run.version == 1  # next, not the existing 0
    assert run.ckpt_dir == os.path.join(base, "version_1")


def test_explicit_resume_continues_in_version_dir(tmp_path):
    base = str(tmp_path / "exp")
    vdir = _mk_version(base, 2, with_last=True)
    load = os.path.join(vdir, "last.ckpt")
    run = resolve_run(load=load, resume=True, auto_resume=True, exp_ckpt_base=base)
    assert run.resume_ckpt == load
    assert run.version == 2
    # continues writing into the checkpoint's own directory
    assert os.path.abspath(run.ckpt_dir) == os.path.abspath(vdir)


def test_explicit_resume_nonversioned_path_gets_fresh_version(tmp_path):
    base = str(tmp_path / "exp")
    _mk_version(base, 0)  # existing -> next is 1
    load = str(tmp_path / "some" / "arbitrary.ckpt")
    run = resolve_run(load=load, resume=True, auto_resume=True, exp_ckpt_base=base)
    assert run.resume_ckpt == load
    assert run.version == 1
    assert run.ckpt_dir == os.path.join(base, "version_1")


def test_weights_only_finetune_starts_fresh_version(tmp_path):
    base = str(tmp_path / "exp")
    vdir = _mk_version(base, 0, with_last=True)
    load = os.path.join(vdir, "last.ckpt")  # source run, must NOT be clobbered
    run = resolve_run(load=load, resume=False, auto_resume=True, exp_ckpt_base=base)
    assert run.weights_only == load
    assert run.resume_ckpt is None
    assert run.version == 1
    assert run.ckpt_dir == os.path.join(base, "version_1")


def test_next_version_considers_both_ckpt_and_log_bases(tmp_path):
    ckpt_base = str(tmp_path / "ckpt" / "exp")
    log_base = str(tmp_path / "logs" / "exp")
    _mk_version(ckpt_base, 0)
    _mk_version(log_base, 0)
    _mk_version(log_base, 1)  # logs ahead of checkpoints
    run = resolve_run(
        load=None, resume=False, auto_resume=False,
        exp_ckpt_base=ckpt_base, log_exp_base=log_base,
    )
    assert run.version == 2  # max across both bases + 1


if __name__ == "__main__":
    import tempfile, pathlib
    for fn in [v for k, v in dict(globals()).items() if k.startswith("test_")]:
        if "tmp_path" in fn.__code__.co_varnames:
            with tempfile.TemporaryDirectory() as d:
                fn(pathlib.Path(d))
        else:
            fn()
    print("test_versioning: OK")
