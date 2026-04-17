"""Periodic rsync of local save_dir to a shared NFS mirror.

A daemon thread running on rank 0 mirrors ``trainer.save_dir`` to a target
directory on shared NFS every ``interval_sec`` seconds, and does a final
sync on training end. Non-blocking: rsync runs in a separate process and
errors are swallowed so training is never interrupted. If save_dir already
lives on NFS the sync is skipped.
"""

import subprocess
import threading
from pathlib import Path

from ultralytics.utils import LOGGER, RANK


def setup(nfs_mirror_root: str | Path, interval_sec: int = 600):
    """Build (on_train_start, on_train_end) callbacks that mirror save_dir to NFS.

    Args:
        nfs_mirror_root (str | Path): NFS directory that will contain the run dir; the save_dir basename is appended to
            form the final target.
        interval_sec (int, optional): Seconds between rsync passes.

    Returns:
        (tuple): (on_train_start callback, on_train_end callback).
    """
    stop_event = threading.Event()
    state: dict = {}

    def _rsync(src: str, dst: str) -> None:
        Path(dst).mkdir(parents=True, exist_ok=True)
        cmd = ["rsync", "-a", "--partial", "--exclude=*.pt.tmp", f"{src.rstrip('/')}/", f"{dst.rstrip('/')}/"]
        subprocess.run(cmd, check=False, capture_output=True)

    def _loop() -> None:
        while not stop_event.wait(interval_sec):
            try:
                _rsync(state["src"], state["dst"])
            except Exception as e:
                LOGGER.warning(f"nfs_sync: rsync failed, will retry next interval: {e}")

    def on_train_start(trainer) -> None:
        if RANK not in (-1, 0):
            return
        src = str(Path(trainer.save_dir).resolve())
        if src.startswith(str(nfs_mirror_root)):
            LOGGER.info(f"nfs_sync: save_dir already under NFS mirror root, skipping ({src})")
            return
        dst = str(Path(nfs_mirror_root) / Path(src).name)
        state.update(src=src, dst=dst)
        threading.Thread(target=_loop, daemon=True, name="nfs-sync").start()
        LOGGER.info(f"nfs_sync: mirroring {src} -> {dst} every {interval_sec}s")

    def on_train_end(trainer) -> None:
        if RANK not in (-1, 0) or "src" not in state:
            return
        stop_event.set()
        dst = state["dst"]
        _rsync(state["src"], dst)
        LOGGER.info(f"nfs_sync: final sync complete -> {dst}")

    return on_train_start, on_train_end
