"""Refuse a memory-hungry local run before it swaps the machine to death."""

import os
import platform
import subprocess

from editjumps.core.utils import get_logger

logger = get_logger(__file__)

#: Workloads below this are cheap enough that memory never matters — skip the check entirely.
LARGE_WORKLOAD_ITEMS = 200_000

#: Free-memory floor for a large local run.
DEFAULT_MIN_FREE_GB = 6.0

#: Swap this full means the machine is already in trouble; another big run is what tips it.
MAX_SWAP_USED_FRACTION = 0.80

ALLOW_ENV = "EDITJUMPS_ALLOW_LOCAL_HEAVY"


def _macos_memory_gb() -> tuple[float | None, float | None]:
    """Free RAM and used-swap fraction on macOS, or ``(None, None)`` if unreadable."""
    free = swap_used = None
    try:
        page = int(subprocess.run(["sysctl", "-n", "hw.pagesize"], capture_output=True,
                                  text=True, check=True).stdout.strip())
        stats = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
        counts = {}
        for line in stats.splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                counts[key.strip()] = value.strip().rstrip(".")
        # "Free" alone understates what is reclaimable; inactive + speculative are too.
        pages = sum(int(counts.get(k, 0)) for k in
                    ("Pages free", "Pages inactive", "Pages speculative") if counts.get(k, "").isdigit())
        free = pages * page / 1024**3
    except (OSError, subprocess.CalledProcessError, ValueError):
        pass
    try:
        usage = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True,
                               text=True, check=True).stdout
        # e.g. "total = 34816.00M  used = 33513.25M  free = 1302.75M  (encrypted)"
        fields = {k: v for k, v in (p.split(" = ") for p in
                                    (s.strip() for s in usage.replace("(encrypted)", "").split("  "))
                                    if " = " in p)}
        total = float(fields["total"].rstrip("M"))
        used = float(fields["used"].rstrip("M"))
        swap_used = used / total if total > 0 else 0.0
    except (OSError, subprocess.CalledProcessError, ValueError, KeyError):
        pass
    return free, swap_used


def _linux_memory_gb() -> tuple[float | None, float | None]:
    """Free RAM and used-swap fraction on Linux, or ``(None, None)`` if unreadable."""
    try:
        info = {}
        with open("/proc/meminfo") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                info[key] = float(rest.strip().split()[0]) / 1024**2  # kB -> GiB
    except (OSError, ValueError, IndexError):
        return None, None
    free = info.get("MemAvailable")
    total_swap, free_swap = info.get("SwapTotal", 0.0), info.get("SwapFree", 0.0)
    swap_used = (total_swap - free_swap) / total_swap if total_swap > 0 else 0.0
    return free, swap_used


def local_memory_state() -> tuple[float | None, float | None]:
    """Best-effort ``(free_gb, swap_used_fraction)`` for this machine."""
    system = platform.system()
    if system == "Darwin":
        return _macos_memory_gb()
    if system == "Linux":
        return _linux_memory_gb()
    return None, None


def require_local_capacity(
    n_items: int,
    what: str,
    stage: str,
    min_free_gb: float = DEFAULT_MIN_FREE_GB,
) -> None:
    """Stop a large local workload that would likely swap this machine to death."""
    if n_items < LARGE_WORKLOAD_ITEMS:
        return
    if os.environ.get(ALLOW_ENV):
        logger.warning(f"{ALLOW_ENV} set - running {what} locally on {n_items:,} items without a capacity check")
        return

    free_gb, swap_used = local_memory_state()
    if free_gb is None:
        return  # unknown platform: never block the pipeline on a check we cannot make

    reasons = []
    if free_gb < min_free_gb:
        reasons.append(f"only {free_gb:.1f} GB memory free (need ~{min_free_gb:.0f} GB)")
    if swap_used is not None and swap_used > MAX_SWAP_USED_FRACTION:
        reasons.append(f"swap already {swap_used:.0%} full")
    if not reasons:
        logger.info(f"local capacity OK for {what} ({n_items:,} items, {free_gb:.1f} GB free)")
        return

    raise RuntimeError(
        f"Refusing to run {what} on {n_items:,} items locally: {'; '.join(reasons)}.\n"
        f"This workload has swapped a 24 GB machine to death before, with no error and no "
        f"crash report - so it fails fast here instead.\n"
        f"Run it on the cloud box:  make jobs-repro STAGES={stage}\n"
        f"Or override deliberately: {ALLOW_ENV}=1 <your command>"
    )
