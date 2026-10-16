"""Default clock service; callers may inject a clock with the same methods."""

from pathlib import Path
import time


class SystemClock:
    """Keep wall, monotonic and boot identities explicit at runtime boundaries."""

    def time(self):
        return time.time()

    def monotonic(self):
        return time.monotonic()

    def monotonic_ns(self):
        return time.monotonic_ns()

    def boot_id(self):
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
