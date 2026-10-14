"""Device-level nvidia-smi sampling, not misleading torch-only CUDA accounting."""
import importlib.metadata
import json
from pathlib import Path
import resource
import shutil
import subprocess
import threading
import time


class BudgetExceeded(RuntimeError):
    pass


class ResourceProfiler:
    def __init__(self, limit_mib=3584, interval=.1):
        self.limit_mib, self.interval = limit_mib, interval
        self.gpu_available = bool(shutil.which('nvidia-smi'))
        self.peak = None
        self.samples = []
        self.errors = []
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = None

    def _sample(self):
        if not self.gpu_available:
            return None
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used',
                                     '--format=csv,noheader,nounits'],
                                    text=True, capture_output=True, check=True, timeout=2)
            used = max(float(line.strip()) for line in result.stdout.splitlines() if line.strip())
            with self._lock:
                self.peak = max(self.peak or 0, used)
            return used
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            with self._lock:
                if not self.errors:
                    self.errors.append(str(error))
            return None

    def _loop(self):
        while not self._stop.wait(self.interval):
            self._sample()

    def start(self):
        self._sample()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def snapshot(self, stage):
        current = self._sample()
        with self._lock:
            row = {'stage': stage, 'monotonic_seconds': time.monotonic(),
                   'gpu_device_used_mib': current, 'peak_device_used_mib': self.peak,
                   'peak_process_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                   'measurement': 'nvidia-smi device-total sampled' if current is not None else 'GPU telemetry unavailable'}
            self.samples.append(row)
            peak = self.peak
        if peak is not None and peak >= self.limit_mib:
            raise BudgetExceeded(f'Sampled device usage {peak} MiB reached {self.limit_mib} MiB guard. Reduce context/offload or use CPU.')
        return row

    def close(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def report(self):
        return {'gpu_peak_mib': self.peak, 'gpu_available': self.gpu_available,
                'budget_mib': self.limit_mib, 'sampling_interval_seconds': self.interval,
                'note': 'Device-total samples include other processes and can miss transient peaks; not a hard allocation cap.',
                'samples': self.samples, 'telemetry_errors': self.errors}


def environment():
    result = {}
    for package in ['torch', 'llama-cpp-python', 'sentence-transformers', 'titans-torch', 'titans-pytorch']:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    if Path('/proc/meminfo').exists():
        info = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        result['system_ram_total_mib'] = int(info['MemTotal'].split()[0]) / 1024
        result['system_ram_available_mib'] = int(info['MemAvailable'].split()[0]) / 1024
    result['gguf_files_in_repo'] = [str(p) for p in Path('.').glob('*.gguf')]
    return result
