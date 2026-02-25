import shutil
from dataclasses import dataclass
from datetime import datetime

from anyio import run_process
from loguru import logger

from exo.shared.types.profiling import MemoryUsage, SystemPerformanceProfile
from exo.utils.info_gatherer.macmon import MacmonMetrics

# Conversion constant
MIB_TO_BYTES = 1024 * 1024


@dataclass
class LinuxGpuMetrics:
    """Clean dataclass for Linux GPU metrics from nvidia-smi or rocm-smi."""

    gpu_utilization: float  # percentage 0-100
    gpu_power_watts: float
    gpu_temp_celsius: float
    vram_total_bytes: int
    vram_free_bytes: int
    timestamp: str


def _safe_parse_float(value: str, default: float = 0.0) -> float:
    """Safely parse a float from smi output, handling [N/A] and other edge cases."""
    value = value.strip()
    if not value or value.startswith("[") or value.lower() in ("n/a", "not supported"):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _safe_parse_int_from_mib(value: str, default: int = 0) -> int:
    """Safely parse MiB value to bytes from smi output."""
    parsed = _safe_parse_float(value, float(default) / MIB_TO_BYTES)
    return int(parsed * MIB_TO_BYTES)


async def _get_nvidia_gpu_metrics() -> LinuxGpuMetrics | None:
    """Collect GPU metrics via nvidia-smi. Returns None if no NVIDIA GPU found."""
    if not shutil.which("nvidia-smi"):
        return None

    timestamp = str(int(datetime.now().timestamp() * 1000))
    try:
        result = await run_process(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,power.draw,temperature.gpu,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ]
        )
        output = result.stdout.decode().strip()
        if not output:
            return None

        lines = output.split("\n")
        total_vram = 0
        total_free = 0
        total_power = 0.0
        max_temp = 0.0
        total_util = 0.0
        gpu_count = 0

        for line in lines:
            parts = line.split(",")
            if len(parts) >= 5:
                total_util += _safe_parse_float(parts[0])
                total_power += _safe_parse_float(parts[1])
                max_temp = max(max_temp, _safe_parse_float(parts[2]))
                total_vram += _safe_parse_int_from_mib(parts[3])
                total_free += _safe_parse_int_from_mib(parts[4])
                gpu_count += 1

        if gpu_count == 0:
            return None

        return LinuxGpuMetrics(
            gpu_utilization=total_util / gpu_count,
            gpu_power_watts=total_power,
            gpu_temp_celsius=max_temp,
            vram_total_bytes=total_vram,
            vram_free_bytes=total_free,
            timestamp=timestamp,
        )
    except Exception as e:
        logger.warning(f"Failed to query nvidia-smi: {e}")
        return None


async def _get_amd_gpu_metrics() -> LinuxGpuMetrics | None:
    """Collect GPU metrics via rocm-smi. Returns None if no AMD GPU found."""
    if not shutil.which("rocm-smi"):
        return None

    timestamp = str(int(datetime.now().timestamp() * 1000))
    try:
        # --showuse: utilization, --showpower: power, --showtemp: temp,
        # --showmeminfo vram: vram total/used, --csv: parseable output
        result = await run_process(
            [
                "rocm-smi",
                "--showuse",
                "--showpower",
                "--showtemp",
                "--showmeminfo",
                "vram",
                "--csv",
            ]
        )
        output = result.stdout.decode().strip()
        if not output:
            return None

        # rocm-smi CSV format varies by version; parse key=value style lines
        gpu_util = 0.0
        gpu_power = 0.0
        gpu_temp = 0.0
        vram_total = 0
        vram_used = 0
        gpu_count = 0

        lines = output.split("\n")
        for line in lines:
            line = line.strip()
            # Skip headers and empty lines
            if not line or line.startswith("device") or line.startswith("GPU"):
                # Try CSV header detection
                if "," in line:
                    continue

            # rocm-smi --csv outputs lines like:
            # card0, 45, 120.5, 67, 24576, 8192
            # Column order with our flags: gpu_use%, power_w, temp_c, vram_total_mb, vram_used_mb
            if "," in line:
                parts = [p.strip() for p in line.split(",")]
                # Skip pure header rows
                if parts[0].lower() in ("device", "gpu", "card"):
                    continue
                if len(parts) >= 5:
                    gpu_util += _safe_parse_float(parts[1])
                    gpu_power += _safe_parse_float(parts[2])
                    gpu_temp = max(gpu_temp, _safe_parse_float(parts[3]))
                    # rocm-smi reports VRAM in bytes with --showmeminfo
                    vram_total += int(_safe_parse_float(parts[4]))
                    vram_used += int(_safe_parse_float(parts[5]) if len(parts) > 5 else 0)
                    gpu_count += 1

        if gpu_count == 0:
            # Fallback: try the non-CSV path to at least confirm AMD GPU exists
            check = await run_process(["rocm-smi", "--showid"])
            if b"GPU" not in check.stdout:
                return None
            # Return minimal metrics if we can't parse properly
            return LinuxGpuMetrics(
                gpu_utilization=0.0,
                gpu_power_watts=0.0,
                gpu_temp_celsius=0.0,
                vram_total_bytes=0,
                vram_free_bytes=0,
                timestamp=timestamp,
            )

        vram_free = max(0, vram_total - vram_used)
        return LinuxGpuMetrics(
            gpu_utilization=gpu_util / gpu_count,
            gpu_power_watts=gpu_power,
            gpu_temp_celsius=gpu_temp,
            vram_total_bytes=vram_total,
            vram_free_bytes=vram_free,
            timestamp=timestamp,
        )

    except Exception as e:
        logger.warning(f"Failed to query rocm-smi: {e}")
        return None


async def get_linux_gpu_metrics() -> LinuxGpuMetrics:
    """
    Collects GPU metrics for Linux.
    Tries NVIDIA (nvidia-smi) first, then AMD (rocm-smi).
    Returns zeroed metrics if neither is available.
    """
    timestamp = str(int(datetime.now().timestamp() * 1000))

    nvidia = await _get_nvidia_gpu_metrics()
    if nvidia is not None:
        return nvidia

    amd = await _get_amd_gpu_metrics()
    if amd is not None:
        return amd

    logger.debug("No GPU found via nvidia-smi or rocm-smi, returning zeroed metrics")
    return LinuxGpuMetrics(
        gpu_utilization=0.0,
        gpu_power_watts=0.0,
        gpu_temp_celsius=0.0,
        vram_total_bytes=0,
        vram_free_bytes=0,
        timestamp=timestamp,
    )


async def get_linux_metrics_async() -> MacmonMetrics:
    """
    Collects metrics for Linux (NVIDIA via nvidia-smi, AMD via rocm-smi).

    Returns a MacmonMetrics object for compatibility with the GatheredInfo interface.
    MacmonMetrics wraps SystemPerformanceProfile + MemoryUsage which are generic types;
    Mac-specific fields (pcpu_usage, ecpu_usage) are set to 0 on Linux.
    Note: Uses VRAM as memory metrics for Linux GPU systems.
    """
    gpu_metrics = await get_linux_gpu_metrics()

    # Convert GPU utilization from percentage (0-100) to decimal (0-1)
    gpu_util_decimal = (
        gpu_metrics.gpu_utilization / 100.0 if gpu_metrics.gpu_utilization > 0 else 0.0
    )

    return MacmonMetrics(
        system_profile=SystemPerformanceProfile(
            gpu_usage=gpu_util_decimal,
            temp=gpu_metrics.gpu_temp_celsius,
            sys_power=gpu_metrics.gpu_power_watts,
            pcpu_usage=0.0,
            ecpu_usage=0.0,
        ),
        memory=MemoryUsage.from_bytes(
            ram_total=gpu_metrics.vram_total_bytes,
            ram_available=gpu_metrics.vram_free_bytes,
            swap_total=0,
            swap_available=0,
        ),
    )
