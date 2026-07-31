import platform
import multiprocessing
import psutil
import shutil
import os


def profile_memory() -> dict:
    total, used, free = shutil.disk_usage(os.path.expanduser("~"))
    ram = psutil.virtual_memory()

    return dict(
        memory=dict(
            total_ram_gb=ram.total / (1024**3),
            available_ram_gb=ram.available / (1024**3),
            used_ram_gb=ram.used / (1024**3),
        ),
        disk_space=dict(
            total_disk_gb=total / (1024**3), free_disk_gb=free / (1024**3), used_disk_gb=(total - free) / (1024**3)
        ),
    )


def profile_system(gb_unit: bool = True) -> dict:
    # convert to either GB or MB
    scale = (1024.0**3) if gb_unit else (1024.0**2)
    memory = psutil.virtual_memory()
    return dict(
        platform=dict(
            system=platform.system(),  # e.g., 'Linux'
            cpu_processor=platform.processor(),  # e.g., 'x86_64'
            architecture=list(platform.architecture()),  # e.g., ('64bit', 'ELF')
            num_cores=multiprocessing.cpu_count(),
        ),
        **profile_memory(),
    )
