"""Memory requirements shared by the plugin and the standalone renderer."""

import ctypes
import os
import re


class SingingResourcesError(RuntimeError):
    """Windows does not have enough available memory for this singing step."""


class SingingComputeBusy(RuntimeError):
    """A worker reported CPU or GPU contention."""


def compute_is_busy(message: str) -> bool:
    return bool(re.search(
        r'(?:CPU|GPU|显卡|处理器)(?:计算|算力|负载|利用率|使用率|占用率)?[ ：:=]*(?:已占满|占满|已满载|满载|繁忙|已饱和|100\s*%)'
        r'|\b(?:cpu|gpu)(?: devices?)?[ _:-]+(?:(?:is|are) )?(?:too )?(?:busy|overloaded|saturated)\b'
        r'|\b(?:cuda-capable )?devices? (?:is|are) busy\b', message, re.I))


class MemoryStatus(ctypes.Structure):
    _fields_ = [
        ('length', ctypes.c_uint32),
        ('load', ctypes.c_uint32),
        ('total_phys', ctypes.c_uint64),
        ('avail_phys', ctypes.c_uint64),
        ('total_page', ctypes.c_uint64),
        ('avail_page', ctypes.c_uint64),
        ('total_virtual', ctypes.c_uint64),
        ('avail_virtual', ctypes.c_uint64),
        ('extended_virtual', ctypes.c_uint64),
    ]


def check_full_song_memory(minimum_mib: int = 2048) -> None:
    if os.name != 'nt':
        return
    status = MemoryStatus(length=ctypes.sizeof(MemoryStatus))
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError()
    available_mib = status.avail_phys // 1024**2
    if available_mib < minimum_mib:
        raise SingingResourcesError(f'Windows 可用内存 {available_mib} MiB，当前步骤需要至少 {minimum_mib} MiB')
