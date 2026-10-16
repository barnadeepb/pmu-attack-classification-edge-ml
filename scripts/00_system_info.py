"""
Hardware / software inventory of the machine running the benchmark.

Writes results/runs/<tag>/system_info.json and system_specs.md. The hardware
model name matters: the paper must state exactly which device the latency
numbers came from, so this records the manufacturer, model, BIOS and CPU
details as the OS reports them (no manual typing).
"""

import json
import os
import platform
import subprocess
import sys

import common as C


def ps(cmd):
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=60)
        return out.stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"unavailable ({exc})"


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"unavailable ({exc})"


def main():
    C.ensure_dirs()
    info = C.env_summary()
    if platform.system() == "Windows":
        info["computer_system"] = ps("Get-CimInstance Win32_ComputerSystem | Select-Object Manufacturer,Model,SystemType,TotalPhysicalMemory | ConvertTo-Json -Compress")
        info["bios"] = ps("Get-CimInstance Win32_BIOS | Select-Object Manufacturer,SMBIOSBIOSVersion,ReleaseDate | ConvertTo-Json -Compress")
        info["cpu"] = ps("Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed,L2CacheSize,L3CacheSize | ConvertTo-Json -Compress")
        info["memory_modules"] = ps("Get-CimInstance Win32_PhysicalMemory | Select-Object Capacity,Speed,Manufacturer | ConvertTo-Json -Compress")
        info["os"] = ps("(Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber | ConvertTo-Json -Compress)")
        info["power_plan"] = sh("powercfg /getactivescheme")
        info["disk"] = ps("Get-CimInstance Win32_DiskDrive | Select-Object Model,MediaType,Size | ConvertTo-Json -Compress")
    else:
        info["lscpu"] = sh("lscpu")
        info["dmi_product"] = sh("cat /sys/devices/virtual/dmi/id/product_name /sys/devices/virtual/dmi/id/sys_vendor 2>/dev/null")
        info["os_release"] = sh("cat /etc/os-release")
        info["governor"] = sh("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null")
        info["meminfo"] = sh("head -2 /proc/meminfo")
    info["pip_freeze"] = sh(f'"{sys.executable}" -m pip freeze').splitlines()

    (C.RESULTS_DIR / "system_info.json").write_text(json.dumps(info, indent=2))
    lines = [f"# Benchmark environment (tag: {C.TAG})", "",
             "Collected automatically by scripts/00_system_info.py.", ""]
    for k, v in info.items():
        if k == "pip_freeze":
            continue
        lines.append(f"- **{k}**: {v}")
    lines += ["", "## pip freeze", "", "```", *info.get("pip_freeze", []), "```"]
    (C.RESULTS_DIR / "system_specs.md").write_text("\n".join(lines), encoding="utf8")
    print("\n".join(lines[:30]))
    print(f"\nWrote {C.RESULTS_DIR / 'system_info.json'}")


if __name__ == "__main__":
    main()
