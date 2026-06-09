import os
import sys
import json
import argparse
import subprocess
import pandas as pd


def ncu_profile_details(
    file: str | None = None,
    module: str | None = None,
    ncu_executable: str | None = None,
    profile_device: int = 0,
    profile_output: str | None = None,
    file_args: list[str] | None = None,
) -> dict:

    if ncu_executable is None:
        ncu_executable = "ncu"
    if profile_output is None:
        profile_output = "out"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(profile_device)

    cmd = [
        ncu_executable,
        "--set", "full",
        "--page", "details",
        "--import-source", "yes",
        "--nvtx",
        "--profile-from-start", "off",
        "--csv",
        "-f",
        "-o", profile_output,
        sys.executable,
        *(["-m", module] if module else [file]),
        *(file_args or []),
    ]

    print(f"file_args: {file_args}")

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as error:
        print(f"NCU profiling failed with return code {error.returncode}")
        print(error.stderr)
        raise
    except FileNotFoundError:
        print(f"NCU executable not found: {ncu_executable}")
        raise


    results_lines = result.stdout.splitlines()
    for start_index in range(len(results_lines)):
        if (results_lines[start_index - 1].startswith("==PROF== Report") and
            results_lines[start_index    ].startswith("\"ID\"")):
            break
    results_lines = [
        eval(line)
        for line in results_lines[start_index:]
    ]

    results_df = pd.DataFrame(
        results_lines[1:],
        columns=results_lines[0],
    )

    results_json = {}
    for _, row in results_df.iterrows():
        metric_name = row["Metric Name"]
        metric_unit = row["Metric Unit"]
        metric_value = row["Metric Value"]
        section_name = row["Section Name"]
        if metric_name == "" or metric_unit == "":
            continue
        key = f"{section_name}/{metric_name}/{metric_unit}"
        assert key not in results_json.keys()
        results_json[key] = eval(metric_value)

    print(json.dumps(results_json, indent=4))
    return results_json


def ncu_profile_source(
    file: str | None = None,
    module: str | None = None,
    ncu_executable: str | None = None,
    profile_device: int = 0,
    profile_output: str | None = None,
    file_args: list[str] | None = None,
) -> pd.DataFrame:

    if ncu_executable is None:
        ncu_executable = "ncu"
    if profile_output is None:
        profile_output = "out"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(profile_device)

    cmd = [
        ncu_executable,
        "--set", "full",
        "--page", "source",
        "--print-source", "sass",
        "--import-source", "yes",
        "--nvtx",
        "--profile-from-start", "off",
        "--csv",
        "-f",
        "-o", profile_output,
        sys.executable,
        *(["-m", module] if module else [file]),
        *(file_args or []),
    ]

    print(f"file_args: {file_args}")

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as error:
        print(f"NCU profiling failed with return code {error.returncode}")
        print(error.stderr)
        raise
    except FileNotFoundError:
        print(f"NCU executable not found: {ncu_executable}")
        raise

    results_lines = result.stdout.splitlines()
    for start_index in range(len(results_lines)):
        if (results_lines[start_index - 1].startswith("\"Kernel Name\"") and
            results_lines[start_index    ].startswith("\"Address\"")):
            break
    results_lines = [
        eval(line)
        for line in results_lines[start_index:]
    ]

    results_df = pd.DataFrame(
        results_lines[1:],
        columns=results_lines[0],
    )

    columns = [
        "Source",
        "Warp Stall Sampling (All Samples)",
        "Address Space",
        "Access Operation",
        "Access Size",
        "L2 Theoretical Sectors Global Excessive",
    ]
    results_df = results_df.loc[:, columns]
    print(results_df.to_markdown(index=False))
    return results_df


def nsys_profile(
    file: str | None = None,
    module: str | None = None,
    nsys_executable: str | None = None,
    profile_device: int = 0,
    profile_output: str | None = None,
    file_args: list[str] | None = None,
) -> None:

    if nsys_executable is None:
        nsys_executable = "nsys"
    if profile_output is None:
        profile_output = "out"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(profile_device)

    cmd = [
        nsys_executable,
        "profile",
        "--trace", "cuda,nvtx,osrt,cudnn,cublas",
        "--capture-range", "cudaProfilerApi",
        "--capture-range-end", "stop",
        "-f", "true",
        "-o", profile_output,
        sys.executable,
        *(["-m", module] if module else [file]),
        *(file_args or []),
    ]

    print(f"file_args: {file_args}")

    try:
        subprocess.run(
            cmd,
            env=env,
            # capture_output=True,
            # text=True,
            check=True,
        )
    except subprocess.CalledProcessError as error:
        print(f"NSYS profiling failed with return code {error.returncode}")
        # print(error.stderr)
        raise
    except FileNotFoundError:
        print(f"NSYS executable not found: {nsys_executable}")
        raise

    print(f"Profile saved to {profile_output}.nsys-rep")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile CUDA kernels using NCU or NSYS")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--file", help="Python file to profile")
    target.add_argument("--module", help="Python module to profile")
    parser.add_argument(
        "--mode",
        choices=["details", "source", "nsys"],
        default="details",
        help="Profiling mode (default: details)"
    )
    parser.add_argument(
        "--ncu-executable",
        default=None,
        help="Path to ncu executable (default: ncu)"
    )
    parser.add_argument(
        "--nsys-executable",
        default=None,
        help="Path to nsys executable (default: nsys)"
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device to profile (default: 0)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file name (default: out)"
    )

    args, file_args = parser.parse_known_args()
    file_args = [a for a in file_args if a != "--"]

    if args.mode == "details":
        ncu_profile_details(
            file=args.file,
            module=args.module,
            ncu_executable=args.ncu_executable,
            profile_device=args.device,
            profile_output=args.output,
            file_args=file_args,
        )
    elif args.mode == "source":
        ncu_profile_source(
            file=args.file,
            module=args.module,
            ncu_executable=args.ncu_executable,
            profile_device=args.device,
            profile_output=args.output,
            file_args=file_args,
        )
    elif args.mode == "nsys":
        nsys_profile(
            file=args.file,
            module=args.module,
            nsys_executable=args.nsys_executable,
            profile_device=args.device,
            profile_output=args.output,
            file_args=file_args,
        )
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
