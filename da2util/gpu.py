import re, subprocess

def cuda_version() -> str:
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"nvcc failed (code {result.returncode}). "
                f"stderr: {result.stderr.strip()}"
            )
    except FileNotFoundError as e:
        print("cuda_version: nvcc command not found")
    else:    
        # Look for the first occurrence of ``release X.Y`` where X and Y are numbers.
        match = re.search(r"release\s+(\d+)\.(\d+)", result.stdout, re.IGNORECASE)
        if not match:
            raise ValueError("cuda_version: Could not locate CUDA version in nvcc output.")
        major, minor = match.groups()
        return f"{major}{minor}"