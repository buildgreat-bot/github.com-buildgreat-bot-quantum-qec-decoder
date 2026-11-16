"""CI gate: Ensures no uncompleted owner TODO markers exist on the main branch."""
import subprocess
import sys


def main():
    target = "TODO(" + "owner)"
    res = subprocess.run(["git", "grep", "-nF", target], capture_output=True, text=True)
    if res.returncode == 0 and res.stdout.strip():
        # Filter out this check script or workflow definition if present
        lines = [
            line for line in res.stdout.splitlines()
            if "check_no_todo.py" not in line and ".github/workflows/ci.yml" not in line
        ]
        if lines:
            print(f"ERROR: Found uncompleted {target} markers on main branch:\n")
            print("\n".join(lines))
            sys.exit(1)
    print(f"PASS: No uncompleted {target} markers found.")


if __name__ == "__main__":
    main()
