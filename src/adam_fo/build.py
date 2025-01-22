"""Build script for find_orb dependencies."""
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


class BuildOutput:
    """Helper class for formatted build output."""
    
    # ANSI color codes
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

    def __init__(self):
        self.indent_level = 0
    
    def header(self, msg: str) -> None:
        """Print a header message."""
        print(f"\n{self.HEADER}{self.BOLD}{msg}{self.ENDC}")
    
    def info(self, msg: str) -> None:
        """Print an info message."""
        indent = "  " * self.indent_level
        print(f"{indent}{self.BLUE}→{self.ENDC} {msg}")
    
    def success(self, msg: str) -> None:
        """Print a success message."""
        indent = "  " * self.indent_level
        print(f"{indent}{self.GREEN}✓{self.ENDC} {msg}")
    
    def warning(self, msg: str) -> None:
        """Print a warning message."""
        indent = "  " * self.indent_level
        print(f"{indent}{self.YELLOW}!{self.ENDC} {msg}", file=sys.stderr)
    
    def error(self, msg: str) -> None:
        """Print an error message."""
        indent = "  " * self.indent_level
        print(f"{indent}{self.RED}✗{self.ENDC} {msg}", file=sys.stderr)


def run_command(cmd: str, cwd: Optional[Path] = None, output: BuildOutput = BuildOutput()) -> bool:
    """Run a command and handle its output."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            output.error(f"Command failed: {cmd}")
            if result.stderr:
                output.error(f"Error: {result.stderr}")
            return False
        return True
    except Exception as e:
        output.error(f"Failed to run command: {cmd}")
        output.error(f"Error: {str(e)}")
        return False


def main():
    """Build and install find_orb."""
    output = BuildOutput()
    
    # Get the installation directory
    xdg_data_home = os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
    build_dir = Path(xdg_data_home) / "adam_fo"
    
    # Get the script directory
    script_dir = Path(__file__).parent
    build_script = script_dir / "build_fo.sh"
    
    if not build_script.exists():
        output.error(f"Build script not found at {build_script}")
        sys.exit(1)
    
    output.header("Building find_orb")
    output.info(f"Installation directory: {build_dir}")
    
    # Make the build script executable
    try:
        build_script.chmod(0o755)
    except Exception as e:
        output.error(f"Failed to make build script executable: {e}")
        sys.exit(1)
    
    output.info("Cloning repositories...")
    output.indent_level += 1
    
    # Execute the build script
    result = subprocess.run(
        ["bash", str(build_script)],
        text=True,
        capture_output=True,
    )
    
    # Process the output line by line to show progress
    for line in result.stdout.splitlines():
        if "Cloning" in line:
            output.info(line)
        elif "Building" in line:
            output.indent_level = 1
            output.info(line)
        elif "Installing" in line:
            output.header(line)
        elif "Error" in line or "error" in line.lower():
            output.error(line)
            if result.returncode != 0:
                sys.exit(1)
        elif "Warning" in line or "warning" in line.lower():
            output.warning(line)
        elif "complete" in line.lower():
            output.success(line)
    
    if result.returncode != 0:
        output.error("Build failed!")
        if result.stderr:
            output.error(result.stderr)
        sys.exit(1)
    
    output.indent_level = 0
    output.success("find_orb installation complete!")
    output.info(f"Build files are located in: {build_dir}")
    output.info("You can now use find_orb in your Python code")


if __name__ == "__main__":
    main() 