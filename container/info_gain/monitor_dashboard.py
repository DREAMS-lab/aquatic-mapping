#!/usr/bin/env python3
"""
Trial Monitoring Dashboard
Real-time terminal UI showing trial progress for both planners.

Usage: python3 monitor_dashboard.py [results_dir]
       python3 monitor_dashboard.py ./trial_results
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Try to import rich for nice terminal UI
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich.text import Text
    from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Note: Install 'rich' for a better UI: pip install rich")


FIELDS = ["radial", "x_compress", "y_compress", "x_compress_tilt", "y_compress_tilt"]


def get_completed_trials(results_dir: Path, planner: str) -> dict:
    """Scan directory for completed trials (those with summary.json)."""
    completed = {}
    planner_dir = results_dir / planner

    for field in FIELDS:
        field_dir = planner_dir / field
        completed[field] = []
        if field_dir.exists():
            for trial_dir in sorted(field_dir.glob("trial_*")):
                if (trial_dir / "summary.json").exists():
                    # Extract trial number
                    trial_num = int(trial_dir.name.split("_")[1])
                    completed[field].append(trial_num)

    return completed


def get_status(results_dir: Path, planner: str) -> dict:
    """Read status.json for a planner."""
    status_file = results_dir / planner / "status.json"
    if status_file.exists():
        try:
            with open(status_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {"state": "unknown", "message": "No status file"}


def check_container_running(container_name: str) -> bool:
    """Check if a Docker container is running."""
    import subprocess
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=5
        )
        return container_name in result.stdout.split("\n")
    except Exception:
        return False


def format_duration(start_time_str: str) -> str:
    """Format duration since start time."""
    try:
        start = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
        now = datetime.now(start.tzinfo) if start.tzinfo else datetime.now()
        delta = now - start
        hours, remainder = divmod(int(delta.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    except Exception:
        return "--:--:--"


class SimpleDashboard:
    """Fallback simple dashboard without rich."""

    def __init__(self, results_dir: Path, num_trials: int):
        self.results_dir = results_dir
        self.num_trials = num_trials

    def run(self):
        print("\n" + "=" * 60)
        print("  INFO GAIN TRIAL MONITOR")
        print("=" * 60)
        print(f"  Results: {self.results_dir}")
        print(f"  Expected: {self.num_trials} trials x 5 fields x 2 planners")
        print("=" * 60)
        print("\nPress Ctrl+C to exit\n")

        try:
            while True:
                os.system('clear' if os.name == 'posix' else 'cls')
                print("\n" + "=" * 60)
                print(f"  INFO GAIN TRIAL MONITOR - {datetime.now().strftime('%H:%M:%S')}")
                print("=" * 60)

                for planner in ["exact", "pose_aware"]:
                    status = get_status(self.results_dir, planner)
                    completed = get_completed_trials(self.results_dir, planner)
                    total_completed = sum(len(v) for v in completed.values())
                    container_running = check_container_running(f"{planner}_trials")

                    print(f"\n  [{planner.upper()}]")
                    print(f"    Container: {'RUNNING' if container_running else 'STOPPED'}")
                    print(f"    State: {status.get('state', 'unknown')}")
                    print(f"    Message: {status.get('message', '')}")
                    print(f"    Current: Trial {status.get('current_trial', '?')} / {status.get('current_field', '?')}")
                    print(f"    Completed: {total_completed} / {self.num_trials * 5}")
                    print(f"    Fields: ", end="")
                    for field in FIELDS:
                        count = len(completed.get(field, []))
                        print(f"{field[:3]}:{count} ", end="")
                    print()

                print("\n" + "-" * 60)
                time.sleep(3)

        except KeyboardInterrupt:
            print("\n\nMonitor stopped.")


class RichDashboard:
    """Rich terminal UI dashboard."""

    def __init__(self, results_dir: Path, num_trials: int):
        self.results_dir = results_dir
        self.num_trials = num_trials
        self.console = Console()
        self.start_time = datetime.now().isoformat()

    def make_planner_panel(self, planner: str) -> Panel:
        """Create a panel for a single planner."""
        status = get_status(self.results_dir, planner)
        completed = get_completed_trials(self.results_dir, planner)
        total_completed = sum(len(v) for v in completed.values())
        total_expected = self.num_trials * 5
        container_running = check_container_running(f"{planner}_trials")

        # Build content
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Key", style="bold")
        table.add_column("Value")

        # Container status
        container_status = "[green]RUNNING[/green]" if container_running else "[red]STOPPED[/red]"
        table.add_row("Container", container_status)

        # Current state
        state = status.get("state", "unknown")
        state_colors = {
            "running": "green",
            "completed": "blue",
            "error": "red",
            "skipped": "yellow",
            "starting": "cyan",
            "finished": "magenta"
        }
        state_color = state_colors.get(state, "white")
        table.add_row("State", f"[{state_color}]{state}[/{state_color}]")

        # Message
        table.add_row("Status", status.get("message", "")[:50])

        # Current trial/field
        current = f"Trial {status.get('current_trial', '?')} / {status.get('current_field', '?')}"
        table.add_row("Current", current)

        # Progress bar
        progress_pct = (total_completed / total_expected * 100) if total_expected > 0 else 0
        progress_bar = self._make_progress_bar(progress_pct)
        table.add_row("Progress", f"{progress_bar} {total_completed}/{total_expected}")

        # Field breakdown
        table.add_row("", "")
        field_text = ""
        for field in FIELDS:
            count = len(completed.get(field, []))
            if count == self.num_trials:
                field_text += f"[green]{field[:6]}:{count}[/green] "
            elif count > 0:
                field_text += f"[yellow]{field[:6]}:{count}[/yellow] "
            else:
                field_text += f"[dim]{field[:6]}:{count}[/dim] "
        table.add_row("Fields", field_text)

        # Title color based on state
        title_style = "green" if container_running else "red"
        return Panel(table, title=f"[bold {title_style}]{planner.upper()}[/bold {title_style}]",
                    border_style=title_style)

    def _make_progress_bar(self, pct: float, width: int = 20) -> str:
        """Create a simple progress bar string."""
        filled = int(width * pct / 100)
        empty = width - filled
        bar = "[green]" + "█" * filled + "[/green]" + "[dim]░[/dim]" * empty
        return f"[{bar}] {pct:.0f}%"

    def make_completed_table(self) -> Table:
        """Create a table showing all completed trials."""
        table = Table(title="Completed Trials Matrix", show_header=True, header_style="bold")
        table.add_column("Field", style="cyan")
        table.add_column("Exact", justify="center")
        table.add_column("Pose-Aware", justify="center")

        exact_completed = get_completed_trials(self.results_dir, "exact")
        pose_completed = get_completed_trials(self.results_dir, "pose_aware")

        for field in FIELDS:
            exact_trials = exact_completed.get(field, [])
            pose_trials = pose_completed.get(field, [])

            # Format as trial numbers or checkmarks
            exact_str = self._format_trials(exact_trials)
            pose_str = self._format_trials(pose_trials)

            table.add_row(field, exact_str, pose_str)

        return table

    def _format_trials(self, trials: list) -> str:
        """Format trial list for display."""
        if not trials:
            return "[dim]-[/dim]"
        if len(trials) <= 5:
            return ", ".join(str(t) for t in trials)
        return f"{trials[0]}-{trials[-1]} ({len(trials)})"

    def generate_layout(self) -> Layout:
        """Generate the full dashboard layout."""
        layout = Layout()

        # Header
        elapsed = format_duration(self.start_time)
        header = Panel(
            Text.from_markup(
                f"[bold cyan]INFO GAIN TRIAL MONITOR[/bold cyan]\n"
                f"Results: {self.results_dir}\n"
                f"Elapsed: {elapsed} | Updated: {datetime.now().strftime('%H:%M:%S')}"
            ),
            style="cyan"
        )

        # Planner panels side by side
        exact_panel = self.make_planner_panel("exact")
        pose_panel = self.make_planner_panel("pose_aware")

        # Completed table
        completed_table = self.make_completed_table()

        layout.split(
            Layout(header, name="header", size=5),
            Layout(name="planners", size=12),
            Layout(Panel(completed_table, title="Trial Completion Status"), name="completed")
        )

        layout["planners"].split_row(
            Layout(exact_panel),
            Layout(pose_panel)
        )

        return layout

    def run(self):
        """Run the live dashboard."""
        self.console.print("\n[bold cyan]Starting Trial Monitor...[/bold cyan]")
        self.console.print("Press Ctrl+C to exit\n")

        try:
            with Live(self.generate_layout(), console=self.console, refresh_per_second=0.5) as live:
                while True:
                    time.sleep(2)
                    live.update(self.generate_layout())

                    # Check if both containers stopped
                    exact_running = check_container_running("exact_trials")
                    pose_running = check_container_running("pose_aware_trials")

                    if not exact_running and not pose_running:
                        # Final update
                        live.update(self.generate_layout())
                        time.sleep(1)
                        break

        except KeyboardInterrupt:
            pass

        self.console.print("\n[bold green]Monitor stopped.[/bold green]")

        # Print final summary
        self.print_final_summary()

    def print_final_summary(self):
        """Print final summary of all trials."""
        self.console.print("\n" + "=" * 60)
        self.console.print("[bold]FINAL SUMMARY[/bold]")
        self.console.print("=" * 60)

        for planner in ["exact", "pose_aware"]:
            completed = get_completed_trials(self.results_dir, planner)
            total = sum(len(v) for v in completed.values())
            self.console.print(f"\n[bold]{planner.upper()}[/bold]: {total} trials completed")

            for field in FIELDS:
                count = len(completed.get(field, []))
                status = "[green]DONE[/green]" if count == self.num_trials else f"[yellow]{count}/{self.num_trials}[/yellow]"
                self.console.print(f"  {field}: {status}")


def main():
    # Parse arguments
    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])
    else:
        results_dir = Path("./trial_results")

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        print("Creating it now...")
        results_dir.mkdir(parents=True, exist_ok=True)

    # Get num_trials from status file or default
    num_trials = 10
    monitor_file = results_dir / "monitor" / "status.json"
    if monitor_file.exists():
        try:
            with open(monitor_file) as f:
                data = json.load(f)
                num_trials = data.get("num_trials", 10)
        except Exception:
            pass

    print(f"Monitoring: {results_dir}")
    print(f"Expected trials: {num_trials}")

    # Run appropriate dashboard
    if RICH_AVAILABLE:
        dashboard = RichDashboard(results_dir, num_trials)
    else:
        dashboard = SimpleDashboard(results_dir, num_trials)

    dashboard.run()


if __name__ == "__main__":
    main()
