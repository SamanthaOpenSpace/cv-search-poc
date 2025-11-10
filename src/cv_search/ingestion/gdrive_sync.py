# cv_search/ingestion/gdrive_sync.py
from __future__ import annotations
import os
import shutil
import subprocess
import click
from typing import List
from cv_search.config.settings import Settings

class GDriveSyncer:
    """
    Handles the execution of rclone to sync files from Google Drive.
    """
    def __init__(self, settings: Settings):
        self.settings = settings
        self.rclone_bin = shutil.which("rclone")

    def _check_dependencies(self):
        """
        Verifies that rclone is installed and the config is accessible.
        """
        if not self.rclone_bin:
            raise FileNotFoundError(
                "rclone binary not found in your system's PATH. "
                "Please install rclone to use this feature."
            )

        if self.settings.gdrive_rclone_config_path and \
                not self.settings.gdrive_rclone_config_path.exists():
            raise FileNotFoundError(
                f"Specified rclone config not found at: "
                f"{self.settings.gdrive_rclone_config_path}"
            )

    def _build_command(self) -> List[str]:
        """
        Builds the rclone command array from settings.
        """
        local_path = self.settings.gdrive_local_dest_dir
        remote_path = (
            f"{self.settings.gdrive_remote_name}:"
            f"{self.settings.gdrive_source_dir}"
        )

        # Ensure the local destination directory exists
        os.makedirs(local_path, exist_ok=True)

        cmd = [
            self.rclone_bin,
            "sync",          # Use "sync" to mirror the source
            "--verbose",     # Show files being transferred
            remote_path,     # Source
            str(local_path)  # Destination
        ]

        # Add the explicit config path ONLY if it's set in settings
        if self.settings.gdrive_rclone_config_path:
            cmd.insert(1, "--config")
            cmd.insert(2, str(self.settings.gdrive_rclone_config_path))

        return cmd

    def sync_files(self):
        """
        Checks dependencies and executes the rclone sync command.

        The subprocess output (stdout/stderr) is streamed directly
        to the console.

        Raises:
            FileNotFoundError: If rclone binary or config is missing.
            subprocess.CalledProcessError: If rclone command fails.
        """
        self._check_dependencies()
        cmd = self._build_command()

        click.secho(f"Source:      {cmd[-2]}", fg="cyan")
        click.secho(f"Destination: {cmd[-1]}", fg="cyan")
        click.secho(f"Executing:   {' '.join(cmd)}", fg="yellow")
        click.echo("--- rclone output ---")

        # Use subprocess.run without output capture to stream live.
        # check=True will raise CalledProcessError if rclone fails.
        try:
            subprocess.run(cmd, check=True)
        finally:
            click.echo("--- end rclone output ---")