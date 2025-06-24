import os
import subprocess
import logging
import re
from pathlib import Path
from dotenv import load_dotenv
import shlex

class RsyncUtils:
    def __init__(self):
        load_dotenv()
        self.inbound = os.getenv("INBOUND_DIR")
        self.outbound = os.getenv("OUTBOUND_DIR")
        self.peer_host = os.getenv("PEER_HOST")
        self.peer_ssh_port = os.getenv("PEER_SSH_PORT")

        self.peer_inbound = os.getenv("PEER_INBOUND")
        self.peer_outbound = os.getenv("PEER_OUTBOUND")


        if not all([self.inbound, self.outbound, self.peer_host, self.peer_ssh_port, self.peer_inbound, self.peer_outbound]):
            raise EnvironmentError("Missing required environment variables for RsyncUtils")

    def push_file_to_peer(self, file_path: Path, on_progress=None):
        rel_path = file_path.relative_to(self.outbound)
        remote_target_dir = Path(self.peer_inbound) / rel_path.parent
        remote_path = f"{self.peer_host}:{shlex.quote(str(remote_target_dir))}/"

        if not file_path.exists():
            logging.warning(f"❌ File not found: {file_path}")
            return

        logging.info(f"📦 Local file: {file_path}")
        logging.info(f"🎯 Remote destination: {remote_target_dir} on {self.peer_host}")

        cmd = [
            "rsync", "-avz",
            "-e", f"ssh -p {self.peer_ssh_port}",
            "--progress", "--info=progress2",
            str(file_path), remote_path
        ]
        logging.info(f"🚀 Executing rsync: {' '.join(shlex.quote(c) for c in cmd)}")
        self._run_rsync_with_progress(cmd, on_progress)

    def pull_file_from_peer(self, file_path: Path, on_progress=None):
        rel_path = file_path.relative_to(self.inbound)
        remote_source_path = f"{self.peer_host}:{shlex.quote(str(Path(self.peer_outbound) / rel_path))}"

        logging.info(f"📦 Remote file: {remote_source_path}")
        logging.info(f"📥 Local destination: {file_path.parent}/")

        cmd = [
            "rsync", "-avz",
            "-e", f"ssh -p {self.peer_ssh_port}",
            "--progress", "--info=progress2",
            remote_source_path, str(file_path.parent) + "/"
        ]
        logging.info(f"🚀 Executing rsync: {' '.join(shlex.quote(c) for c in cmd)}")
        self._run_rsync_with_progress(cmd, on_progress)


    def _run_rsync_with_progress(self, cmd, on_progress):
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        step = 0.1
        tresh = 0
        for line in process.stdout:
            logging.debug1(line.strip())
            if on_progress and "%" in line:
                percent = self._parse_progress_percent(line)
                if percent is not None:
                    on_progress(percent / 100.0)
                    if percent >= tresh:
                        tresh += step
                        logging.debug2("✅ Transfer progress: {:.2f}%".format(percent))
        process.wait()
        if process.returncode != 0:
            logging.error(f"❌ Rsync failed with exit code {process.returncode}")

    def _parse_progress_percent(self, line):
        match = re.search(r"(\d+)%", line)
        if match:
            return int(match.group(1))
        return None

    def cleanup_file_and_marker(self, file_path: Path):
        marker_path = file_path.with_suffix(file_path.suffix + ".synced")
        try:
            if file_path.exists():
                file_path.unlink()
            if marker_path.exists():
                marker_path.unlink()
            logging.info(f"🧹 Cleaned up: {file_path.name} and its marker")
        except Exception as e:
            logging.warning(f"⚠️ Cleanup failed for {file_path.name}: {e}")

