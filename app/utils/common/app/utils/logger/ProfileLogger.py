import re
import uuid
import pandas as pd
from datetime import datetime
from sqlalchemy.orm import Session
from app.utils.SQL.models.progress.orm.ProfileArchive import ProfileArchive  # <- import the model above

class ProfileLogger:
    """
    Logger for profiling memory usage and storing logs in the database.
    """
    def __init__(self, task_group, device, session: Session, task_name: str, profile_type: str = "memProfile"):
        self.device = device
        self.profile_type = profile_type
        self.task_group = task_group
        self.task_name = task_name
        self.time = datetime.now()
        self.task_uuid = uuid.uuid4()
        self.session = session

    def log_stream_to_db(self, mem_stream):
        entries = self._parse_log(mem_stream)

        for entry in entries:
            log_entry = ProfileArchive(
                task_uuid=self.task_uuid,
                task_name=self.task_name,
                task_group=self.task_group,
                device=self.device,
                profile_type=self.profile_type,
                time=self.time,
                line_number=entry["line_number"],
                mem_usage=entry["mem_usage"],
                increment=entry["increment"],
                occurrences=entry["occurrences"],
                line_contents=entry["line_contents"]
            )
            self.session.add(log_entry)

        self.session.commit()

    def _parse_log(self, mem_stream):
        parsed = []
        columns = ['line_number', 'mem_usage', 'increment', 'occurrences', 'line_contents']
        mem_stream.seek(0)

        for line in mem_stream.read().splitlines():
            if re.match(r'\s*\d+\s+\d+\.\d+ MiB\s+\d+\.\d+ MiB\s+\d+', line):
                parts = re.split(r'\s{2,}', line.strip())
                if len(parts) >= 5:
                    try:
                        parsed.append({
                            'line_number': int(parts[0]),
                            'mem_usage': float(parts[1].split()[0]),
                            'increment': float(parts[2].split()[0]),
                            'occurrences': int(parts[3]),
                            'line_contents': ' '.join(parts[4:])
                        })
                    except Exception:
                        continue
        return parsed
