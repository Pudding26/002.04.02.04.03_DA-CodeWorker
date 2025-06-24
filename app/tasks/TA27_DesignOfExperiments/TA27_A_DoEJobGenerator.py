import uuid
import yaml
import logging
from copy import deepcopy
import pandas as pd
from typing import List, Dict








logger = logging.getLogger(__name__)

class TA27_A_DoEJobGenerator:
    @staticmethod
    def generate(df: pd.DataFrame, template_path: str) -> List[Dict]:
        logger.debug3(f"📄 Loading job template from: {template_path}")
        with open(template_path, 'r', encoding='utf-8') as f:
            template = yaml.safe_load(f)

        jobs = []
        logger.debug3(f"🧪 Starting job generation for {len(df)} rows...")

        for idx, (_, row) in enumerate(df.iterrows()):
            def inject(obj):
                if isinstance(obj, dict):
                    return {k: inject(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [inject(x) for x in obj]
                elif isinstance(obj, str) and obj.startswith("{") and obj.endswith("}"):
                    key = obj[1:-1]
                    if key not in row:
                        logger.debug3(f"⚠️  Placeholder '{key}' not found in row {idx}")
                    return row.get(key, obj)
                return obj

            job_data = inject(deepcopy(template))
            job_uuid = str(uuid.uuid4())[:8]
            jobs.append({
                "job_uuid": job_uuid,
                "job_data": job_data
            })

        logger.info(f"✅ Generated {len(jobs)} jobs using template '{template_path}'")

        if not jobs:
            logger.warning("⚠️ No jobs were generated. Check DoE expansion or template placeholders.")

        return jobs
