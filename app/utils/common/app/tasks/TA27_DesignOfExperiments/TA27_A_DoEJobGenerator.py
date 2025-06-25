import uuid
import yaml
import logging
import ast
from copy import deepcopy
import pandas as pd
from typing import List

from app.utils.common.app.utils.dataModels.Jobs.DoEJob import (
    DoEJob, DOE_config, PrimaryData, SegmentationCfg,
    SecondaryData, PreProcessingCfg, ModelingCfg
)


class TA27_A_DoEJobGenerator:
    @staticmethod
    def generate(df: pd.DataFrame, template_path: str) -> List[DoEJob]:
        def _parse_value(val):
            return val





        # ── fields that the schema says must be lists ────────────────────────────────
        LIST_FIELDS = {
            "sourceNo", "woodType", "family", "genus", "species", "view",
            "lens", "maxShots", "noShotsRange",
            "filterNo", "secondaryDataBins", "preProcessingNo", "metricModelNo",
        }

        def inject(obj):
            """Recursively expand placeholders in the YAML template using the current
            DataFrame row and normalise “None”-like artefacts to [] where a list is
            required."""
            
            def normalise(key: str, val):
                """Turn None / 'None' / stringified or real lists containing None
                into clean Python objects. Ensures list-fields always return a list."""
                
                match val:
                    # ── 1. Null-like scalars ─────────────────────────────────────────
                    case None | float() if isinstance(val, float) and pd.isna(val):
                        return [] if key in LIST_FIELDS else None
                    
                    case "None":
                        return [] if key in LIST_FIELDS else None
                    
                    # ── 2. String that *looks* like a list → try to parse ────────────
                    case str() as s if s.strip().startswith("["):
                        try:
                            val = ast.literal_eval(s)
                        except Exception:
                            logging.warning(f"⛔ Could not parse list for key '{key}': {s!r}")
                            return [] if key in LIST_FIELDS else None
                        # fall-through to list handler below

                    # ── 3. Real list (parsed or already native) ──────────────────────
                    case list() as lst:
                        cleaned = [
                            None if x in (None, "None", "[None]") else x
                            for x in lst
                        ]
                        cleaned = [x for x in cleaned if x is not None]
                        return cleaned  # may be empty [], and that's fine

                    # ── 4. Anything else ────────────────────────────────────────────
                    case _:
                        # Ensure list-fields still get wrapped into a list
                        return [val] if key in LIST_FIELDS and not isinstance(val, list) else val

            # ── recursive descent over the template structure ────────────────────────
            match obj:
                case dict():
                    return {k: inject(v) for k, v in obj.items()}
                
                case list():
                    return [inject(x) for x in obj]
                
                case str() as s if s.startswith("{") and s.endswith("}"):
                    key = s[1:-1]
                    
                    if key not in row:
                        logging.debug3(f"⚠️ Placeholder '{key}' not found in row {idx}")
                        return [] if key in LIST_FIELDS else None
                    
                    return normalise(key, row[key])
                
                case _:
                    return obj  # literal value in the template, untouched








        logging.debug3(f"📄 Loading job template from: {template_path}")
        with open(template_path, 'r', encoding='utf-8') as f:
            template = yaml.safe_load(f)

        jobs: List[DoEJob] = []
        logging.debug3(f"🧪 Starting job generation for {len(df)} rows...")

        for idx, (_, row) in enumerate(df.iterrows()):
            try:
                injected = inject(deepcopy(template))

                # Patch missing fields (if segmentation isn't in YAML)
                segmentation_block = injected.get("segmentation", {"filterNo": ["GS"]})

                doe_cfg = DOE_config(
                    primary_data=PrimaryData(**injected["primary_data"]),
                    segmentation=SegmentationCfg(**segmentation_block),
                    secondary_data=SecondaryData(**injected["secondary_data"]),
                    preprocessing=PreProcessingCfg(**injected.get("preprocessing", {})),
                    modeling=ModelingCfg(**injected.get("modeling", {}))
                )

                job = DoEJob(
                    job_uuid=injected.get("DoE_UUID"),
                    doe_config=doe_cfg
                )


                jobs.append(job)

            except Exception as e:
                logging.warning(f"⛔ Failed to generate job from row {idx}: {e}", exc_info=True)

        logging.info(f"✅ Generated {len(jobs)} DoEJob objects using template '{template_path}'")

        if not jobs:
            logging.warning("⚠️ No jobs were generated. Check DoE expansion or template placeholders.")

        return jobs
