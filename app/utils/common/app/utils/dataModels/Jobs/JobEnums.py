from enum import Enum   

class JobStatus(str, Enum):
    BLOCKED = "blocked"
    READY = "ready"
    DONE = "done"
    IN_PROGRESS = "in_progress"
    FAILED = "failed"



class RelationState(str, Enum):
    FREE        = "free"         # child finished successfully
    IN_PROGRESS = "in_progress"  # child still running
    BLOCKED     = "blocked"      # waiting for something else
    FAILED      = "failed"       # child failed


class JobKind(str, Enum):
    DOE       = "DoE"
    PROVIDER   = "provider"
    SEGMENTER  = "segmenter"
    EXTRACTOR  = "extractor"
    MODELER    = "modeler"
    VALIDATOR  = "validator"
    TRANSFER   = "transfer"