from enum import Enum   

class JobStatus(str, Enum):
    BLOCKED = "BLOCKED"
    READY = "READY"
    DONE = "DONE"
    IN_PROGRESS = "IN_PROGRESS"
    FAILED = "FAILED"



class RelationState(str, Enum):
    FREE        = "FREE"         # child finished successfully
    IN_PROGRESS = "IN_PROGRESS"  # child still running
    BLOCKED     = "BLOCKED"      # waiting for something else
    FAILED      = "FAILED"       # child failed


class JobKind(str, Enum):
    DOE       = "DoE"
    PROVIDER   = "provider"
    SEGMENTER  = "segmenter"
    EXTRACTOR  = "extractor"
    MODELER    = "modeler"
    VALIDATOR  = "validator"
    TRANSFER   = "transfer"