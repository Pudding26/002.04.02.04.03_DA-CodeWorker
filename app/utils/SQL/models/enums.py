from enum import Enum



class woodTypeEnum(str, Enum):
    hardwood = "Hardwood"
    softwood = "Softwood"



class sampleID_statusEnum(str, Enum):
    transfered = "transfered"
    in_transfer = "in_transfer"
    todo = "todo"
    failed = "failed"