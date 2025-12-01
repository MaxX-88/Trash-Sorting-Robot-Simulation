from enum import Enum, auto

class ArmState(Enum):
    """
    States for Finite State Machine
    """
    IDLE = auto()
    WAIT_FOR_OBJECT = auto()
    PREPARE_PICK = auto()
    PICKING = auto()
    LIFTING = auto()
    RESETTING = auto()
