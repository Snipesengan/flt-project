class ChessboardNotFoundError(Exception):
    """No chessboard could be found in searched image"""


class InconsistentCalibrationSizeError(Exception):
    """Performing calibration with varying calibration image size"""


class CalibrationFolderExistsError(Exception):
    """Calibration output folder already exists when trying to save"""
