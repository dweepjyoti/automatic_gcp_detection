class LeptonError(ChildProcessError):
    """
    This error is raised when Lepton which is called using subprocess
    throws an error due to certain runtime conditions.
    """
    pass

class CameraModelNotRecognisedError(Exception):
    """
    This error is raised when the camera model of an image detected does not
    match any model in our internal database. Camera model is used to determine
    the sensor width and height which is critical to image overlap calculation.
    """
    pass


class NoExifFoundError(ValueError):
    """
    This error is raised when there are no EXIF metadata found in an image.
    Real world examples of images with no EXIF metadata were found when using
    Caesium compression software which by default removed EXIF metadata.
    """
    pass


class PositionNotFoundError(ValueError):
    """
    This error is raised when the position of a point cannot be found due to
    missing data. For instance if the position is provided in UTM coordinates
    but the zone number or any other information is missing due to various
    reasons.
    """
    pass


class NoFilesFoundError(Exception):
    """
    When looking for files to operate on, but gets no files at a location
    """
    pass


class ChecksumMismatchError(Exception):
    """
    When making a file transfer the total memory transferred is not same as
    what it is supposed to be
    """


class InvalidIpError(ValueError):
    """
    When the IP to be accessed by the ssh class object is invalid, this is
    raised
    """
    pass


class AuthenticationFailure(ConnectionError):
    """
    When the log in credentials of the emlid are wrong, this error is
    encountered
    """
    pass


class NoValidConnectionsError(ConnectionError):
    """
    When the connections to the Emlid are wrong, this error is raised.
    """
    pass


class FileTransferError(IOError):
    """
    When there is some error on the transfer end, this error is raised
    """
    pass


class NoGcpsInsideFlightError(ValueError):
    """Error raised when no GCPs are found inside a flight polygon"""
    pass


class MissingBrightnessExifError(ValueError):
    """Error raised when no brightness EXIF value is found in an image"""
    pass