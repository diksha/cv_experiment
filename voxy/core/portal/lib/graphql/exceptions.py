import sys
from enum import Enum, unique


# Error codes for all module exceptions
@unique
class ErrorCode(Enum):
    UNKNOWN = "unknown_errcode"  # error code passed is not specified in enum ErrorCode
    PERMISSION_DENIED = "permission_errcode"  # permissions error when user takes forbidden action


class CustomException(Exception):
    def __init__(self, error_code: ErrorCode, message: str = ""):
        # Raise a separate exception in case the error code passed isn't specified in the ErrorCode enum
        if not isinstance(error_code, ErrorCode):
            msg = (
                "Error code passed in the error_code param must be of type {0}"
            )
            raise CustomException(
                ErrorCode.UNKNOWN,
                msg,
            )

        self.code = error_code
        self.traceback = sys.exc_info()

        try:
            msg = f"[{error_code.name}] {message}"
            self.message = msg
        except (IndexError, KeyError):
            msg = f"[{error_code.name}] {message}"
            self.message = msg

        super().__init__(msg)


class PermissionDenied(CustomException):
    def __init__(self, message: str = None) -> None:
        default_message = "You do not have permission to perform this action"
        if message is None:
            message = default_message
        super().__init__(ErrorCode.PERMISSION_DENIED, message)
