"""Exceptions module."""


class MonitoringException(Exception):
    """Exception for monitoring errors."""


class ReadLangsmithDatasetException(MonitoringException):
    """Exception for reading Langsmith dataset errors."""


class WriteLangsmithDatasetException(MonitoringException):
    """Exception for writing Langsmith dataset errors."""
