import os

class PathUtil:
    @staticmethod
    def to_absoluate(path: str) -> str:
        """Convert path to absolute path"""
        return os.path.abspath(path) if not os.path.isabs(path) else path

    @staticmethod
    def ensure_dir_exists(directory: str) -> None:
        """ensure directory exists, if not exists make directory"""
        os.makedirs(directory, exist_ok=True)

    @staticmethod
    def validate_path_is_dir(logger, path: str, path_name:str) -> bool:
        """validate path is a directory and exists"""
        if not os.path.exists(path):
            logger.error(f"{path_name} does not exist: {path}")
            return False
        if not os.path.isdir(path):
            logger.error(f"{path_name} is not a directory: {path}")
            return False
        return True
