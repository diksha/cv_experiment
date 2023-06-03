import os
import tempfile
import time
from datetime import datetime

from loguru import logger
from neo4j import GraphDatabase

from core.metaverse.metaverse import Metaverse
from core.utils.aws_utils import upload_directory_to_s3
from third_party.neo4j_backup.neo4j_extractor import Extractor


class MetaverseBackupWrapper:
    """Class to backup metaverse"""

    _USER = "neo4j"
    _TRUST = "TRUST_ALL_CERTIFICATES"
    _BUCKET = "voxel-metaverse-backup"

    def __init__(self, metaverse_environment: str):
        """Constructor
        Args:
            metaverse_environment
        """
        self.environment = metaverse_environment
        uri = Metaverse.database_uri(metaverse_environment)
        password = Metaverse.database_password(metaverse_environment)
        encrypted = Metaverse.is_database_encrypted(metaverse_environment)
        self.__driver = GraphDatabase.driver(
            uri,
            auth=(self._USER, password),
            encrypted=encrypted,
            trust=self._TRUST,
            max_connection_lifetime=100,
        )

    def backup_metaverse(self) -> None:
        """Backup metaverse
        Raises:
            RuntimeError: failed to backup to s3
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            backup_time = datetime.utcnow().strftime("%Y-%m-%d-T%H-%M-%S")
            backup_path = os.path.join(tmpdir, backup_time)
            extractor = Extractor(
                backup_path,
                driver=self.__driver,
                database="neo4j",
                input_yes=True,
                compress=True,
            )
            extraction_start_time = time.time()
            extractor.extract_data()
            logger.info(
                (
                    "Extraction completed, total time, "
                    f"{time.time() - extraction_start_time}s",
                )
            )
            upload_dir = os.path.join(self.environment, backup_time)
            s3_upload_path = upload_directory_to_s3(
                self._BUCKET,
                backup_path,
                upload_dir,
            )
            if s3_upload_path is None:
                raise RuntimeError("Failed to dump metaverse data to s3")
            logger.info(f"Uploaded backup to {s3_upload_path}")
