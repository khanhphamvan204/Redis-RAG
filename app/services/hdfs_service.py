from hdfs import InsecureClient
from typing import Optional, List
import logging
import os
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

class HostnameReplacingAdapter(HTTPAdapter):
    """Custom HTTP Adapter to replace Docker hostnames with localhost"""
    
    def __init__(self, hostname_map=None, *args, **kwargs):
        self.hostname_map = hostname_map or {}
        super().__init__(*args, **kwargs)
    
    def send(self, request, *args, **kwargs):
        # Replace hostnames in the request URL
        for old_host, new_host in self.hostname_map.items():
            if old_host in request.url:
                request.url = request.url.replace(old_host, new_host)
                logger.debug(f"Replaced {old_host} with {new_host} in URL")
        return super().send(request, *args, **kwargs)

class HDFSService:
    """Service for interacting with Hadoop HDFS"""
    
    def __init__(self, namenode_url: str = None, user: str = "root"):
        """
        Initialize HDFS client
        
        Args:
            namenode_url: URL of HDFS NameNode (default from env or http://hadoop-namenode:9870)
            user: HDFS user (default: root)
        """
        self.namenode_url = namenode_url or os.getenv("HDFS_NAMENODE_URL", "http://hadoop-namenode:9870")
        self.user = user or os.getenv("HDFS_USER", "root")
        self.client = None
        self._connect()
    
    def _connect(self):
        """Establish connection to HDFS with hostname replacement"""
        try:
            # Create custom session with hostname replacement
            session = Session()
            
            # Map Docker hostnames to localhost with correct ports
            hostname_map = {
                'hadoop-namenode:9870': 'localhost:9870',
                'namenode:9870': 'localhost:9870',
                'hadoop-datanode:9864': 'localhost:9864',
                'datanode:9864': 'localhost:9864',
            }
            
            # Add retry strategy
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
            )
            
            adapter = HostnameReplacingAdapter(
                hostname_map=hostname_map,
                max_retries=retry_strategy
            )
            
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            
            # Create HDFS client with custom session
            self.client = InsecureClient(self.namenode_url, user=self.user, session=session)
            logger.info(f"✓ Connected to HDFS at {self.namenode_url} as user {self.user}")
        except Exception as e:
            logger.error(f"✗ Failed to connect to HDFS: {e}")
            raise
    
    def create_directory(self, path: str, permission: str = None) -> bool:
        """
        Create directory in HDFS
        
        Args:
            path: HDFS directory path
            permission: Unix-style permissions (e.g., '755')
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.makedirs(path, permission=permission)
            logger.info(f"Created directory: {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create directory {path}: {e}")
            return False
    
    def upload_file(self, local_path: str, hdfs_path: str, overwrite: bool = False) -> bool:
        """
        Upload file to HDFS
        
        Args:
            local_path: Local file path
            hdfs_path: Destination HDFS path
            overwrite: Whether to overwrite existing file
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.upload(hdfs_path, local_path, overwrite=overwrite)
            logger.info(f"Uploaded {local_path} to {hdfs_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload file: {e}")
            return False
    
    def download_file(self, hdfs_path: str, local_path: str, overwrite: bool = False) -> bool:
        """
        Download file from HDFS
        
        Args:
            hdfs_path: Source HDFS path
            local_path: Destination local path
            overwrite: Whether to overwrite existing file
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.download(hdfs_path, local_path, overwrite=overwrite)
            logger.info(f"Downloaded {hdfs_path} to {local_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download file: {e}")
            return False
    
    def list_files(self, path: str = '/') -> List[str]:
        """
        List files in HDFS directory
        
        Args:
            path: HDFS directory path
        
        Returns:
            List of filenames
        """
        try:
            return self.client.list(path)
        except Exception as e:
            logger.error(f"Failed to list files in {path}: {e}")
            return []
    
    def read_file(self, hdfs_path: str, encoding: str = 'utf-8') -> Optional[str]:
        """
        Read file content from HDFS
        
        Args:
            hdfs_path: HDFS file path
            encoding: File encoding
        
        Returns:
            File content as string, or None if failed
        """
        try:
            with self.client.read(hdfs_path, encoding=encoding) as reader:
                return reader.read()
        except Exception as e:
            logger.error(f"Failed to read file {hdfs_path}: {e}")
            return None
    
    def write_file(self, hdfs_path: str, content: str, overwrite: bool = False, encoding: str = 'utf-8') -> bool:
        """
        Write content to HDFS file
        
        Args:
            hdfs_path: HDFS file path
            content: Content to write
            overwrite: Whether to overwrite existing file
            encoding: File encoding
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.write(hdfs_path, content, overwrite=overwrite, encoding=encoding)
            logger.info(f"Written content to {hdfs_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to write to {hdfs_path}: {e}")
            return False
    
    def append_file(self, hdfs_path: str, content: str, encoding: str = 'utf-8') -> bool:
        """
        Append content to HDFS file
        
        Args:
            hdfs_path: HDFS file path
            content: Content to append
            encoding: File encoding
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.client.write(hdfs_path, append=True, encoding=encoding) as writer:
                writer.write(content)
            logger.info(f"Appended content to {hdfs_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to append to {hdfs_path}: {e}")
            return False
    
    def delete(self, hdfs_path: str, recursive: bool = False) -> bool:
        """
        Delete file or directory from HDFS
        
        Args:
            hdfs_path: HDFS path to delete
            recursive: Whether to delete recursively
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.delete(hdfs_path, recursive=recursive)
            logger.info(f"Deleted {hdfs_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete {hdfs_path}: {e}")
            return False
    
    def get_file_status(self, hdfs_path: str) -> Optional[dict]:
        """
        Get file/directory status
        
        Args:
            hdfs_path: HDFS path
        
        Returns:
            Status dict with metadata, or None if failed
        """
        try:
            return self.client.status(hdfs_path)
        except Exception as e:
            logger.error(f"Failed to get status of {hdfs_path}: {e}")
            return None
    
    def exists(self, hdfs_path: str) -> bool:
        """
        Check if path exists in HDFS
        
        Args:
            hdfs_path: HDFS path
        
        Returns:
            True if exists, False otherwise
        """
        try:
            self.client.status(hdfs_path)
            return True
        except:
            return False
    
    def get_file_checksum(self, hdfs_path: str) -> Optional[str]:
        """
        Get file checksum
        
        Args:
            hdfs_path: HDFS file path
        
        Returns:
            Checksum string, or None if failed
        """
        try:
            return self.client.checksum(hdfs_path)
        except Exception as e:
            logger.error(f"Failed to get checksum of {hdfs_path}: {e}")
            return None

# Singleton instance
hdfs_service = HDFSService()
