import ssl
import certifi
import os

def configure_ssl():
    """Configure SSL settings for the application."""
    # Set SSL certificate path
    os.environ['SSL_CERT_FILE'] = certifi.where()
    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
    
    # Create SSL context
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    
    # For development only: disable SSL verification
    # WARNING: Don't use this in production!
    ssl._create_default_https_context = ssl._create_unverified_context
    
    return ssl_context

def get_chroma_settings():
    """Get ChromaDB settings with SSL configuration."""
    from chromadb.config import Settings
    
    return Settings(
        anonymized_telemetry=False,
        allow_reset=True,
        is_persistent=True,
        persist_directory="chroma_db"
    ) 