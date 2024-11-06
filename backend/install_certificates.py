import ssl
import certifi
import os

def install_certificates():
    # Get the path to the certifi certificate bundle
    cafile = certifi.where()
    
    # Set the SSL_CERT_FILE environment variable
    os.environ['SSL_CERT_FILE'] = cafile
    os.environ['REQUESTS_CA_BUNDLE'] = cafile
    
    # Create a default SSL context using the certifi certificates
    ssl_context = ssl.create_default_context(cafile=cafile)
    
    return ssl_context

if __name__ == "__main__":
    ssl_context = install_certificates()
    print("Certificates installed successfully!") 