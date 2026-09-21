import os
from dotenv import load_dotenv
from neuprint import Client

# Load the token from .env
load_dotenv()
token = os.getenv("NEUPRINT_TOKEN")

if token is None:
    print("HATA: Token bulunamadı. .env dosyasını kontrol et.")
else:
    print("Token yüklendi, uzunluk:", len(token))

    # Connect to the hemibrain dataset
    client = Client("neuprint.janelia.org", dataset="hemibrain:v1.2.1", token=token)

    print("\nBağlantı başarılı!")
    print("Dataset:", client.dataset)