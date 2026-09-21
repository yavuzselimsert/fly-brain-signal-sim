import os
from dotenv import load_dotenv
from neuprint import Client, fetch_neurons, NeuronCriteria as NC

load_dotenv()
token = os.getenv("NEUPRINT_TOKEN")

client = Client("neuprint.janelia.org", dataset="hemibrain:v1.2.1", token=token)

# Search for neurons in the antennal lobe (olfactory input region)
criteria = NC(rois=["AL(R)"])  # AL = Antennal Lobe, right hemisphere
neurons_df, roi_counts_df = fetch_neurons(criteria)

print(f"Antennal Lobe'da bulunan nöron sayısı: {len(neurons_df)}")
print("\nİlk 10 nöron:")
print(neurons_df[["bodyId", "type", "instance", "pre", "post"]].head(10))