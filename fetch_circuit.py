import os
from dotenv import load_dotenv
from neuprint import Client, fetch_neurons, fetch_adjacencies, NeuronCriteria as NC

load_dotenv()
token = os.getenv("NEUPRINT_TOKEN")

client = Client("neuprint.janelia.org", dataset="hemibrain:v1.2.1", token=token)


def get_projection_neurons():
    """
    Fetch projection neurons connecting the antennal lobe to
    higher brain regions (mushroom body, lateral horn).
    """
    # Projection neuron types typically contain "PN" in their name
    # and reside in the antennal lobe.
    criteria = NC(rois=["AL(R)"], type=".*PN.*", regex=True)
    neurons_df, roi_counts_df = fetch_neurons(criteria)

    return neurons_df


def get_connectivity(neurons_df):
    """
    Fetch synaptic connections between the given neurons
    (connections where both source and target are in our set).
    """
    body_ids = neurons_df["bodyId"].tolist()

    neuron_df, conn_df = fetch_adjacencies(sources=body_ids, targets=body_ids)

    return conn_df


def main():
    print("Projeksiyon nöronları çekiliyor...")
    neurons_df = get_projection_neurons()
    print(f"Bulunan projeksiyon nöronu sayısı: {len(neurons_df)}")
    print("\nÖrnek nöronlar:")
    print(neurons_df[["bodyId", "type", "pre", "post"]].head(10))

    print("\nBağlantılar çekiliyor (bu biraz zaman alabilir)...")
    conn_df = get_connectivity(neurons_df)
    print(f"\nBulunan bağlantı (sinaps) sayısı: {len(conn_df)}")
    print("\nÖrnek bağlantılar:")
    print(conn_df.head(10))

    # Save to disk so we don't need to re-fetch every time
    neurons_df.to_csv("neurons.csv", index=False)
    conn_df.to_csv("connections.csv", index=False)
    print("\nVeriler neurons.csv ve connections.csv olarak kaydedildi.")


if __name__ == "__main__":
    main()