import streamlit as st
import requests
from PIL import Image
import io
import os

# Récupérez l'URL de base de l'API à partir des variables d'environnement ou utilisez localhost par défaut

BASE_URL = os.getenv("API_URL", "http://backendp8.azurewebsites.net")
#BASE_URL = "http://localhost:8000"


st.title('Image Semantic Segmentation')

# Téléchargement de l'image par l'utilisateur
file = st.file_uploader("Upload an image for segmentation", type=[
                        "jpg", "png", "jpeg"])

if file is not None:
    # Afficher l'image uploadée
    image = Image.open(file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Bouton pour démarrer la segmentation
    if st.button('Segment Image'):
        # Préparer les données à envoyer
        files = {"file": file.getvalue()}
        segment_url = f"{BASE_URL}/segment-image/"

        # Faire la requête POST à l'API
        response = requests.post(segment_url, files=files)

        if response.status_code == 200:
            # Obtenir l'image segmentée et l'afficher
            segmented_image_bytes = response.content
            segmented_image = Image.open(io.BytesIO(segmented_image_bytes))
            st.image(segmented_image, caption='Segmented Image',
                     use_column_width=True)
        else:
            st.error("Failed to segment the image. Please try again.")
