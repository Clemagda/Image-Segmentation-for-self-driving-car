from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
import uvicorn
import io
import numpy as np
import tensorflow as tf
from transformers import SegformerFeatureExtractor, TFSegformerForSemanticSegmentation
id2label = {
    0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 1, 8: 1, 9: 1, 10: 1, 11: 2,
    12: 2, 13: 2, 14: 2, 15: 2, 16: 2, 17: 3, 18: 3, 19: 3, 20: 3, 21: 4,
    22: 4, 23: 5, 24: 6, 25: 6, 26: 7, 27: 7, 28: 7, 29: 7, 30: 7, 31: 7,
    32: 7, 33: 7, 34: 7}

# Dictionnaire pour les couleurs des classes
class_colors = {
    0: (0, 0, 0),       # Classe 0 : Noir - Background/Void
    1: (128, 0, 0),     # Classe 1 : Rouge Foncé - Flat
    2: (0, 128, 0),     # Classe 2 : Vert Foncé - Sky
    3: (128, 128, 0),   # Classe 3 : Olive - Human
    4: (0, 0, 128),     # Classe 4 : Bleu Foncé - Vehicle
    5: (128, 0, 128),   # Classe 5 : Pourpre - Object
    6: (0, 128, 128),   # Classe 6 : Teal - Construction
    7: (128, 128, 128)  # Classe 7 : Gris - Nature
}

app = FastAPI()

# Initialiser le modèle pré-entraîné et le feature extractor
# Mettez à jour avec le chemin de votre modèle
model_path = "HF_model"
pretrained_model = TFSegformerForSemanticSegmentation.from_pretrained(
    model_path)
feature_extractor = SegformerFeatureExtractor(size=224)  # 256


@app.post("/segment-image/")
async def segment_image(file: UploadFile = File(...)):
    try:
        # Lecture de l'image uploadée
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Redimensionnement et normalisation de l'image pour le modèle
        image_resized = image.resize((224, 224))  # (512, 512)
        image_np = np.array(image_resized) / 255.0
        # Transposition des canaux pour le modèle
        image_np = np.transpose(image_np, (2, 0, 1))
        image_tensor = np.expand_dims(image_np, axis=0)

        # Prédiction avec le modèle
        inputs = feature_extractor(
            images=image_tensor, return_tensors="tf", do_rescale=False)
        outputs = pretrained_model(**inputs)
        logits = outputs.logits
        logits = tf.transpose(logits, [0, 2, 3, 1])
        upsampled_logits = tf.image.resize(
            logits, image.size[::-1])  # Redimensionnement des logits
        pred_seg = tf.math.argmax(upsampled_logits, axis=-1)[0].numpy()

        # Mapper les 35 labels en 8 classes
        mapped_seg = np.vectorize(id2label.get)(pred_seg)

        # Application de la palette de couleurs
        color_seg = np.zeros((1024, 2048, 3), dtype=np.uint8)
        for label, color in class_colors.items():
            color_seg[mapped_seg == label, :] = color
        color_image = Image.fromarray(color_seg)

        # Retour de l'image originale et de l'image segmentée
        img_byte_arr = io.BytesIO()
        color_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        return StreamingResponse(img_byte_arr, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if __name__ == "__main__":
        port = int(os.getenv("PORT", 8000))  # Utiliser le port défini par Azure ou, par défaut, 8000
        uvicorn.run(app, host="0.0.0.0", port=port)
