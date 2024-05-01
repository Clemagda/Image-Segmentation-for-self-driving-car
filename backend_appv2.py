

# TODO: Ajuster le pétraitement des images en conséquence
# TODO: Debuggage du script. Nettoyer les parties de conversion résiduelles
# TODO: Ajuster les requirements si nécessaires (j'en doute).
# TODO: Vérifier l'execution avec streamlit en local
# TODO: Pousser et déployer.

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import os
from PIL import Image
import uvicorn
import io
import numpy as np
import tensorflow as tf


# Configuration des identifiants de classe et des couleurs
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

# Charger le modèle TFLite


# Prétraiter l'image pour correspondre aux attentes d'entrée de TFLite


def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_resized = image.resize((512, 512))
    image_np = np.array(image_resized) / 255.0  # Normalisation
    # Changement de l'ordre des axes pour [C, H, W]
    image_np = np.transpose(image_np, (2, 0, 1))
    image_np = image_np.astype(np.float32)
    # Ajoute une dimension de lot [N, C, H, W]
    return np.expand_dims(image_np, axis=0)


app = FastAPI()


@app.post("/segment-image/")
async def segment_image(file: UploadFile = File(...)):
    interpreter = tf.lite.Interpreter(model_path='segformer.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Lecture et prétraitement de l'image
    image_bytes = await file.read()
    input_data = preprocess_image(image_bytes)

    # Exécution de l'inférence
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    output_prob = tf.nn.softmax(output_data, 1)
    # Application des couleurs basées sur la prédiction
    pred_seg = np.argmax(output_data, axis=0)  # -1
    print(f"Unique labels predicted : {np.unique(pred_seg)}")
    mapped_seg = np.vectorize(id2label.get)(pred_seg)
    color_seg = np.zeros((128, 128, 3), dtype=np.uint8)
    for label, color in class_colors.items():
        color_seg[mapped_seg == label] = color
    color_image = Image.fromarray(color_seg)

    # Préparation de la réponse
    img_byte_arr = io.BytesIO()
    color_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return StreamingResponse(img_byte_arr, media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
