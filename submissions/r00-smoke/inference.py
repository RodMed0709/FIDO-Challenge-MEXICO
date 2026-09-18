"""Submission de humo — R0.

No predice nada útil. Existe para verificar que el contrato de entrega completo
funciona de punta a punta: la firma de 4 argumentos, el nombre model_<id>.pth,
el manejo de oct_volume=None, y los formatos de retorno de ambas tasks.

El score esperado es ~0. Lo que se está probando es que salga un NÚMERO en vez
de un error de ingestión.
"""

import numpy as np


def load_model(model_path):
    # Un modelo real cargaría pesos aquí. La ruta que llega es
    # <submission>/model_0.pth o <submission>/model_1.pth según la task.
    return {"model_path": str(model_path)}


def inference(task_id, oct_volume, opmi_image, model):
    # oct_volume llega None en ~10% de los casos de Task 1 (semilla determinista)
    # y cuando falta el directorio en Task 2. Reventar aquí aborta la corrida
    # entera, no solo el caso.
    has_oct = oct_volume is not None

    height, width = opmi_image.shape[:2]

    if task_id == 0:
        # Task 1: centro de la imagen y una distancia arbitraria.
        return {
            "keypoints": [width / 2.0, height / 2.0],
            "tool_tissue_distance": 50.0,
        }

    # Task 2: similitud reflejada con la escala media observada (~217 px) y
    # rotación cero, centrada en la imagen. Construida con la misma
    # parametrización de 4 DOF que usará el modelo real.
    scale = 217.0
    cos_theta, sin_theta = 1.0, 0.0
    return np.array(
        [
            [scale * cos_theta,  scale * sin_theta, width / 2.0],
            [scale * sin_theta, -scale * cos_theta, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
