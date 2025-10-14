import json
from config.seed_utils import set_seed
# Establecer una semilla para reproducibilidad (descomenta para activar)
# set_seed(72)

# Clase para gestionar la carga y acceso a la configuración general del proyecto.
class ConfigManager:
    def __init__(self, config_path='config/config_modelos.json'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

    def get(self, key, default=None):
        return self.config.get(key, default)

    @property
    def modelo_generador(self):
        return self.config['modelo_generador']

    @property
    def modelo_a_evaluar(self):
        return self.config['modelo_a_evaluar']

    @property
    def modelo_analisis_de_sentimiento(self):
        return self.config['modelo_analisis_de_sentimiento']
