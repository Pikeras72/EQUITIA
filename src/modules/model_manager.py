from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModelForSequenceClassification
from config.seed_utils import set_seed
# Establecer una semilla para reproducibilidad (descomenta para activar)
# set_seed(72)

# Clase encargada de cargar y gestionar los modelos y tokenizadores utilizados en el proyecto.
class ModelManager:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.tokenizers = {}

    # Carga el modelo generador de prompts y su tokenizer con cuantización 4-bit (bitsandbytes)
    def load_generator(self):
        modelo_id = self.config.modelo_generador['id_modelo']
        modo = self.config.modelo_generador['modo_interaccion']
        if modelo_id != '' and modo != '':
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype="float16",
                bnb_4bit_quant_type="nf4"
            )
            tokenizer = AutoTokenizer.from_pretrained(modelo_id, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(modelo_id, quantization_config=bnb_config, low_cpu_mem_usage=True, device_map="auto")
            self.models['generator'] = model
            self.tokenizers['generator'] = tokenizer

    # Carga el modelo LLM a evaluar y su tokenizer con cuantización 4-bit.
    def load_evaluator(self):
        modelo_id = self.config.modelo_a_evaluar['id_modelo']
        modo = self.config.modelo_a_evaluar['modo_interaccion']
        if modelo_id != '' and modo != '':
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype="float16",
                bnb_4bit_quant_type="nf4"
            )
            tokenizer = AutoTokenizer.from_pretrained(modelo_id, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(modelo_id, quantization_config=bnb_config, low_cpu_mem_usage=True, device_map="auto")
            self.models['evaluator'] = model
            self.tokenizers['evaluator'] = tokenizer

    # Carga el modelo de análisis de sentimiento (SequenceClassification) en CUDA y lo pone en eval().
    def load_sentiment(self):
        modelo_id = self.config.modelo_analisis_de_sentimiento['id_modelo']
        modo = self.config.modelo_analisis_de_sentimiento['modo_interaccion']
        if modelo_id != '' and modo != '':
            tokenizer = AutoTokenizer.from_pretrained(modelo_id, use_fast=True)
            model = AutoModelForSequenceClassification.from_pretrained(modelo_id, low_cpu_mem_usage=True)
            model = model.to("cuda")
            model.eval()
            self.models['sentiment'] = model
            self.tokenizers['sentiment'] = tokenizer

    def get_model(self, key):
        return self.models.get(key), self.tokenizers.get(key)
