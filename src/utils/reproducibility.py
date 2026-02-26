def set_seed(seed: int = 72):
    """
    Establece la semilla para PyTorch con opciones deterministas en cuDNN.
    """
    import torch
    # Establecer una semilla para aumentar la reproducibilidad del script
    torch.manual_seed(seed)  # Establece la semilla para todos los generadores de números aleatorios en CPU de PyTorch
    torch.cuda.manual_seed_all(seed)  # Establece la semilla para todas las GPUs que estés usando
    torch.backends.cudnn.deterministic = True  # Hace que cuDNN utilice algoritmos deterministas
    torch.backends.cudnn.benchmark = False  # Evita selección del algoritmo "más rápido" no determinista