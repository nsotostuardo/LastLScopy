import cupy as cp
def Vram_clean(func):
    """
    Cleans allocated vram 
    """
    def wrapper(*args, **kwargs):
        resultado = func(*args, **kwargs)

        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

        return resultado
    
    return wrapper