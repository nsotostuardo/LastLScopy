import time

def time_function(func):
    """
    Prints execution time of any function
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        resultado = func(*args, **kwargs)
        end = time.time()
        print(f'Execution time: {end - start:.3f}s at {func.__name__}')
        return resultado
    
    return wrapper