from GetLines_package.utils.argparser import parse_args
from GetLines_package.core.controller import main
#import time
if __name__ == "__main__":
    args = parse_args()
#    start = time.time()
    main(args)
#    end = time.time()
#    print(f'Tiempo de ejecucion: {end - start:3f} segundos')