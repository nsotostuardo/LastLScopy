from SearchLine_package.utils.argparser import parse_args
from SearchLine_package.core.controller import main

if __name__ == "__main__":
    args = parse_args()
    main(args)