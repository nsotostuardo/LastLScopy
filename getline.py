from GetLines_package.utils.argparser import parse_args
from GetLines_package.core.controller import main

if __name__ == "__main__":
    args = parse_args()
    main(args)