import sys

from flowsheet_rl.cli import main

if __name__ == "__main__":
    main(["transfer", *sys.argv[1:]])
