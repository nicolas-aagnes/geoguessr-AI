import argparse
import os


def count():
    for root, dirs, files in os.walk("data", topdown=False):
        print(root, len(files))


def makedirs():
    with open("generator/countries.txt", "r") as f:
        for country in f.readlines():
            print(country.strip())
            os.mkdir(f"data/{country.strip()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", action="store_true")
    parser.add_argument("--makedirs", action="store_true")
    args = parser.parse_args()

    if args.count:
        count()

    if args.makedirs:
        makedirs()


if __name__ == "__main__":
    main()
