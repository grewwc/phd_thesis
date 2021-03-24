import argparse
import os

parser = argparse.ArgumentParser()


parser.add_argument("url")
parser.add_argument("rootdir", nargs="?")

if __name__ == '__main__':
    parsed, unknown = parser.parse_known_args()
    print(parsed)
    print(unknown)