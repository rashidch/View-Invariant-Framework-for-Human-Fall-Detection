import argparse

parser = argparse.ArgumentParser(description="Fall Detection App")
parser.add_argument("echo")
args = parser.parse_args()
print(args.echo)