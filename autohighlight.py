import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", type=str,
                        help="source file path")
    parser.add_argument("-a", "--annotation", type=str,
                        help="annotation file path")
    parser.add_argument("-o", "--output", type=str,
                        help="output file path")
    args = parser.parse_args()

    if not args.source:
        raise Exception('No source file path specified.')


    print(args.source)
    print(args.annotation)
    print(args.output)
    print("Done.")
