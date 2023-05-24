import unittest

def print_suite(suite):
    if hasattr(suite, '__iter__'):
        for x in suite:
            print_suite(x)
    else:
        print(suite)

if __name__ == '__main__':
    print_suite(unittest.defaultTestLoader.discover('.'))