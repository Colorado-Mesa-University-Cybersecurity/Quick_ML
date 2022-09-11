import pprint

def pretty(*args):
    return pprint.PrettyPrinter(indent=4, width=30).pprint(*args)
