""" Functions for sorting text data """
import re

def sort_alphanumeric(data):
    """ Given a list of text, sort it alphanumerically such as when sorting filenames"""
    def convert(text): 
        if text.isdigit():
            return int(text)
        else:
            return text.lower()

    def alphanum_key(key): 
        key_out = []
        for c in re.split('([0-9]+)', key):
            key_out.append(convert(c))
        return key_out

    return sorted(data, key=alphanum_key)
