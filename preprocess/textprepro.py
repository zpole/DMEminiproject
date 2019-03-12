import re

def pp(data):
    # data = data.replace("\n", " ")
    data = data.lower()
    data = ' ' + data
    data = re.sub(r'[^\w.]ph\.? ?d\.?', ' phd', data)
    return data