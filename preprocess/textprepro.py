import re

def pp(data):
    # data = data.replace("\n", " ")
    data = data.lower()
    data = re.sub(r'ph\.?d\.?', 'phd', data)
    return data