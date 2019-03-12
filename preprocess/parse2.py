import os
import sys

c = ''
u = ''
name = ''


for line in sys.stdin:
    if line[:7] == '<<webkb':
        c, u, name = line.strip().split('/')[1:]
        name = name[:-2]
        os.makedirs(os.path.join('processed',c,u), exist_ok=True)
        with open('allfiles.txt', 'a+') as g:
            print(c + '/' + u + '/' + name, file=g)
    else:
        with open(os.path.join('processed',c, u, name+'.txt'), 'a+') as f:
            print(line.strip(), file=f)