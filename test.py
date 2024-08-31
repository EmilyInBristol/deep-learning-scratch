s = "  the sky is blue##"
print(s)

while s[0] == ' ':
    s = s[1:]

print(s)

while s[-1] == '#':
    s = s[:-1]

print(s)

l = []
temp = ''
t = False
while s:
    while s[0] == ' ':
        s = s[1:]
        t = True
    
    if t:
        l.append(temp)
        temp = ''
    else:
        temp += s[0]
        s = s[1:]
        t = False

print(l)