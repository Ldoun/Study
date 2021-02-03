n = int(input())
map = [[0 for i in range(n)] for j in range(n)]

def draw(n):
    global map
    if n == 3:
        map[0][:3] = [1]*3
        map[1][:3] = [1,0,1]
        map[2][:3] = [1]*3
        return

    a = n//3
    draw(n//3)
    for i in range(3):
        for j in range(3):
            if i == 1 and j==1:
                continue
            for k in range(a):
                map[a*i+k][a*j:a*(j+1)] = map[k][:a]
draw(n)
for i in map:
    for j in i:
        if j ==1:
            print('*',end ="")
        else:
            print(' ',end ="")
    print('')
