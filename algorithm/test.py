import sys
from collections import deque

dx = [-2,-1,1,2,2,1,-1,-2]
dy = [-1,-2,-2,-1,1,2,2,1]
f = open("text.txt", "a")
T = int(sys.stdin.readline())
print(T)
for _ in range(T):
    q = deque()
    size = int(sys.stdin.readline())
    x,y = [int(i) for i in sys.stdin.readline().split()]
    cx,cy = [int(i) for i in sys.stdin.readline().split()]

    q.append([x,y,0])

    v = [[False] * size for _ in range(size)]
    v[y][x] = True

    while q:
        tx,ty,tt = q.popleft()
        f.write("{x} {y} {d}\n".format(x=tx,y=ty,d=tt))
        if cx == tx and cy == ty:
            print(tt)
            break

        for i in range(8):
            if 0<= tx+dx[i] <size and 0<=ty+dy[i] <size and not v[ty+dy[i]][tx+dx[i]]:
                q.append([tx+dx[i],ty+dy[i],tt+1])
                v[ty+dy[i]][tx+dx[i]] = True


f.close()