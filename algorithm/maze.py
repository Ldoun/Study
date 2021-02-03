from queue import Queue
from sys import stdin

r, c = map(int, stdin.readline().split())
maze = [stdin.readline().rstrip() for _ in range(r)]
path = Queue()
visited = [[False]*c for _ in range(r)]
path.put([0,0,1])

while not path.empty():
    row,col ,depth= path.get()
    if row == r-1 and col ==c-1:
        print(depth)
        break

    if row+1 < r and visited[row+1][col] == False and maze[row+1][col] == '1':
        visited[row+1][col] == True
        path.put([row+1,col,depth+1])

    if col+1 < c and visited[row][col+1] == False and maze[row][col+1] == '1':
        visited[row][col+1] == True
        path.put([row,col+1,depth+1])

    if row-1 >=0 and visited[row-1][col] == False and maze[row-1][col] == '1':
        visited[row-1][col] == True
        path.put([row-1,col,depth+1])

    if col-1 >= 0 and visited[row][col-1] == False and maze[row][col-1] == '1':
        visited[row][col-1] == True
        path.put([row,col-1,depth+1])   

