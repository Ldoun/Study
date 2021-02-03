from queue import Queue
r, c = map(int,input().split(' '))

row_cnt = 0
field = []
result = 0
for i in range(r):
    field.append(list(input()))

where = [[0,0]]
directions = [1,0,-1]
for i in range(r):
    where = [[i,0]]
    while where:
        row,col = where.pop()
        field[row][col] = '0'

        if col == c-1:
            result+=1
            break
        
        for direction in directions:
            temp_row = row + direction
            if 0<=temp_row<r and field[temp_row][col+1] == '.':
                where.append([temp_row,col+1])

print(result)

        