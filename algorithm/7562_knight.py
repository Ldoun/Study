from collections import deque
pro_num = int(input())
case = [{} for i in range(pro_num)]
for i in range(pro_num):
    case[i]['length'] = int(input())
    case[i]['now'] = tuple(map(int,input().split(' ')))
    case[i]['to'] = tuple(map(int,input().split(' ')))


directions = [[-1,-2],[-2,-1],[-2,1],[-1,2],[1,2],[2,1],[2,-1],[1,-2]]

for i in range(pro_num):
    visit = deque()
    visit.append([case[i]['now'][0],case[i]['now'][1],0])
    visited = [[False]*case[i]['length'] for _ in range(case[i]['length'])]
    visited[case[i]['now'][0]][case[i]['now'][1]] = 1
    while visit:
        row,col,depth = visit.popleft()
        if row == case[i]['to'][0] and col == case[i]['to'][1]:
            print(depth)
            break

        for direc_row,direc_col in directions:
            temp_row = row+direc_row
            temp_col = col+direc_col
            if 0<= temp_row< case[i]['length'] and 0<= temp_col< case[i]['length'] and not visited[temp_row][temp_col] :
                visit.append([temp_row,temp_col,depth+1])
                visited[temp_row][temp_col] = True