from queue import PriorityQueue
import heapq
node_cnt, line_cnt = map(int,input().split(' '))
#graph = [[float('inf')] * node_cnt for _ in range(node_cnt)]
graph = [{} for _ in range(node_cnt)]
target_node  = int(input()) - 1


for i in range(line_cnt):
    here, to, weight = map(int,input().split(' '))
    try:
        if weight < graph[here-1][to-1]:
            graph[here-1][to-1] = weight
    except:
        graph[here-1][to-1] = weight

distance = [float('INF')] * node_cnt # distance from target_node to other
visited = [False] * node_cnt
visitlist = []

distance[target_node] = 0
heapq.heappush(visitlist,(0,target_node))

while visitlist:
    temp, node = heapq.heappop(visitlist)
    if distance[node] < temp:   #알았다 이거 31번째 줄 visitlist.put((distance[i],i)) 이때 최소가 아닌게 들어올수도 있는데 queue에 작은거 들어왔다가 그다음 더 작은거 들어와서 그떄 skip역할
        continue
    visited[node] = True
    for i,weight in graph[node].items():
        if distance[i] > distance[node] + weight:
            distance[i] = distance[node] + weight
            heapq.heappush(visitlist,(distance[i],i))

for i,d in enumerate(distance):
    if d == float('inf'):
        print('INF')
    else:
        print(d)