from queue import Queue
node, line, start = map(int,input().split(' '))

graph = [[] for i in range(node + 1)]  #0번 사용 X
for i in range(line):
    here , there = map(int,input().split(' '))
    
    graph[here].append(there)
    graph[there].append(here)

dfs_went = []
bfs_went = []
for i in graph:
    i.sort()

def dfs_search(graph,From):
    global dfs_went
    dfs_went.append(From)
    n = graph[From]
    for cannidate in n:
        if cannidate in dfs_went:
            continue
        else:
            dfs_search(graph,cannidate)
    return

def bfs_search(graph, start):
    global bfs_went
    to = Queue()
    to.put(start)
    
    while not to.empty():
        n = to.get()
        if n not in bfs_went:
            bfs_went.append(n)
        for node in graph[n]:
            if node not in bfs_went:
                to.put(node)
    return
dfs_search(graph,start)
bfs_search(graph,start)
for i in dfs_went:
    print(i,end=" ")
print("")
for j in bfs_went:
    print(j,end=" ")

    