row, dap = map(int,input().split(' '))

money = []
cnt = 0
for i in range(row):
    money.append(int(input()))

money = list(reversed(money))

for m in money:
    if m > dap:
        continue
    
    temp = dap//m
    dap -= temp*m
    cnt += temp

print(cnt)

