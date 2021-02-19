import numpy as np
data = np.load('G:\내 드라이브\EEG_classification\output')
lst = data.files
print(lst)
for item in lst:
    print(item)
    print(data[item])

