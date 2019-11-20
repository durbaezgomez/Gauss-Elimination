arr = [[3 for i in range(2)] for j in range(4)]

s = ""
for i in range(2):
	for j in range(4):
		s += str(arr[j][i])
	s += "\n"

print(s)
print(arr)