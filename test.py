x = [1,2,3,4,5,6,7,8]
with open("testfile. txt", "w") as file:
    for i in x:
        file.write(str(i) + ' ')

file.close()