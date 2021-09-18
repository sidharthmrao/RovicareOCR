x = open("requirements.txt", 'r+')
for line in x:
    print(f"RUN pip install {line}")