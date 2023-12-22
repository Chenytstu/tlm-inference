def comm(path):
    with open(path, 'r') as f:
        all = 0
        for i in f.readlines():
            all += int(i)
            
    return all

if __name__ == "__main__":
    print(comm("log2"))