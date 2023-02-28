with open("abalone.data", "rt") as fin:
    with open("out.txt", "wt") as fout:
        for line in fin:
            if line[0] == "M":
                fout.write(line.replace("M", "0"))
            elif line[0] == "F":
                fout.write(line.replace("F", "1"))
            elif line[0] == "I":
                fout.write(line.replace("I", "2"))
