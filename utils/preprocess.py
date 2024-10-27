import os

if __name__ == "__main__":
    programs = []
    with open('data/train.csv', 'r') as fp:
        for line in fp.readlines()[1:]:
            if line.startswith('"/'):
                program = line[1:]
            elif line.startswith('"'):
                programs.append(program)
            else:
                program += line
    import pdb; pdb.set_trace()
