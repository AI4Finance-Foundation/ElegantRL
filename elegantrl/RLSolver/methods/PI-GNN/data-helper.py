import os

os.makedirs("data/raw")
with open("./data/links.txt", "w") as f:
    for i in range(82):
        f.write(f"https://web.stanford.edu/~yyye/yyye/Gset/G{i}\n")
    # # These following data files are not in the same form as the others
    # for i in (1, 11, 13, 32, 48):
    #     f.write(f"https://web.stanford.edu/~yyye/yyye/Gset/ecutG{i}\n")
    #     f.write(f"https://web.stanford.edu/~yyye/yyye/Gset/maxG{i}\n")
    #     f.write(f"https://web.stanford.edu/~yyye/yyye/Gset/stableG{i}\n")

# Some errors are expected when downloading as some of the download links are redundant
os.system("cd data/raw && aria2c -i ../links.txt")
# If you don't have aria2c installed, you can use wget instead:
# os.system("cd data && wget -i links.txt")
