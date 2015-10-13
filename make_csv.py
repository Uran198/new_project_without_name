import os

full_pref = "imgs/full/"
empty_pref = "imgs/empty/"
full = [full_pref + x for x in os.listdir(full_pref)]
empty = [empty_pref + x for x in os.listdir(empty_pref)]

fd = open("data.csv", 'w')

for x in full:
	fd.write(x + ",1\n")
for x in empty:
	fd.write(x + ",0\n")

fd.close()

