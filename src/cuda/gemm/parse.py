import re
for t in ['relative', 'uint']:
	for b in [1, 34]:
		for h in ["mixed", "none", "full"]:
			file = "{}_{}_{}.csv".format(h, b, t)
			with open(file) as fp:
				lines = fp.readlines()
				for l in lines:
					r = re.match("Elapsed time: (\S+) s", l)
					if r:
						print("{}, {}, {}, {}".format(t, b, h, r.group(1).rstrip()))
						break


