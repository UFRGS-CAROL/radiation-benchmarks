import re
print("Exec")
for t in ['relative', 'uint']:
	for b in [1, 194]:
		for h in ["dmrmixed", "none", "dmr"]:
			file = "{}_{}_{}.csv".format(h, b, t)
			with open(file) as fp:
				lines = fp.readlines()
				average = 0
				for l in lines:
					r = re.match(".*Kernel time:(\S+)", l)
					if r:
						average += float(r.group(1)) / 10.0
				print("{},{},{},{}".format(t, b, h, average))
