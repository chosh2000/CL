import numpy
a = numpy.asarray([ [1,2,3], [4,5,6], [7,8,9] ])
print(a[1])
numpy.savetxt("foo.csv", a, delimiter=",")