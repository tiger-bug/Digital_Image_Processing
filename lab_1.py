import numpy as np

x = 3
print(x)

y = "guy"
print(type(y))

print("I am a "+y+".")

z = 9

r = 33


print(z+r)

x = 1
z = 1 - 9j

n = 9.3333


e = 1.0e9
print(type(x),type(z),type(n),type(e))

x = int(9)
x_1 = int("4")
x_2 = float(4)

a = "Sup, dude"

print(a[2:4])

print(a.lower())
print(a.strip())
print(a.replace("S","L"))



print(10 %5)
print(10==5)
print(10!=5)
print(10/5)
print(10*5)
print(10**5)


list0 = ["what","is","this"]
print(list0[1])

for l in list0:
	print(l)

if "dog" in list0:
	print("Sure is!")
else:
	print("Nope!")

	
p = (1,3,5.9)

print(p[0])


del p

thisset = {"a", "b", "c"}

for x in thisset:
	print(x)
	
print("d" in thisset)


thisset.add("d")

d = {

	"1":33.3,
	"2":44,
	"3":22.0

}


print(d)


print(d["1"])

if x == y:
	print("They are equal!!!")
else:
	print("they ain't equal!!!")

	
i = 2

while i <99:
    i += 2
    print(i)
    if i == 44:
        continue
	
string = "Python is awesome!!!"

for s in string:
	if s == "!":
		break
	print(s)
	
def a_function(r = 4):
	area = np.pi * r**2
	circum = np.pi * 2*r
	return (area,circum)
	

print(a_function(9))

	






