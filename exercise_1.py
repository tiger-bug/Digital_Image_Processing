from math import sqrt
import cmath

def quad(a,b,c):
    """This is not the quadratic formula!!!!!"""
#    print()
#    if b**2+-1*4*b*c<0:
#        return (complex(-1*b)+cmath.sqrt(b**2+-1*4*a*c))/complex(2*a),(complex(-1*b)-cmath.sqrt(b**2+-1*4*a*c))/complex(2*a)
#    else:
    return (-1*b+sqrt(b**2+4*b*c))/2*a

def sum_even(a,b):
    """Sums all even numbers between two values,'a' and 'b'"""
    test = list(range(a,b+1))
#    print(test)
    sum_even = 0
    for t in test:
        if t%2==0:
            sum_even+=t
    return sum_even

if __name__=="__main__":
    print("Question 1")
    x = 'This is GIS 5060'
    for l in x:
        print(l)
    print("Question 2")
    a = 3
    b = 10
    c = 5
    print("a,b,c are ",str(a),str(b),str(c))
    roots = quad(a,b,c)
    print(roots)
    print("Question 3")
    x = sum_even(1,100)
    print(x)

