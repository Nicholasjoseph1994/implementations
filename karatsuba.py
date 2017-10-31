def karatsuba(num1, num2):
    """ 
    Multiplies two numbers using the Karatsuba algorithm
    ARGS:
        num1 (str) the first number to be multiplied
        num2 (str) the second number to be multiplied
    RETURNS:
        (str) - The product of num1 and num2
    """
    
    # Pad zeros if the strings are different lengths
    if len(num2) > len(num1):
        num1 = '0'*(len(num2) - len(num1)) + num1
    elif len(num2) < len(num1):
        num2 = '0'*(len(num1) - len(num2)) + num2
        
    # Base case of recursion
    n = len(num1) 
    if n == 1:
        return str(int(num1)*int(num2))

    # Split into smaller subproblems
    midpoint = n / 2
    a = num1[:midpoint]
    b = num1[midpoint:]
    c = num2[:midpoint]
    d = num2[midpoint:]

    # prep (a+c)*(b+d)
    sum1 = str(int(a) + int(b))
    sum2 = str(int(c)+int(d))
    if len(sum1) > len(sum2):
        sum2 = '0'+sum2
    if len(sum2) > len(sum1):
        sum1 = '0'+sum1

    # Recursive calls: a*c, b*d, (a+b)*(c+d)
    p1 = karatsuba(a, c)
    p2 = karatsuba(b, d)
    p3 = karatsuba(str(int(a)+int(b)), str(int(c) + int(d)))

    # Compute the sum using recursive returns
    term1 = p1+'0'*(2*(n-midpoint))
    term2 =  str(int(p3) - int(p2) - int(p1)) + '0'*(n-midpoint)
    term3 = p2

    return str(int(term1) + int(term2) + int(term3))
    
def test_karatsuba():
    assert karatsuba("1", "1") == "1"
    assert karatsuba("0", "0") == "0"
    assert karatsuba("25", "25") == "625"
    assert karatsuba("61", "61") == "3721"
    assert karatsuba("5678", "1234") == "7006652"
    assert karatsuba("10000", "52317") == "523170000"

    bignum1 = "3141592653589793238462643383279502884197169399375105820974944592"
    bignum2 = "2718281828459045235360287471352662497757247093699959574966967627"
    prod = "8539734222673567065463550869546574495034888535765114961879601127067743044893204848617875072216249073013374895871952806582723184"
    assert karatsuba(bignum1, bignum2) == prod
    
    print 'Tests passed'
test_karatsuba()

