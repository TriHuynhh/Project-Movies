from cProfile import label
import math
from time import time
import math

a_number = 34 
if a_number % 2 == 0:
    print("We're inside an if block")
    print('The given number {} is even'.format(a_number),"This is an example")

#Shorthand "IF" conditional expression
parity = 'even' if a_number % 2 == 0 else 'odd'

print('The number {} is {}.'.format(a_number,parity))


result = 1
i = 1
while i <= 100: 
    result *= i 
    i += 1
print(result)


line = "*"
max_length = 10

while len(line) < max_length:
    print(line)
    line += "*"
while len(line) > 0: 
    print(line)
    line = line[:-1]  ##Remove 1 character from string

space_count = 10
max_length1 = 11 

if {None: None}:
    print("Well Done!")
else:
    print("Wrong")


#for i in range(12,31,2):
#    print(i)

i = 1
j = 0
while 1:
    i+=1
    if i%2 == 0:
        continue
    j+=1
    if i > 10:
        break
c = i + j*2
print(c)

#for i in range (1,3):
#    print(i)

#for i in range(1,1000):
#    if i>100:
#        break
#    else:
#        print(i)

star_list = ""
for i in range(5):
    star_list += " *"
    print(" "*(4-i),star_list)

test_list = "0"
for i in range(5):
    print(test_list*(4-i))

test_list1 = "*"
for i in range(11):
    print(" "*(11-i),test_list1)
    test_list1 += "*"

for i in range(1,10):
    print(i,end = " ")

for i in range(1,11):
    print(i,sep = " ")


#FUNCTION

def loan_emi(amount,duration,rate,down_payment=0):   #Optional argument assign = 0 
    """Calculates the equal monthly installments (EMI) for a loan.
    
    Args:
        amount - Total amount to be spent (loan + down payment)
        duration - Duration of the loan (in months)
        rate - Rate of interest (monthly)
        down_payment (int, optional): Optional initial payment (deducted from amount). Defaults to 0.
    """
    loan_amount = amount - down_payment
    try:
        emi = loan_amount * rate * ((1+rate)**duration) / (((1+rate)**duration)-1)
    except ZeroDivisionError:
        emi = loan_amount/duration
    emi = math.ceil(emi)
    return emi

emi1 = loan_emi(1260000,8*12,0.1/12,3e5)
emi2 = loan_emi(1260000,10*12,0.08/12)
emi3 = loan_emi(
    amount=1260000,
    duration=8*12,
    rate=0.1/12,
    down_payment=3e5,
)
print(emi1,emi2,emi3)

help(math.ceil)

if emi1 < emi2:
    print("Option 1 has the lower EMI: ${}".format(emi1))
else: 
    print("Option 2 has the lower EMI: ${}".format(emi2))

emi_without_interest = loan_emi(
    amount=100000,
    duration=12*10,
    rate=0/12
)

emi_with_interest = loan_emi(
    amount=100000,
    duration=12*10,
    rate=0.09/12
)

total_interest = (emi_with_interest-emi_without_interest)*10*12
print("The total interest paid is ${}".format(total_interest))

help(loan_emi)

def cost_to_visit(city,return_flight,hotel_per_day,weekly_car_rental,total_days):
    hotel = total_days*hotel_per_day
    if(total_days % 7 == 0):
        car = (total_days/7)*weekly_car_rental 
    else:
        car = math.ceil(total_days/7)*weekly_car_rental
    total_cost = math.ceil(hotel + car + return_flight)
    return total_cost

days = 15
ctv1 = cost_to_visit(
    city="Paris",
    return_flight=200,
    hotel_per_day=20,
    weekly_car_rental=200,
    total_days=days
)

ctv2 = cost_to_visit(
    city="London",
    return_flight=250,
    hotel_per_day=30,
    weekly_car_rental=120,
    total_days=days
)

ctv3 = cost_to_visit(
    city="Dubai",
    return_flight=370,
    hotel_per_day=15,
    weekly_car_rental=80,
    total_days=days
)

ctv4 = cost_to_visit(
    city="Mumbai",
    return_flight=450,
    hotel_per_day=10,
    weekly_car_rental=70,
    total_days=days
)

print(ctv1,ctv2,ctv3,ctv4)

# Add two lists using map and lambda
  
numbers1 = [1, 2, 3]
numbers2 = [4, 5, 6]

#map() function returns a map object(which is an iterator)
#of the results after applying the given function to each item of a given iterable (list, tuple etc.)

result = map(lambda x, y: x + y, numbers1, numbers2)
print(type(result))
print(result)
print(list(result))
print(type(numbers1))

result1 = list(map(lambda x,y: x+y, numbers1,numbers2))
print(result1)

def func(x):
    return x%2 == 0

li = [2,9,5,10,11,34,57,66,93,102]
li = list(map(func,li))
print(sum(li))

print(type(True / True))
print(False / True)
print(type(5 // 2))
#print(type(False // False))
print(type(124 / 2))
print(type(5.3 // 2))


li = [[(i+1) for i in range(5)] for j in range(5)]
for line in li:
    print(*line[::-1])

def make_multiplier(n):
    def multiplier(x,y):
        return x * n + y
    return multiplier

num1 = make_multiplier(6)
num2 = make_multiplier(9)

print(num1)
print(num2)
print(num1(2,5))
#print(num2(5,5))
#print(num2(num2(3,3),num1(5,10)))

