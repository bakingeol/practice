result2 = 3

# def add2(num):
#     global result2
#     result2 += num
#     return result2

# print(add2(3))

#클래스를 이용해서 여러 인스턴스를 찌겅낸다. 

# class Calculator:
#     def __init__(self):
#         self.result = 0
    
#     def add(self, num):
#         self.result += num
#         return self.result
# cal1 = Calculator()
# cal2 = Calculator()
# print(cal1.add(3))
# print(cal1.add(3))
# print(cal1.add(3))

class FourCal:
    def __init__(self, first, second): #예약어 __init__
        self.first = first
        self.second = second
    def setdata(self, first, second):
        self.first = first
        self.second = second
    def add(self):
        result = self.first + self.second
        return result
    
class MoreFourCal(FourCal):
    def pow(self):
        result = self.first ** self.second
        return result

class SafeCal(FourCal):
    def div(self):
        if self.second ==0:
            return 0
        else:
            return self.first/self.second
