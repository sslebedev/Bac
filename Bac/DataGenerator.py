class MyClass:
    def __getitem__(self, key):
        return key * 2

myobj = MyClass()
myobj[3] #Output: 6