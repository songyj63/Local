#%%

def number_generator(stop):
    n = 0              # 숫자는 0부터 시작
    while n < stop:    # 현재 숫자가 반복을 끝낼 숫자보다 작을 때 반복
        yield n        # 현재 숫자를 바깥으로 전달
        n += 1         # 현재 숫자를 증가시킴

for i in number_generator(3):
    print(i)

g = number_generator(3)
print(g.__next__())
print(next(g))
print(next(g))

#%%
def upper_gen(x):
    for i in x:
        yield i.upper()

fruites = ['apple', 'pear', 'grape']
for i in upper_gen(fruites):
    print(i)

#%%
def number_generator():
    x = [1, 2, 3]
    yield from x    # 리스트에 들어있는 요소를 한 개씩 바깥으로 전달
 
for i in number_generator():
    print(i)