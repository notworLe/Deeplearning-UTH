def temp_kwargs(**kwargs):
    for key, val in kwargs.items():
        print(key, val)

def temp_args(*args):
    for val in args:
        print(str(val),val)

if __name__ == '__main__':
    # temp_kwargs(a=1, b=2, c="hello")

    # 2
    train = 1
    test = "cc"
    val = "lỏ"
    temp_args(train, test, val)