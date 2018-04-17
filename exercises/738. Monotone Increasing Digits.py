def f(ch):
    return eval(ch)


def monotoneIncreasingDigits(N):
    """
    :type N: int
    :rtype: int
    """
    l = list(str(N))
    l = list(map(f, l))
    length = len(l)
    j = 1
    while j < length:
        if j < 1:
            j += 1
            continue
        if l[j] < l[j - 1]:
            l[j - 1] -= 1
            for i in range(j,length):
                l[i] = 9
            j = j - 2
        j += 1
    return eval("".join(list(map(str,l))).lstrip('0'))

print(monotoneIncreasingDigits(10))