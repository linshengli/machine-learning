def dailyTemperatures(temperatures):
    """
    :type temperatures: List[int]
    :rtype: List[int]
    """
    
    '''
    ret = []
    length = len(temperatures)
    for i in range(length):
        index = 0
        for j in range(i + 1, length):
            if temperatures[j] > temperatures[i]:
                index = j - i
                break
        ret.append(index)
    return ret
    '''
    
    l = [0 for i in range(101)]
    length = len(temperatures)
    ret = [0 for i in range(length)]
    l[temperatures[-1]] = length - 1
    for i in range(length - 2, -1, -1):
        t = temperatures[i]
        l[t] = i
        for j in range(t + 1,101):
            if l[j] > 0:
                if ret[i]>0:
                    ret[i] = min(ret[i], l[j] - i)
                else :
                    ret[i] = l[j] - i
    return ret

print(dailyTemperatures([73,74,75,71,69,72,76,73]))