def QuiclSort(alist,start,end):

    if start>=end:
        return
    mid = alist[start]

    low = start
    high = end

    while low<high:

        while low<high and alist[high]>=mid:
            high -=1

        alist[low]=alist[high]

        while low<high and alist[low]<=mid:
            low+=1

        alist[high] = alist[low]

    alist[low] = mid

    QuiclSort(alist,start,low-1)
    QuiclSort(alist,high+1,end)

alist = [1,5,2,4,3,-1,56]

QuiclSort(alist,0,len(alist)-1)
print(alist)