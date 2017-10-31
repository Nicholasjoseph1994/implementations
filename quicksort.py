import sys
import numpy as np
def quicksort(array, left = 0, right = None):
    """
    This sorts array in place using the QuickSort algorithm

    ARGS:
        array (List[int]) - an unsorted array of ints

    RETURNS:
        None - sorts in place
    """
    def choose_pivot(array, left, right, method = 'RANDOM'):
        if method == 'RANDOM':
            return np.random.randint(left, right+1)
    def partition(array, pivot, left, right):
        def swap(array, i, j):
            tmp = array[i]
            array[i] = array[j]
            array[j] = tmp

        # Move pivot to first element
        swap(array, left, pivot)

        # i is the first element greater than the pivot
        i = left + 1

        # Run j through the list comparing each element to the pivot
        for j in range(left + 1, right + 1):
            # If it is less than the pivot, move to the left section and increase size of that by 1
            if array[j] < array[pivot]:
                swap(array, i, j)
                i += 1

        # Move the pivot into position
        swap(array, left, i - 1)


    if right is None:
        right = len(array) - 1

    # Base case:
    if right - left <= 0:
        return

    pivot = choose_pivot(array, left, right)
    partition(array, pivot, left, right)
    quicksort(array, left, pivot - 1)
    quicksort(array, pivot + 1, right)



def test_sort():
    tests = [[], [1, 2, 3], [3, 2, 1], [2, 3, 1]]
    answers = [[]] + [[1,2,3]]*3
    for test, ans in zip(tests, answers):
        quicksort(test)
        assert test == ans

    
    print 'Tests passed'
test_sort()