import sys
def inversions(array):
    """
    This returns the number of inversions in the array,
    where an inversion is defined as any pair of indices, (i, j)
    such that array[i] < array[j], but i > j.

    ARGS:
        array (List[int]) - an unsorted array of ints

    RETURNS:
        (int) - the number of inversions in array
    """
    def sort_and_count(arr):
        """
        Subroutine that sorts an array and counts the inversions
        """
        if len(arr) == 0:
            return ([], 0)
        if len(arr) == 1:
            return (arr, 0)
        
        mid = len(arr) / 2
        
        # Find inversions in left and right half
        l_sort, l_inversions = sort_and_count(arr[:mid])
        r_sort, r_inversions = sort_and_count(arr[mid:])

        # Find the cross inversions
        l_index = 0
        r_index = 0
        sorted_list = []
        cross_inversions = 0

        # Iterate merge auntil both indices have reached the end
        while l_index < len(l_sort) or r_index < len(r_sort):
            # If an index is past the end of the array, assign it max int
            l_val = l_sort[l_index] if l_index < len(l_sort) else sys.maxint
            r_val = r_sort[r_index] if r_index < len(r_sort) else sys.maxint
            
            # Append the lower of the two.
            if l_val < r_val:
                sorted_list.append(l_val)
                l_index += 1
            else:
                sorted_list.append(r_val)
                r_index += 1  
                # If appending from the right, add all remaining elements in the left that are larger
                cross_inversions += mid - l_index
        total_inversions = l_inversions + r_inversions + cross_inversions
        return (sorted_list, total_inversions) 

    # Return the count of inversions
    return sort_and_count(array)[1]

def test_inversions():
    assert inversions([]) == 0
    assert inversions([1]) == 0
    assert inversions([2, 1]) == 1
    assert inversions(range(100)) == 0
    assert inversions([3, 1, 2]) == 2
    with open('integers.txt', 'r') as f:
        text = f.read()
        nums = map(int, text.split('\n'))
        assert inversions(nums) == 2407905288
    print 'Tests passed'
test_inversions()