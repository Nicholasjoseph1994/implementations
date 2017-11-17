from collections import deque
import numpy as np
def bfs(start, is_goal, neighbors):
    """
    Performs breadth-first search on the graph starting from start
    and returning True if a goal is reachable and False otherwise

    ARGS:
        start (node) - This is the starting node
        neighbors - (node -> List[node]) - This function takes a node and returns all of its neighbors
        is_goal - (node -> Bool) - This function returns whether a node is a goal node    

    RETURNS:
        (bool) - Whether a goal node is reachable from the start node
    """

    # Initialize the frontier with the start node
    frontier = deque([start])

    # Initialize the explored set as empty
    explored = set()

    while frontier:
        # Pop a node from the queue
        current_node = frontier.popleft()

        # Check if we have reached a goal
        if is_goal(current_node):
            return True

        # Mark the current node as explored
        explored.add(current_node)

        # Add the neighbors to the end of the queue
        for neighbor in neighbors(current_node):
            if neighbor not in explored:
                frontier.append(neighbor)
    return False

def test_bfs():
    # Test 1: Searching through a grid
    grid1 = np.array([
            ['.', '.', '.', '.'],
            ['W', 'W', '.', '.'],
            ['.', '.', '.', 'W'],
            ['.', 'G', 'W', 'G']])
    grid2 = np.array([
            ['.', '.', '.', '.'],
            ['W', 'W', 'W', '.'],
            ['.', '.', 'W', 'W'],
            ['.', 'G', 'W', 'G']])
    start = (0, 0)
    def neighbor(graph, node):
        def inrange(x, low, high):
            return x >= low and x < high
        r, c = node
        options = [
             (r-1, c-1), (r-1, c), (r-1, c+1),
             (r, c-1),             (r, c+1),
             (r+1, c-1), (r+1, c), (r+1, c+1)]
        return [(r, c) for r, c in options
                        if inrange(r, 0, graph.shape[0])
                        and inrange(c, 0, graph.shape[1])
                        and graph[r][c] != 'W']
    assert bfs(start, lambda coords: grid1[coords] == 'G', lambda node: neighbor(grid1, node))
    assert not bfs(start, lambda coords: grid2[coords] == 'G', lambda node: neighbor(grid2, node))
    print 'Tests passed'
test_bfs()