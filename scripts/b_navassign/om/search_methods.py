
from math import inf, sqrt
import numpy as np
from heapq import *

class Node():
   
    def __init__(self, parent = None, position = None):
        
        self.parent = parent
        self.position = position
        self.g = 0
        self.h = 0
        self.f = 0
    
    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.g < other.g


def ASTAR(graph, start, end, rtype = 'pathAndCost'):
    
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0  # in reality, H cost of start node will not be 0
    
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0
    
    open_list = []             # For PERFORMANCE_2
    # closed_list = []           For PERFORMANCE
    closed_list = set()        # For PERFORMANCE
    
    distance_array = np.empty(shape=(graph.shape[0], graph.shape[1]))
    distance_array.fill(inf)

    
    # open_list.append(start_node)                     # For PERFORMANCE_2
    heappush(open_list, [start_node.f, start_node])    # priority, element
    distance_array[start_node.position[0]][start_node.position[1]] = start_node.g

    
    while len(open_list) > 0:
        
        # Getting 1st node from the list and then comparing with full list to obtain the min F node
        
        current_node = open_list[0][1]
        
        
        # closed_list.append(current_node)            For PERFORMANCE
        
        
        # If the ejected node is the goal node
        if current_node == end_node:   # comparison will be based on 'position'
            if rtype == 'costOnly':
                return current_node.g, len(closed_list)
            elif rtype == 'pathAndCost':
                path = []
                current = current_node
                while current is not None:
                    path.append(current.position)
                    current = current.parent
                # return path[::-1]  aks 
                return path[::-1], current_node.g, len(closed_list)


        heappop(open_list)

        if current_node.position in closed_list:
            continue


        closed_list.add(current_node.position)      
        
        for move in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]:  # motion primitives
            node_position = (current_node.position[0] + move[0], current_node.position[1] + move[1])
            
            if node_position[0] > (len(graph) - 1) or node_position[0] < 0 or \
                node_position[1] > (len(graph[0]) - 1) or node_position[1] < 0:  
                       
                continue
            
            if graph[node_position[0]][node_position[1]] != 0:    # i.e. if obstacle is present, then it is not a valid cell
                continue
            
            
            # New change 527 - avoiding diagonal movement if the sideway cell(s) have obstacles

            if move == (1, 1):

                if (graph[current_node.position[0] + 1][current_node.position[1] + 0] != 0) or \
                    (graph[current_node.position[0] + 0][current_node.position[1] + 1] != 0):    
                    continue
            
            if move == (1, -1):

                if (graph[current_node.position[0] + 1][current_node.position[1] + 0] != 0) or \
                    (graph[current_node.position[0] + 0][current_node.position[1] - 1] != 0):    
                    continue

            if move == (-1, 1):

                if (graph[current_node.position[0] + 0][current_node.position[1] + 1] != 0) or \
                    (graph[current_node.position[0] - 1][current_node.position[1] + 0] != 0):    
                    continue

            if move == (-1, -1):

                if (graph[current_node.position[0] + 0][current_node.position[1] - 1] != 0) or \
                    (graph[current_node.position[0] - 1][current_node.position[1] + 0] != 0):    
                    continue

            # New change 527 ends


            if node_position in closed_list:  
                continue  

            

            new_node = Node(current_node, node_position)
        
        
        

            if move == (-1, -1) or move == (-1, 1) or move == (1, -1) or move == (1, 1):  # Diagonal movements
                new_node.g = current_node.g + 1.5
            else:
                new_node.g = current_node.g + 1     
            

            
            if new_node.g < distance_array[new_node.position[0]][new_node.position[1]]:

                squared_hcost = ((new_node.position[0] - end_node.position[0]) ** 2) + (
                    (new_node.position[1] - end_node.position[1]) ** 2)
                # new_node.h = float(format(sqrt(squared_hcost), '.2f'))
                new_node.h = sqrt(squared_hcost)

                new_node.f = new_node.g + new_node.h
            
            
                heappush(open_list, [new_node.f, new_node])    # priority, element 
                distance_array[new_node.position[0]][new_node.position[1]] = new_node.g


    
    # When no path is found
    if rtype == 'costOnly':
        return inf, len(closed_list)
    elif rtype == 'pathAndCost':
        return [], inf, len(closed_list)



# FRAstar


def FRASTAR(graph, start, end, open_list, closed_list, distance_array, rtype = 'pathAndCost'):
    
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0  # in reality, H cost of start node will not be 0, but it doesn't matter for it.
    
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0
    
    
    # If the goal node is already in CL, we just need to pull out the path and return

    # end_node_in_CL = False
            
    if end in closed_list:
        # return 
        path = []
        current = closed_list[end]
        end_node = closed_list[end]
        while current is not None:
            path.append(current.position)
            current = current.parent
        # return path[::-1]  aks 
        return path[::-1], end_node.g, open_list, closed_list, distance_array
    
    
    
    # for closed_node in closed_list:
    #     if closed_node == end_node:
    #         end_node_in_CL = True
    #         end_node = closed_node
    #         break

    # if end_node_in_CL == True:
    #     # return 
    #     path = []
    #     current = end_node
    #     while current is not None:
    #         path.append(current.position)
    #         current = current.parent
    #     # return path[::-1]  aks 
    #     return path[::-1], end_node.g, open_list, closed_list
    

    # --------------------------------------------------------------------------------

    # Now, if goal node is not in CL, then we need to find the path. We check if we already have something in OL to reuse.
    # If yes, we reuse the already available data in OL, otherwise we start normal A-star search from the start node to goal node.

    if len(open_list) > 0:
        for open_node in open_list:
            
            squared_hcost = ((open_node[1].position[0] - end_node.position[0]) ** 2) + (
                        (open_node[1].position[1] - end_node.position[1]) ** 2)
            # open_node[1].h = float(format(sqrt(squared_hcost), '.2f'))
            open_node[1].h = sqrt(squared_hcost)

            open_node[1].f = open_node[1].g + open_node[1].h
            open_node[0] = open_node[1].f


        heapify(open_list)

    else:

        heappush(open_list, [start_node.f, start_node])    # priority, element
        distance_array[start_node.position[0]][start_node.position[1]] = start_node.g

    # --------------------------------------------------------------------------------

    while len(open_list) > 0:
        
        # Getting 1st node from the list and then comparing with full list to obtain the min F node
        
        current_node = open_list[0][1]
        
        
        # closed_list.add(current_node.position)      # For PERFORMANCE
        
        
        # If the ejected node is the goal node
        if current_node == end_node:   # comparison will be based on 'position'
            if rtype == 'costOnly':
                return current_node.g, open_list, closed_list, distance_array
            elif rtype == 'pathAndCost':
                path = []
                current = current_node
                while current is not None:
                    path.append(current.position)
                    current = current.parent
                # return path[::-1]  aks 
                return path[::-1], current_node.g, open_list, closed_list, distance_array


        
        # current_node = heappop(open_list)[1]
        heappop(open_list)
        
        if current_node.position in closed_list:
            continue

        # closed_list.append(current_node)
        closed_list[current_node.position] = current_node
        
        
        for move in [(0, -1), (0, 1), (-1, 0), (1, 0)]:  # 4-directional only (matches BFS cost model)
            node_position = (current_node.position[0] + move[0], current_node.position[1] + move[1])

            if node_position[0] > (len(graph) - 1) or node_position[0] < 0 or \
                node_position[1] > (len(graph[0]) - 1) or node_position[1] < 0:

                continue

            if graph[node_position[0]][node_position[1]] != 0:    # i.e. if obstacle is present, then it is not a valid cell
                continue

            if node_position in closed_list:
                continue


            new_node = Node(current_node, node_position)

            new_node.g = current_node.g + 1
            
            
            if new_node.g < distance_array[new_node.position[0]][new_node.position[1]]:
            
                squared_hcost = ((new_node.position[0] - end_node.position[0]) ** 2) + (
                        (new_node.position[1] - end_node.position[1]) ** 2)
                # new_node.h = float(format(sqrt(squared_hcost), '.2f'))
                new_node.h = sqrt(squared_hcost)

                new_node.f = new_node.g + new_node.h
                

                heappush(open_list, [new_node.f, new_node])    # priority, element
                distance_array[new_node.position[0]][new_node.position[1]] = new_node.g

        
        



    
    # When no path is found
    if rtype == 'costOnly':
        return inf, open_list, closed_list, distance_array
    elif rtype == 'pathAndCost':
        return [], inf, open_list, closed_list, distance_array


# FRAstar  ends

# New function for baseline -- DIJKSTRA's algorithm

def DIJKSTRA_2D(graph, start, all_goal_loc, h_cost, rtype = 'pathAndCost'):
    
    start_node = Node(None, start)
    start_node.g = 0
    start_node.h = h_cost
    start_node.f = start_node.g + start_node.h
    
    # end_node = Node(None, end)
    # end_node.g = end_node.h = end_node.f = 0
    
    open_list = []             # For PERFORMANCE_2
    # closed_list = []           For PERFORMANCE
    closed_list = set()        # For PERFORMANCE
    
    cost_dict  = {}  # for returning result
    path_dict  = {}  # for returning result
    # goal_nodes = set()
    goal_nodes = []

    
    # initialization
    for ps in all_goal_loc:
        cost_dict[ps] = inf
        path_dict[ps] = []



    distance_array = np.empty(shape=(graph.shape[0], graph.shape[1]))
    distance_array.fill(inf)

    # open_list.append(start_node)                     # For PERFORMANCE_2
    heappush(open_list, [start_node.f, start_node])    # priority, element
    distance_array[start_node.position[0]][start_node.position[1]] = start_node.g

    while len(open_list) > 0:
        
        # Getting 1st node from the list and then comparing with full list to obtain the min F node
        
        # current_node = heappop(open_list)[1]
        current_node = open_list[0][1]
        
        
        # closed_list.append(current_node)            For PERFORMANCE
        
        
        
        
        if current_node.position in all_goal_loc:
            cost_dict[current_node.position] = current_node.g
            goal_nodes.append(current_node)
            all_goal_loc.discard(current_node.position)


        if not all_goal_loc:
            # return
            if rtype == 'costOnly':
                return cost_dict
            elif rtype == 'pathAndCost':
                for goal_node in goal_nodes:
                    path = []
                    current = goal_node
                    while current is not None:
                        path.append(current.position)
                        current = current.parent
                    path_dict[goal_node.position] = path[::-1]
                return path_dict, cost_dict, len(closed_list)



        # # If the ejected node is the goal node
        # if current_node == end_node:   # comparison will be based on 'position'
        #     if rtype == 'costOnly':
        #         return current_node.f
        #     elif rtype == 'pathAndCost':
        #         path = []
        #         current = current_node
        #         while current is not None:
        #             path.append(current.position)
        #             current = current.parent
        #         # return path[::-1]  aks 
        #         return path[::-1], current_node.f


        heappop(open_list)

        if current_node.position in closed_list:
            continue

        closed_list.add(current_node.position)      # For PERFORMANCE
        
        for move in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]:  # motion primitives
            node_position = (current_node.position[0] + move[0], current_node.position[1] + move[1])
            
            if node_position[0] > (len(graph) - 1) or node_position[0] < 0 or \
                node_position[1] > (len(graph[0]) - 1) or node_position[1] < 0:  
                       
                continue
            
            if graph[node_position[0]][node_position[1]] != 0:    # i.e. if obstacle is present, then it is not a valid cell
                continue
            
            
            # New change 527 - avoiding diagonal movement if the sideway cell(s) have obstacles

            if move == (1, 1):

                if (graph[current_node.position[0] + 1][current_node.position[1] + 0] != 0) or \
                    (graph[current_node.position[0] + 0][current_node.position[1] + 1] != 0):    
                    continue
            
            if move == (1, -1):

                if (graph[current_node.position[0] + 1][current_node.position[1] + 0] != 0) or \
                    (graph[current_node.position[0] + 0][current_node.position[1] - 1] != 0):    
                    continue

            if move == (-1, 1):

                if (graph[current_node.position[0] + 0][current_node.position[1] + 1] != 0) or \
                    (graph[current_node.position[0] - 1][current_node.position[1] + 0] != 0):    
                    continue

            if move == (-1, -1):

                if (graph[current_node.position[0] + 0][current_node.position[1] - 1] != 0) or \
                    (graph[current_node.position[0] - 1][current_node.position[1] + 0] != 0):    
                    continue

            # New change 527 ends


            if node_position in closed_list:  # For PERFORMANCE
                continue  

            

            new_node = Node(current_node, node_position)
        
        
        

            if move == (-1, -1) or move == (-1, 1) or move == (1, -1) or move == (1, 1):  # Diagonal movements
                new_node.g = current_node.g + 1.5
            else:
                new_node.g = current_node.g + 1     
            
            # squared_hcost = ((new_node.position[0] - end_node.position[0]) ** 2) + (
            #         (new_node.position[1] - end_node.position[1]) ** 2)
            # new_node.h = float(format(sqrt(squared_hcost), '.2f'))
            
            if new_node.g < distance_array[new_node.position[0]][new_node.position[1]]:
            
                new_node.h = h_cost

                new_node.f = new_node.g + new_node.h
            
                heappush(open_list, [new_node.f, new_node])    # priority, element 
                distance_array[new_node.position[0]][new_node.position[1]] = new_node.g

    
    # When graph nodes get exhausted
    if rtype == 'costOnly':
        return cost_dict
    elif rtype == 'pathAndCost':
        # return path_dict, cost_dict, len(closed_list)
        for goal_node in goal_nodes:
            path = []
            current = goal_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            path_dict[goal_node.position] = path[::-1]
        return path_dict, cost_dict, len(closed_list)







def ASTAR_3D(graph, start, end, rtype = 'pathAndCost'):
    
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0  # in reality, H cost of start node will not be 0
    
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0
    
    open_list = []             # For PERFORMANCE_2
    # closed_list = []           For PERFORMANCE
    closed_list = set()        # For PERFORMANCE
    
    distance_array = np.empty(shape=(graph.shape[0], graph.shape[1], graph.shape[2]))
    distance_array.fill(inf)

    
    # open_list.append(start_node)                     # For PERFORMANCE_2
    heappush(open_list, [start_node.f, start_node])    # priority, element
    distance_array[start_node.position[0]][start_node.position[1]][start_node.position[2]] = start_node.g
    
    while len(open_list) > 0:
        
        # Getting 1st node from the list and then comparing with full list to obtain the min F node
        
        # current_node = heappop(open_list)[1]
        current_node = open_list[0][1]
        
        
        # closed_list.append(current_node)            For PERFORMANCE
        
        
        
        # If the ejected node is the goal node
        if current_node == end_node:   # comparison will be based on 'position'
            if rtype == 'costOnly':
                return current_node.g, len(closed_list)
            elif rtype == 'pathAndCost':
                path = []
                current = current_node
                while current is not None:
                    path.append(current.position)
                    current = current.parent
                # return path[::-1]  aks 
                return path[::-1], current_node.g, len(closed_list)


        heappop(open_list)

        if current_node.position in closed_list:
            continue


        closed_list.add(current_node.position)      # For PERFORMANCE

        # for move in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]:  # motion primitives
        for move in [(0, 0, -1), (0, 0, 1), (0, -1, 0), (0, 1, 0), (0, -1, -1), (0, -1, 1), (0, 1, -1), (0, 1, 1),
                     (1, 0, -1), (1, 0, 1), (1, -1, 0), (1, 1, 0), (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1),
                     (-1, 0, -1), (-1, 0, 1), (-1, -1, 0), (-1, 1, 0), (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
                     (-1, 0, 0), (1, 0, 0) ]:  # motion primitives

            node_position = (current_node.position[0] + move[0], current_node.position[1] + move[1], current_node.position[2] + move[2])
            
            if node_position[0] > (len(graph) - 1) or node_position[0] < 0 or \
                node_position[1] > (len(graph[0]) - 1) or node_position[1] < 0 or \
                node_position[2] > (len(graph[0][0]) - 1) or node_position[2] < 0:
                       
                continue
            
            if graph[node_position[0]][node_position[1]][node_position[2]] != 0:    # i.e. if obstacle is present, then it is not a valid cell
                continue
            
            
            # New change 527 - avoiding diagonal movement if the sideway cell(s) have obstacles

            # if move == (1, 1):
            if move == (0, -1, -1):

                if (graph[current_node.position[0] + 0][current_node.position[1] + 0][current_node.position[2] - 1] != 0) or \
                    (graph[current_node.position[0] + 0][current_node.position[1] - 1][current_node.position[2] + 0] != 0):    
                    continue
            
            
            elif move == (0, -1, 1):

                if (graph[current_node.position[0] + 0][current_node.position[1] + 0][current_node.position[2] + 1] != 0) or \
                    (graph[current_node.position[0] + 0][current_node.position[1] - 1][current_node.position[2] + 0] != 0):    
                    continue
            

            elif move == (0, 1, -1):

                if (graph[current_node.position[0] + 0][current_node.position[1] + 0][current_node.position[2] - 1] != 0) or \
                    (graph[current_node.position[0] + 0][current_node.position[1] + 1][current_node.position[2] + 0] != 0):    
                    continue
            
            
            elif move == (0, 1, 1):

                if (graph[current_node.position[0] + 0][current_node.position[1] + 0][current_node.position[2] + 1] != 0) or \
                    (graph[current_node.position[0] + 0][current_node.position[1] + 1][current_node.position[2] + 0] != 0):    
                    continue
            
            
            elif move == (1, 0, -1):

                if (graph[current_node.position[0] + 0][current_node.position[1] + 0][current_node.position[2] - 1] != 0) or \
                    (graph[current_node.position[0] + 1][current_node.position[1] + 0][current_node.position[2] + 0] != 0):    
                    continue
            
            
            elif move == (1, 0, 1):

                if (graph[current_node.position[0] + 0][current_node.position[1] + 0][current_node.position[2] + 1] != 0) or \
                    (graph[current_node.position[0] + 1][current_node.position[1] + 0][current_node.position[2] + 0] != 0):    
                    continue

            
            elif move == (1, -1, 0):

                if (graph[current_node.position[0] + 0][current_node.position[1] - 1][current_node.position[2] + 0] != 0) or \
                    (graph[current_node.position[0] + 1][current_node.position[1] + 0][current_node.position[2] + 0] != 0):    
                    continue

            
            elif move == (1, 1, 0):

                if (graph[current_node.position[0] + 0][current_node.position[1] + 1][current_node.position[2] + 0] != 0) or \
                    (graph[current_node.position[0] + 1][current_node.position[1] + 0][current_node.position[2] + 0] != 0):    
                    continue


            elif move == (-1, 0, -1):

                if (graph[current_node.position[0] - 1][current_node.position[1] + 0][current_node.position[2] + 0] != 0) or \
                    (graph[current_node.position[0] + 0][current_node.position[1] + 0][current_node.position[2] - 1] != 0):    
                    continue


            elif move == (-1, 0, 1):

                if (graph[current_node.position[0] - 1][current_node.position[1] + 0][current_node.position[2] + 0] != 0) or \
                    (graph[current_node.position[0] + 0][current_node.position[1] + 0][current_node.position[2] + 1] != 0):    
                    continue


            elif move == (-1, -1, 0):

                if (graph[current_node.position[0] - 1][current_node.position[1] + 0][current_node.position[2] + 0] != 0) or \
                    (graph[current_node.position[0] + 0][current_node.position[1] - 1][current_node.position[2] + 0] != 0):    
                    continue


            elif move == (-1, 1, 0):

                if (graph[current_node.position[0] - 1][current_node.position[1] + 0][current_node.position[2] + 0] != 0) or \
                    (graph[current_node.position[0] + 0][current_node.position[1] + 1][current_node.position[2] + 0] != 0):    
                    continue

            # 3-coordinate movement check

            elif move == (1, -1, -1):

                if (graph[current_node.position[0] + 1][current_node.position[1] + 0][current_node.position[2] + 0] != 0) or \
                    (graph[current_node.position[0] + 0][current_node.position[1] - 1][current_node.position[2] + 0] != 0) or \
                        (graph[current_node.position[0] + 0][current_node.position[1] + 0][current_node.position[2] - 1] != 0) or \
                            (graph[current_node.position[0] + 0][current_node.position[1] - 1][current_node.position[2] - 1] != 0) or \
                                (graph[current_node.position[0] + 1][current_node.position[1] + 0][current_node.position[2] - 1] != 0) or \
                                    (graph[current_node.position[0] + 1][current_node.position[1] - 1][current_node.position[2] + 0] != 0):
                    continue


            elif move == (1, -1, 1):

                if (graph[current_node.position[0] + 1][current_node.position[1] + 0][current_node.position[2] + 0] != 0) or \
                    (graph[current_node.position[0] + 0][current_node.position[1] - 1][current_node.position[2] + 0] != 0) or \
                        (graph[current_node.position[0] + 0][current_node.position[1] + 0][current_node.position[2] + 1] != 0) or \
                            (graph[current_node.position[0] + 0][current_node.position[1] - 1][current_node.position[2] + 1] != 0) or \
                                (graph[current_node.position[0] + 1][current_node.position[1] + 0][current_node.position[2] + 1] != 0) or \
                                    (graph[current_node.position[0] + 1][current_node.position[1] - 1][current_node.position[2] + 0] != 0):
                    continue


            elif move == (1, 1, -1):

                if (graph[current_node.position[0] + 1][current_node.position[1] + 0][current_node.position[2] + 0] != 0) or \
                    (graph[current_node.position[0] + 0][current_node.position[1] + 1][current_node.position[2] + 0] != 0) or \
                        (graph[current_node.position[0] + 0][current_node.position[1] + 0][current_node.position[2] - 1] != 0) or \
                            (graph[current_node.position[0] + 0][current_node.position[1] + 1][current_node.position[2] - 1] != 0) or \
                                (graph[current_node.position[0] + 1][current_node.position[1] + 0][current_node.position[2] - 1] != 0) or \
                                    (graph[current_node.position[0] + 1][current_node.position[1] + 1][current_node.position[2] + 0] != 0):
                    continue


            elif move == (1, 1, 1):

                if (graph[current_node.position[0] + 1][current_node.position[1] + 0][current_node.position[2] + 0] != 0) or \
                    (graph[current_node.position[0] + 0][current_node.position[1] + 1][current_node.position[2] + 0] != 0) or \
                        (graph[current_node.position[0] + 0][current_node.position[1] + 0][current_node.position[2] + 1] != 0) or \
                            (graph[current_node.position[0] + 0][current_node.position[1] + 1][current_node.position[2] + 1] != 0) or \
                                (graph[current_node.position[0] + 1][current_node.position[1] + 0][current_node.position[2] + 1] != 0) or \
                                    (graph[current_node.position[0] + 1][current_node.position[1] + 1][current_node.position[2] + 0] != 0):
                    continue


            elif move == (-1, -1, -1):

                if (graph[current_node.position[0] - 1][current_node.position[1] + 0][current_node.position[2] + 0] != 0) or \
                    (graph[current_node.position[0] + 0][current_node.position[1] - 1][current_node.position[2] + 0] != 0) or \
                        (graph[current_node.position[0] + 0][current_node.position[1] + 0][current_node.position[2] - 1] != 0) or \
                            (graph[current_node.position[0] + 0][current_node.position[1] - 1][current_node.position[2] - 1] != 0) or \
                                (graph[current_node.position[0] - 1][current_node.position[1] + 0][current_node.position[2] - 1] != 0) or \
                                    (graph[current_node.position[0] - 1][current_node.position[1] - 1][current_node.position[2] + 0] != 0):
                    continue


            elif move == (-1, -1, 1):

                if (graph[current_node.position[0] - 1][current_node.position[1] + 0][current_node.position[2] + 0] != 0) or \
                    (graph[current_node.position[0] + 0][current_node.position[1] - 1][current_node.position[2] + 0] != 0) or \
                        (graph[current_node.position[0] + 0][current_node.position[1] + 0][current_node.position[2] + 1] != 0) or \
                            (graph[current_node.position[0] + 0][current_node.position[1] - 1][current_node.position[2] + 1] != 0) or \
                                (graph[current_node.position[0] - 1][current_node.position[1] + 0][current_node.position[2] + 1] != 0) or \
                                    (graph[current_node.position[0] - 1][current_node.position[1] - 1][current_node.position[2] + 0] != 0):
                    continue


            elif move == (-1, 1, -1):

                if (graph[current_node.position[0] - 1][current_node.position[1] + 0][current_node.position[2] + 0] != 0) or \
                    (graph[current_node.position[0] + 0][current_node.position[1] + 1][current_node.position[2] + 0] != 0) or \
                        (graph[current_node.position[0] + 0][current_node.position[1] + 0][current_node.position[2] - 1] != 0) or \
                            (graph[current_node.position[0] + 0][current_node.position[1] + 1][current_node.position[2] - 1] != 0) or \
                                (graph[current_node.position[0] - 1][current_node.position[1] + 0][current_node.position[2] - 1] != 0) or \
                                    (graph[current_node.position[0] - 1][current_node.position[1] + 1][current_node.position[2] + 0] != 0):
                    continue


            elif move == (-1, 1, 1):

                if (graph[current_node.position[0] - 1][current_node.position[1] + 0][current_node.position[2] + 0] != 0) or \
                    (graph[current_node.position[0] + 0][current_node.position[1] + 1][current_node.position[2] + 0] != 0) or \
                        (graph[current_node.position[0] + 0][current_node.position[1] + 0][current_node.position[2] + 1] != 0) or \
                            (graph[current_node.position[0] + 0][current_node.position[1] + 1][current_node.position[2] + 1] != 0) or \
                                (graph[current_node.position[0] - 1][current_node.position[1] + 0][current_node.position[2] + 1] != 0) or \
                                    (graph[current_node.position[0] - 1][current_node.position[1] + 1][current_node.position[2] + 0] != 0):
                    continue

            # New change 527 ends


            if node_position in closed_list:  # For PERFORMANCE
                continue  

            

            new_node = Node(current_node, node_position)
        
        
        

            # if move == (-1, -1) or move == (-1, 1) or move == (1, -1) or move == (1, 1):  # Diagonal movements
            if move == (0, -1, -1) or move == (0, -1, 1) or move == (0, 1, -1) or move == (0, 1, 1) or \
                move == (1, 0, -1) or move == (1, 0, 1) or move == (1, -1, 0) or move == (1, 1, 0) or \
                    move == (-1, 0, -1) or move == (-1, 0, 1) or move == (-1, -1, 0) or move == (-1, 1, 0):
                new_node.g = current_node.g + 1.5
            elif move == (1, -1, -1) or move == (1, -1, 1) or move == (1, 1, -1) or move == (1, 1, 1) or \
                move == (-1, -1, -1) or move == (-1, -1, 1) or move == (-1, 1, -1) or move == (-1, 1, 1):
                new_node.g = current_node.g + 2
            else:
                new_node.g = current_node.g + 1     
            
            
            
            if new_node.g < distance_array[new_node.position[0]][new_node.position[1]][new_node.position[2]]:
            
                squared_hcost = ((new_node.position[0] - end_node.position[0]) ** 2) + (
                    (new_node.position[1] - end_node.position[1]) ** 2) + (
                        (new_node.position[2] - end_node.position[2]) ** 2)
                # new_node.h = float(format(sqrt(squared_hcost), '.2f'))
                new_node.h = sqrt(squared_hcost)

                new_node.f = new_node.g + new_node.h
            
                heappush(open_list, [new_node.f, new_node])    # priority, element 
                distance_array[new_node.position[0]][new_node.position[1]][new_node.position[2]] = new_node.g


    
    # When no path is found
    if rtype == 'costOnly':
        return inf, len(closed_list)
    elif rtype == 'pathAndCost':
        return [], inf, len(closed_list)



def DIJKSTRA_3D(graph, start, all_goal_loc, h_cost, rtype = 'pathAndCost'):
    
    start_node = Node(None, start)
    start_node.g = 0
    start_node.h = h_cost
    start_node.f = start_node.g + start_node.h
    
    # end_node = Node(None, end)
    # end_node.g = end_node.h = end_node.f = 0
    
    open_list = []             # For PERFORMANCE_2
    # closed_list = []           For PERFORMANCE
    closed_list = set()        # For PERFORMANCE

    cost_dict  = {}  # for returning result
    path_dict  = {}  # for returning result
    # goal_nodes = set()
    goal_nodes = []


    # initialization
    for ps in all_goal_loc:
        cost_dict[ps] = inf
        path_dict[ps] = []

    
    distance_array = np.empty(shape=(graph.shape[0], graph.shape[1], graph.shape[2]))
    distance_array.fill(inf)

    
    # open_list.append(start_node)                     # For PERFORMANCE_2
    heappush(open_list, [start_node.f, start_node])    # priority, element
    distance_array[start_node.position[0]][start_node.position[1]][start_node.position[2]] = start_node.g
    
    while len(open_list) > 0:
        
        # Getting 1st node from the list and then comparing with full list to obtain the min F node
        
        # current_node = heappop(open_list)[1]
        current_node = open_list[0][1]
        
        
        # closed_list.append(current_node)            For PERFORMANCE
        
        
        
        if current_node.position in all_goal_loc:
            cost_dict[current_node.position] = current_node.g
            goal_nodes.append(current_node)
            all_goal_loc.discard(current_node.position)


        if not all_goal_loc:
            # return
            if rtype == 'costOnly':
                return cost_dict
            elif rtype == 'pathAndCost':
                for goal_node in goal_nodes:
                    path = []
                    current = goal_node
                    while current is not None:
                        path.append(current.position)
                        current = current.parent
                    path_dict[goal_node.position] = path[::-1]
                return path_dict, cost_dict, len(closed_list)



        # # If the ejected node is the goal node
        # if current_node == end_node:   # comparison will be based on 'position'
        #     if rtype == 'costOnly':
        #         return current_node.f
        #     elif rtype == 'pathAndCost':
        #         path = []
        #         current = current_node
        #         while current is not None:
        #             path.append(current.position)
        #             current = current.parent
        #         # return path[::-1]  aks 
        #         return path[::-1], current_node.f

        heappop(open_list)

        if current_node.position in closed_list:
            continue


        closed_list.add(current_node.position)      # For PERFORMANCE
        
        # for move in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]:  # motion primitives
        for move in [(0, 0, -1), (0, 0, 1), (0, -1, 0), (0, 1, 0), (0, -1, -1), (0, -1, 1), (0, 1, -1), (0, 1, 1),
                     (1, 0, -1), (1, 0, 1), (1, -1, 0), (1, 1, 0), (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1),
                     (-1, 0, -1), (-1, 0, 1), (-1, -1, 0), (-1, 1, 0), (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
                     (-1, 0, 0), (1, 0, 0) ]:  # motion primitives

            node_position = (current_node.position[0] + move[0], current_node.position[1] + move[1], current_node.position[2] + move[2])
            
            if node_position[0] > (len(graph) - 1) or node_position[0] < 0 or \
                node_position[1] > (len(graph[0]) - 1) or node_position[1] < 0 or \
                node_position[2] > (len(graph[0][0]) - 1) or node_position[2] < 0:
                       
                continue
            
            if graph[node_position[0]][node_position[1]][node_position[2]] != 0:    # i.e. if obstacle is present, then it is not a valid cell
                continue
            
            
            # New change 527 - avoiding diagonal movement if the sideway cell(s) have obstacles

            # if move == (1, 1):
            if move == (0, -1, -1):

                if (graph[current_node.position[0] + 0][current_node.position[1] + 0][current_node.position[2] - 1] != 0) or \
                    (graph[current_node.position[0] + 0][current_node.position[1] - 1][current_node.position[2] + 0] != 0):    
                    continue
            
            
            elif move == (0, -1, 1):

                if (graph[current_node.position[0] + 0][current_node.position[1] + 0][current_node.position[2] + 1] != 0) or \
                    (graph[current_node.position[0] + 0][current_node.position[1] - 1][current_node.position[2] + 0] != 0):    
                    continue
            

            elif move == (0, 1, -1):

                if (graph[current_node.position[0] + 0][current_node.position[1] + 0][current_node.position[2] - 1] != 0) or \
                    (graph[current_node.position[0] + 0][current_node.position[1] + 1][current_node.position[2] + 0] != 0):    
                    continue
            
            
            elif move == (0, 1, 1):

                if (graph[current_node.position[0] + 0][current_node.position[1] + 0][current_node.position[2] + 1] != 0) or \
                    (graph[current_node.position[0] + 0][current_node.position[1] + 1][current_node.position[2] + 0] != 0):    
                    continue
            
            
            elif move == (1, 0, -1):

                if (graph[current_node.position[0] + 0][current_node.position[1] + 0][current_node.position[2] - 1] != 0) or \
                    (graph[current_node.position[0] + 1][current_node.position[1] + 0][current_node.position[2] + 0] != 0):    
                    continue
            
            
            elif move == (1, 0, 1):

                if (graph[current_node.position[0] + 0][current_node.position[1] + 0][current_node.position[2] + 1] != 0) or \
                    (graph[current_node.position[0] + 1][current_node.position[1] + 0][current_node.position[2] + 0] != 0):    
                    continue

            
            elif move == (1, -1, 0):

                if (graph[current_node.position[0] + 0][current_node.position[1] - 1][current_node.position[2] + 0] != 0) or \
                    (graph[current_node.position[0] + 1][current_node.position[1] + 0][current_node.position[2] + 0] != 0):    
                    continue

            
            elif move == (1, 1, 0):

                if (graph[current_node.position[0] + 0][current_node.position[1] + 1][current_node.position[2] + 0] != 0) or \
                    (graph[current_node.position[0] + 1][current_node.position[1] + 0][current_node.position[2] + 0] != 0):    
                    continue


            elif move == (-1, 0, -1):

                if (graph[current_node.position[0] - 1][current_node.position[1] + 0][current_node.position[2] + 0] != 0) or \
                    (graph[current_node.position[0] + 0][current_node.position[1] + 0][current_node.position[2] - 1] != 0):    
                    continue


            elif move == (-1, 0, 1):

                if (graph[current_node.position[0] - 1][current_node.position[1] + 0][current_node.position[2] + 0] != 0) or \
                    (graph[current_node.position[0] + 0][current_node.position[1] + 0][current_node.position[2] + 1] != 0):    
                    continue


            elif move == (-1, -1, 0):

                if (graph[current_node.position[0] - 1][current_node.position[1] + 0][current_node.position[2] + 0] != 0) or \
                    (graph[current_node.position[0] + 0][current_node.position[1] - 1][current_node.position[2] + 0] != 0):    
                    continue


            elif move == (-1, 1, 0):

                if (graph[current_node.position[0] - 1][current_node.position[1] + 0][current_node.position[2] + 0] != 0) or \
                    (graph[current_node.position[0] + 0][current_node.position[1] + 1][current_node.position[2] + 0] != 0):    
                    continue

            # 3-coordinate movement check

            elif move == (1, -1, -1):

                if (graph[current_node.position[0] + 1][current_node.position[1] + 0][current_node.position[2] + 0] != 0) or \
                    (graph[current_node.position[0] + 0][current_node.position[1] - 1][current_node.position[2] + 0] != 0) or \
                        (graph[current_node.position[0] + 0][current_node.position[1] + 0][current_node.position[2] - 1] != 0) or \
                            (graph[current_node.position[0] + 0][current_node.position[1] - 1][current_node.position[2] - 1] != 0) or \
                                (graph[current_node.position[0] + 1][current_node.position[1] + 0][current_node.position[2] - 1] != 0) or \
                                    (graph[current_node.position[0] + 1][current_node.position[1] - 1][current_node.position[2] + 0] != 0):
                    continue


            elif move == (1, -1, 1):

                if (graph[current_node.position[0] + 1][current_node.position[1] + 0][current_node.position[2] + 0] != 0) or \
                    (graph[current_node.position[0] + 0][current_node.position[1] - 1][current_node.position[2] + 0] != 0) or \
                        (graph[current_node.position[0] + 0][current_node.position[1] + 0][current_node.position[2] + 1] != 0) or \
                            (graph[current_node.position[0] + 0][current_node.position[1] - 1][current_node.position[2] + 1] != 0) or \
                                (graph[current_node.position[0] + 1][current_node.position[1] + 0][current_node.position[2] + 1] != 0) or \
                                    (graph[current_node.position[0] + 1][current_node.position[1] - 1][current_node.position[2] + 0] != 0):
                    continue


            elif move == (1, 1, -1):

                if (graph[current_node.position[0] + 1][current_node.position[1] + 0][current_node.position[2] + 0] != 0) or \
                    (graph[current_node.position[0] + 0][current_node.position[1] + 1][current_node.position[2] + 0] != 0) or \
                        (graph[current_node.position[0] + 0][current_node.position[1] + 0][current_node.position[2] - 1] != 0) or \
                            (graph[current_node.position[0] + 0][current_node.position[1] + 1][current_node.position[2] - 1] != 0) or \
                                (graph[current_node.position[0] + 1][current_node.position[1] + 0][current_node.position[2] - 1] != 0) or \
                                    (graph[current_node.position[0] + 1][current_node.position[1] + 1][current_node.position[2] + 0] != 0):
                    continue


            elif move == (1, 1, 1):

                if (graph[current_node.position[0] + 1][current_node.position[1] + 0][current_node.position[2] + 0] != 0) or \
                    (graph[current_node.position[0] + 0][current_node.position[1] + 1][current_node.position[2] + 0] != 0) or \
                        (graph[current_node.position[0] + 0][current_node.position[1] + 0][current_node.position[2] + 1] != 0) or \
                            (graph[current_node.position[0] + 0][current_node.position[1] + 1][current_node.position[2] + 1] != 0) or \
                                (graph[current_node.position[0] + 1][current_node.position[1] + 0][current_node.position[2] + 1] != 0) or \
                                    (graph[current_node.position[0] + 1][current_node.position[1] + 1][current_node.position[2] + 0] != 0):
                    continue


            elif move == (-1, -1, -1):

                if (graph[current_node.position[0] - 1][current_node.position[1] + 0][current_node.position[2] + 0] != 0) or \
                    (graph[current_node.position[0] + 0][current_node.position[1] - 1][current_node.position[2] + 0] != 0) or \
                        (graph[current_node.position[0] + 0][current_node.position[1] + 0][current_node.position[2] - 1] != 0) or \
                            (graph[current_node.position[0] + 0][current_node.position[1] - 1][current_node.position[2] - 1] != 0) or \
                                (graph[current_node.position[0] - 1][current_node.position[1] + 0][current_node.position[2] - 1] != 0) or \
                                    (graph[current_node.position[0] - 1][current_node.position[1] - 1][current_node.position[2] + 0] != 0):
                    continue


            elif move == (-1, -1, 1):

                if (graph[current_node.position[0] - 1][current_node.position[1] + 0][current_node.position[2] + 0] != 0) or \
                    (graph[current_node.position[0] + 0][current_node.position[1] - 1][current_node.position[2] + 0] != 0) or \
                        (graph[current_node.position[0] + 0][current_node.position[1] + 0][current_node.position[2] + 1] != 0) or \
                            (graph[current_node.position[0] + 0][current_node.position[1] - 1][current_node.position[2] + 1] != 0) or \
                                (graph[current_node.position[0] - 1][current_node.position[1] + 0][current_node.position[2] + 1] != 0) or \
                                    (graph[current_node.position[0] - 1][current_node.position[1] - 1][current_node.position[2] + 0] != 0):
                    continue


            elif move == (-1, 1, -1):

                if (graph[current_node.position[0] - 1][current_node.position[1] + 0][current_node.position[2] + 0] != 0) or \
                    (graph[current_node.position[0] + 0][current_node.position[1] + 1][current_node.position[2] + 0] != 0) or \
                        (graph[current_node.position[0] + 0][current_node.position[1] + 0][current_node.position[2] - 1] != 0) or \
                            (graph[current_node.position[0] + 0][current_node.position[1] + 1][current_node.position[2] - 1] != 0) or \
                                (graph[current_node.position[0] - 1][current_node.position[1] + 0][current_node.position[2] - 1] != 0) or \
                                    (graph[current_node.position[0] - 1][current_node.position[1] + 1][current_node.position[2] + 0] != 0):
                    continue


            elif move == (-1, 1, 1):

                if (graph[current_node.position[0] - 1][current_node.position[1] + 0][current_node.position[2] + 0] != 0) or \
                    (graph[current_node.position[0] + 0][current_node.position[1] + 1][current_node.position[2] + 0] != 0) or \
                        (graph[current_node.position[0] + 0][current_node.position[1] + 0][current_node.position[2] + 1] != 0) or \
                            (graph[current_node.position[0] + 0][current_node.position[1] + 1][current_node.position[2] + 1] != 0) or \
                                (graph[current_node.position[0] - 1][current_node.position[1] + 0][current_node.position[2] + 1] != 0) or \
                                    (graph[current_node.position[0] - 1][current_node.position[1] + 1][current_node.position[2] + 0] != 0):
                    continue

            # New change 527 ends


            if node_position in closed_list:  # For PERFORMANCE
                continue  

            

            new_node = Node(current_node, node_position)
        
        
        

            # if move == (-1, -1) or move == (-1, 1) or move == (1, -1) or move == (1, 1):  # Diagonal movements
            if move == (0, -1, -1) or move == (0, -1, 1) or move == (0, 1, -1) or move == (0, 1, 1) or \
                move == (1, 0, -1) or move == (1, 0, 1) or move == (1, -1, 0) or move == (1, 1, 0) or \
                    move == (-1, 0, -1) or move == (-1, 0, 1) or move == (-1, -1, 0) or move == (-1, 1, 0):
                new_node.g = current_node.g + 1.5
            elif move == (1, -1, -1) or move == (1, -1, 1) or move == (1, 1, -1) or move == (1, 1, 1) or \
                move == (-1, -1, -1) or move == (-1, -1, 1) or move == (-1, 1, -1) or move == (-1, 1, 1):
                new_node.g = current_node.g + 2
            else:
                new_node.g = current_node.g + 1     
            
            # squared_hcost = ((new_node.position[0] - end_node.position[0]) ** 2) + (
            #         (new_node.position[1] - end_node.position[1]) ** 2) + (
            #             (new_node.position[2] - end_node.position[2]) ** 2)
            # new_node.h = float(format(sqrt(squared_hcost), '.2f'))
            # new_node.h = h_cost

            # new_node.f = new_node.g + new_node.h
            
            # We check whether a child is already in OL
            # If not, then simply add it in OL
            # If it is already in OL, then see of it's g value is more than child's g value
            # If yes, the update it's parent, g and f values with that of child's as child brings a better path

            # ind = 0
            # new_node_in_OL = False

            # for ind, heap_elem in enumerate(open_list):
            #     if new_node == heap_elem[1]:
            #         new_node_in_OL = True
            #         break
            
            
            # if new_node_in_OL == True:  # i.e. if child is already present in OL
            #     if (new_node.g < open_list[ind][1].g):     # open_list[ind] is heap's element, and open_list[ind][1] is Node obj
            #         open_list[ind][1].parent = new_node.parent
            #         open_list[ind][1].g = new_node.g
            #         open_list[ind][1].f = new_node.f

            #         open_list[ind][0] = new_node.f   # updating priority, i.e. f-value of heap element
            #         heapify(open_list)        
            # else:
            
            if new_node.g < distance_array[new_node.position[0]][new_node.position[1]][new_node.position[2]]:

                new_node.h = h_cost
                new_node.f = new_node.g + new_node.h

                heappush(open_list, [new_node.f, new_node])    # priority, element 
                distance_array[new_node.position[0]][new_node.position[1]][new_node.position[2]] = new_node.g

    
    # When no path is found
    if rtype == 'costOnly':
        # return inf
        return cost_dict
    elif rtype == 'pathAndCost':
        # return [], inf
        # return path_dict, cost_dict, len(closed_list)
        for goal_node in goal_nodes:
            path = []
            current = goal_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            path_dict[goal_node.position] = path[::-1]
        return path_dict, cost_dict, len(closed_list)


# FRASTAR 3D

def FRASTAR_3D(graph, start, end, open_list, closed_list, distance_array, rtype = 'pathAndCost'):
    
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0  # in reality, H cost of start node will not be 0
    
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0
    
    # open_list = []             # For PERFORMANCE_2
    # closed_list = []           For PERFORMANCE
    # closed_list = set()        # For PERFORMANCE
    
    if end in closed_list:
        # return 
        path = []
        current = closed_list[end]
        end_node = closed_list[end]
        while current is not None:
            path.append(current.position)
            current = current.parent
        # return path[::-1]  aks 
        return path[::-1], end_node.g, open_list, closed_list, distance_array
  
    
    # --------------------------------------------------------------------------------

    # Now, if goal node is not in CL, then we need to find the path. We check if we already have something in OL to reuse.
    # If yes, we reuse the already available data in OL, otherwise we start normal A-star search from the start node to goal node.

    if len(open_list) > 0:
        for open_node in open_list:
            
            squared_hcost = ((open_node[1].position[0] - end_node.position[0]) ** 2) + (
                        (open_node[1].position[1] - end_node.position[1]) ** 2) + (
                            (open_node[1].position[2] - end_node.position[2]) ** 2)
            # open_node[1].h = float(format(sqrt(squared_hcost), '.2f'))
            open_node[1].h = sqrt(squared_hcost)

            open_node[1].f = open_node[1].g + open_node[1].h
            open_node[0] = open_node[1].f
            


        heapify(open_list)

    else:

        heappush(open_list, [start_node.f, start_node])    # priority, element
        distance_array[start_node.position[0]][start_node.position[1]][start_node.position[2]] = start_node.g

    # --------------------------------------------------------------------------------
    
    while len(open_list) > 0:
        
        # Getting 1st node from the list and then comparing with full list to obtain the min F node
        
        # current_node = heappop(open_list)[1]
        current_node = open_list[0][1]
        
        
        
        # If the ejected node is the goal node
        if current_node == end_node:   # comparison will be based on 'position'
            if rtype == 'costOnly':
                return current_node.g, open_list, closed_list, distance_array
            elif rtype == 'pathAndCost':
                path = []
                current = current_node
                while current is not None:
                    path.append(current.position)
                    current = current.parent
                # return path[::-1]  aks 
                return path[::-1], current_node.g, open_list, closed_list, distance_array


        # current_node = heappop(open_list)[1]
        heappop(open_list)

        if current_node.position in closed_list:
            continue

                
        # closed_list.append(current_node)
        closed_list[current_node.position] = current_node
        
        
        # for move in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]:  # motion primitives
        for move in [(0, 0, -1), (0, 0, 1), (0, -1, 0), (0, 1, 0), (0, -1, -1), (0, -1, 1), (0, 1, -1), (0, 1, 1),
                     (1, 0, -1), (1, 0, 1), (1, -1, 0), (1, 1, 0), (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1),
                     (-1, 0, -1), (-1, 0, 1), (-1, -1, 0), (-1, 1, 0), (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
                     (-1, 0, 0), (1, 0, 0) ]:  # motion primitives

            node_position = (current_node.position[0] + move[0], current_node.position[1] + move[1], current_node.position[2] + move[2])
            
            if node_position[0] > (len(graph) - 1) or node_position[0] < 0 or \
                node_position[1] > (len(graph[0]) - 1) or node_position[1] < 0 or \
                node_position[2] > (len(graph[0][0]) - 1) or node_position[2] < 0:
                       
                continue
            
            if graph[node_position[0]][node_position[1]][node_position[2]] != 0:    # i.e. if obstacle is present, then it is not a valid cell
                continue
            
            
            # New change 527 - avoiding diagonal movement if the sideway cell(s) have obstacles

            # if move == (1, 1):
            if move == (0, -1, -1):

                if (graph[current_node.position[0] + 0][current_node.position[1] + 0][current_node.position[2] - 1] != 0) or \
                    (graph[current_node.position[0] + 0][current_node.position[1] - 1][current_node.position[2] + 0] != 0):    
                    continue
            
            
            elif move == (0, -1, 1):

                if (graph[current_node.position[0] + 0][current_node.position[1] + 0][current_node.position[2] + 1] != 0) or \
                    (graph[current_node.position[0] + 0][current_node.position[1] - 1][current_node.position[2] + 0] != 0):    
                    continue
            

            elif move == (0, 1, -1):

                if (graph[current_node.position[0] + 0][current_node.position[1] + 0][current_node.position[2] - 1] != 0) or \
                    (graph[current_node.position[0] + 0][current_node.position[1] + 1][current_node.position[2] + 0] != 0):    
                    continue
            
            
            elif move == (0, 1, 1):

                if (graph[current_node.position[0] + 0][current_node.position[1] + 0][current_node.position[2] + 1] != 0) or \
                    (graph[current_node.position[0] + 0][current_node.position[1] + 1][current_node.position[2] + 0] != 0):    
                    continue
            
            
            elif move == (1, 0, -1):

                if (graph[current_node.position[0] + 0][current_node.position[1] + 0][current_node.position[2] - 1] != 0) or \
                    (graph[current_node.position[0] + 1][current_node.position[1] + 0][current_node.position[2] + 0] != 0):    
                    continue
            
            
            elif move == (1, 0, 1):

                if (graph[current_node.position[0] + 0][current_node.position[1] + 0][current_node.position[2] + 1] != 0) or \
                    (graph[current_node.position[0] + 1][current_node.position[1] + 0][current_node.position[2] + 0] != 0):    
                    continue

            
            elif move == (1, -1, 0):

                if (graph[current_node.position[0] + 0][current_node.position[1] - 1][current_node.position[2] + 0] != 0) or \
                    (graph[current_node.position[0] + 1][current_node.position[1] + 0][current_node.position[2] + 0] != 0):    
                    continue

            
            elif move == (1, 1, 0):

                if (graph[current_node.position[0] + 0][current_node.position[1] + 1][current_node.position[2] + 0] != 0) or \
                    (graph[current_node.position[0] + 1][current_node.position[1] + 0][current_node.position[2] + 0] != 0):    
                    continue


            elif move == (-1, 0, -1):

                if (graph[current_node.position[0] - 1][current_node.position[1] + 0][current_node.position[2] + 0] != 0) or \
                    (graph[current_node.position[0] + 0][current_node.position[1] + 0][current_node.position[2] - 1] != 0):    
                    continue


            elif move == (-1, 0, 1):

                if (graph[current_node.position[0] - 1][current_node.position[1] + 0][current_node.position[2] + 0] != 0) or \
                    (graph[current_node.position[0] + 0][current_node.position[1] + 0][current_node.position[2] + 1] != 0):    
                    continue


            elif move == (-1, -1, 0):

                if (graph[current_node.position[0] - 1][current_node.position[1] + 0][current_node.position[2] + 0] != 0) or \
                    (graph[current_node.position[0] + 0][current_node.position[1] - 1][current_node.position[2] + 0] != 0):    
                    continue


            elif move == (-1, 1, 0):

                if (graph[current_node.position[0] - 1][current_node.position[1] + 0][current_node.position[2] + 0] != 0) or \
                    (graph[current_node.position[0] + 0][current_node.position[1] + 1][current_node.position[2] + 0] != 0):    
                    continue

            # 3-coordinate movement check

            elif move == (1, -1, -1):

                if (graph[current_node.position[0] + 1][current_node.position[1] + 0][current_node.position[2] + 0] != 0) or \
                    (graph[current_node.position[0] + 0][current_node.position[1] - 1][current_node.position[2] + 0] != 0) or \
                        (graph[current_node.position[0] + 0][current_node.position[1] + 0][current_node.position[2] - 1] != 0) or \
                            (graph[current_node.position[0] + 0][current_node.position[1] - 1][current_node.position[2] - 1] != 0) or \
                                (graph[current_node.position[0] + 1][current_node.position[1] + 0][current_node.position[2] - 1] != 0) or \
                                    (graph[current_node.position[0] + 1][current_node.position[1] - 1][current_node.position[2] + 0] != 0):
                    continue


            elif move == (1, -1, 1):

                if (graph[current_node.position[0] + 1][current_node.position[1] + 0][current_node.position[2] + 0] != 0) or \
                    (graph[current_node.position[0] + 0][current_node.position[1] - 1][current_node.position[2] + 0] != 0) or \
                        (graph[current_node.position[0] + 0][current_node.position[1] + 0][current_node.position[2] + 1] != 0) or \
                            (graph[current_node.position[0] + 0][current_node.position[1] - 1][current_node.position[2] + 1] != 0) or \
                                (graph[current_node.position[0] + 1][current_node.position[1] + 0][current_node.position[2] + 1] != 0) or \
                                    (graph[current_node.position[0] + 1][current_node.position[1] - 1][current_node.position[2] + 0] != 0):
                    continue


            elif move == (1, 1, -1):

                if (graph[current_node.position[0] + 1][current_node.position[1] + 0][current_node.position[2] + 0] != 0) or \
                    (graph[current_node.position[0] + 0][current_node.position[1] + 1][current_node.position[2] + 0] != 0) or \
                        (graph[current_node.position[0] + 0][current_node.position[1] + 0][current_node.position[2] - 1] != 0) or \
                            (graph[current_node.position[0] + 0][current_node.position[1] + 1][current_node.position[2] - 1] != 0) or \
                                (graph[current_node.position[0] + 1][current_node.position[1] + 0][current_node.position[2] - 1] != 0) or \
                                    (graph[current_node.position[0] + 1][current_node.position[1] + 1][current_node.position[2] + 0] != 0):
                    continue


            elif move == (1, 1, 1):

                if (graph[current_node.position[0] + 1][current_node.position[1] + 0][current_node.position[2] + 0] != 0) or \
                    (graph[current_node.position[0] + 0][current_node.position[1] + 1][current_node.position[2] + 0] != 0) or \
                        (graph[current_node.position[0] + 0][current_node.position[1] + 0][current_node.position[2] + 1] != 0) or \
                            (graph[current_node.position[0] + 0][current_node.position[1] + 1][current_node.position[2] + 1] != 0) or \
                                (graph[current_node.position[0] + 1][current_node.position[1] + 0][current_node.position[2] + 1] != 0) or \
                                    (graph[current_node.position[0] + 1][current_node.position[1] + 1][current_node.position[2] + 0] != 0):
                    continue


            elif move == (-1, -1, -1):

                if (graph[current_node.position[0] - 1][current_node.position[1] + 0][current_node.position[2] + 0] != 0) or \
                    (graph[current_node.position[0] + 0][current_node.position[1] - 1][current_node.position[2] + 0] != 0) or \
                        (graph[current_node.position[0] + 0][current_node.position[1] + 0][current_node.position[2] - 1] != 0) or \
                            (graph[current_node.position[0] + 0][current_node.position[1] - 1][current_node.position[2] - 1] != 0) or \
                                (graph[current_node.position[0] - 1][current_node.position[1] + 0][current_node.position[2] - 1] != 0) or \
                                    (graph[current_node.position[0] - 1][current_node.position[1] - 1][current_node.position[2] + 0] != 0):
                    continue


            elif move == (-1, -1, 1):

                if (graph[current_node.position[0] - 1][current_node.position[1] + 0][current_node.position[2] + 0] != 0) or \
                    (graph[current_node.position[0] + 0][current_node.position[1] - 1][current_node.position[2] + 0] != 0) or \
                        (graph[current_node.position[0] + 0][current_node.position[1] + 0][current_node.position[2] + 1] != 0) or \
                            (graph[current_node.position[0] + 0][current_node.position[1] - 1][current_node.position[2] + 1] != 0) or \
                                (graph[current_node.position[0] - 1][current_node.position[1] + 0][current_node.position[2] + 1] != 0) or \
                                    (graph[current_node.position[0] - 1][current_node.position[1] - 1][current_node.position[2] + 0] != 0):
                    continue


            elif move == (-1, 1, -1):

                if (graph[current_node.position[0] - 1][current_node.position[1] + 0][current_node.position[2] + 0] != 0) or \
                    (graph[current_node.position[0] + 0][current_node.position[1] + 1][current_node.position[2] + 0] != 0) or \
                        (graph[current_node.position[0] + 0][current_node.position[1] + 0][current_node.position[2] - 1] != 0) or \
                            (graph[current_node.position[0] + 0][current_node.position[1] + 1][current_node.position[2] - 1] != 0) or \
                                (graph[current_node.position[0] - 1][current_node.position[1] + 0][current_node.position[2] - 1] != 0) or \
                                    (graph[current_node.position[0] - 1][current_node.position[1] + 1][current_node.position[2] + 0] != 0):
                    continue


            elif move == (-1, 1, 1):

                if (graph[current_node.position[0] - 1][current_node.position[1] + 0][current_node.position[2] + 0] != 0) or \
                    (graph[current_node.position[0] + 0][current_node.position[1] + 1][current_node.position[2] + 0] != 0) or \
                        (graph[current_node.position[0] + 0][current_node.position[1] + 0][current_node.position[2] + 1] != 0) or \
                            (graph[current_node.position[0] + 0][current_node.position[1] + 1][current_node.position[2] + 1] != 0) or \
                                (graph[current_node.position[0] - 1][current_node.position[1] + 0][current_node.position[2] + 1] != 0) or \
                                    (graph[current_node.position[0] - 1][current_node.position[1] + 1][current_node.position[2] + 0] != 0):
                    continue

            # New change 527 ends


            if node_position in closed_list:  # For PERFORMANCE
                continue  

            

            new_node = Node(current_node, node_position)
        
        
        

            # if move == (-1, -1) or move == (-1, 1) or move == (1, -1) or move == (1, 1):  # Diagonal movements
            if move == (0, -1, -1) or move == (0, -1, 1) or move == (0, 1, -1) or move == (0, 1, 1) or \
                move == (1, 0, -1) or move == (1, 0, 1) or move == (1, -1, 0) or move == (1, 1, 0) or \
                    move == (-1, 0, -1) or move == (-1, 0, 1) or move == (-1, -1, 0) or move == (-1, 1, 0):
                new_node.g = current_node.g + 1.5
            elif move == (1, -1, -1) or move == (1, -1, 1) or move == (1, 1, -1) or move == (1, 1, 1) or \
                move == (-1, -1, -1) or move == (-1, -1, 1) or move == (-1, 1, -1) or move == (-1, 1, 1):
                new_node.g = current_node.g + 2
            else:
                new_node.g = current_node.g + 1     
            
            # squared_hcost = ((new_node.position[0] - end_node.position[0]) ** 2) + (
            #         (new_node.position[1] - end_node.position[1]) ** 2) + (
            #             (new_node.position[2] - end_node.position[2]) ** 2)
            # # new_node.h = float(format(sqrt(squared_hcost), '.2f'))
            # new_node.h = sqrt(squared_hcost)

            # new_node.f = new_node.g + new_node.h
            
            # We check whether a child is already in OL
            # If not, then simply add it in OL
            # If it is already in OL, then see of it's g value is more than child's g value
            # If yes, the update it's parent, g and f values with that of child's as child brings a better path

            # ind = 0
            # new_node_in_OL = False

            # for ind, heap_elem in enumerate(open_list):
            #     if new_node == heap_elem[1]:
            #         new_node_in_OL = True
            #         break
            
            
            # if new_node_in_OL == True:  # i.e. if child is already present in OL
            #     if (new_node.g < open_list[ind][1].g):     # open_list[ind] is heap's element, and open_list[ind][1] is Node obj
            #         open_list[ind][1].parent = new_node.parent
            #         open_list[ind][1].g = new_node.g
            #         open_list[ind][1].f = new_node.f

            #         open_list[ind][0] = new_node.f   # updating priority, i.e. f-value of heap element
            #         heapify(open_list)        
            # else:
            
            if new_node.g < distance_array[new_node.position[0]][new_node.position[1]][new_node.position[2]]:
                
                squared_hcost = ((new_node.position[0] - end_node.position[0]) ** 2) + (
                    (new_node.position[1] - end_node.position[1]) ** 2) + (
                        (new_node.position[2] - end_node.position[2]) ** 2)
                # new_node.h = float(format(sqrt(squared_hcost), '.2f'))
                new_node.h = sqrt(squared_hcost)

                new_node.f = new_node.g + new_node.h

                heappush(open_list, [new_node.f, new_node])    # priority, element
                distance_array[new_node.position[0]][new_node.position[1]][new_node.position[2]] = new_node.g


    
    # When no path is found
    if rtype == 'costOnly':
        return inf, open_list, closed_list, distance_array
    elif rtype == 'pathAndCost':
        return [], inf, open_list, closed_list, distance_array

# RRA-star 3D ends

def main():
    
    '''
    graph1 = [
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0]
            ]
    
    graph2 = [
            [0, 1, 0, 0, 0, 0],
            [1, 0, 1, 0, 1, 0],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0]
            ]
    
    start = (0, 0)
    end = (5, 5)

    
    path, cost = ASTAR(graph1, start, end)
    print(path, "Cost: ", cost)
    '''


    import numpy as np
    
    workSpace = [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 0],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
            ]



    start_loc = (0, 0)
    # goal_loc  = (5, 5)


    all_goal_loc_dict = {}
    all_goal_loc_dict['g1'] = (5, 5)
    all_goal_loc_dict['g2'] = (4, 0)
    all_goal_loc_dict['g3'] = (1, 5)


    # obstacles = ( [2, 1], [2, 2], [3, 1], [1, 3] ) 

    # update obstacles in workspace matrix :- make obs cells 1

    # for obs in obstacles:
    #     workSpace[obs[0]][obs[1]] = 1


    
    h_cost = 0
    path_dict, cost_dict = DIJKSTRA_2D( workSpace, start_loc, set(all_goal_loc_dict.values()), h_cost, rtype = 'pathAndCost' )
    
    print(path_dict, "\nCost: ", cost_dict)

    

def main2():
    
    
    workSpace = [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 0],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
            ]


    start_loc = (0, 0)
    list_goal_loc  = [ (3, 0), (5, 0) ] 

    open_list = []
    closed_list = {}

    for goal_loc in list_goal_loc:
        path, cost, open_list, closed_list = FRASTAR( workSpace, start_loc, goal_loc, open_list, closed_list, rtype = 'pathAndCost' )
        print("Cost: ", cost)
        print("Path: ", path)
    
    

if __name__ == '__main__':
    main2()