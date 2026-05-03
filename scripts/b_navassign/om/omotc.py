import time
from heapq import *
import numpy as np
from math import sqrt, inf, isinf

from .search_methods import FRASTAR, FRASTAR_3D
from .class_definitions import Graph, Edge
from .utility_functions import find_MinVertexCover, get_initial_match, maximize_match, get__equality_subgraph, update__equality_subgraph, get__minimal_subgraph


left_vertices = set()
right_vertices = set()
path = {}
no_of_explorations = 0

OL_dict = {}
CL_dict = {}
distance_lookup = {}


t1 = 0
t2 = 0
t3 = 0
t4 = 0

# ---------------------------------------------------------------------------------------------------------------------
# collect_uncovered_costs : Function to collect delta (weight) for edges between uncovered robots and uncovered goals
# ---------------------------------------------------------------------------------------------------------------------

def collect_uncovered_costs(G, mvc):
	

	# Collecting delta:

	uncovered_robots = left_vertices - mvc
	uncovered_goals = right_vertices - mvc
	
	
	deltas = []
	for i in uncovered_robots:
		for j in uncovered_goals: 
			edge = G.vertices[i].incident_edges[j]
			# slack = edge.weight - ( G.vertices[i].label + G.vertices[j].label )
			# if not math.isinf(delta):
			# deltas.append( ( i, j, slack, edge.typeOfWeight ) )
			# DELTA.append( ( edge.weight, slack, edge.typeOfWeight, i, j ) )  # c77
			deltas.append( ( edge.weight, edge.typeOfWeight, i, j ) )  # c77

	
	heapify(deltas) # c77
			
	return deltas

# **********************************************************************************************************************


# ---------------------------------------------------------------------------------------------------------------------
# collect_slacks : Function to collect slacks for edges between uncovered robots and uncovered goals
# ---------------------------------------------------------------------------------------------------------------------

def collect_slacks(G, mvc):
	

	# Collecting delta:

	uncovered_robots = left_vertices - mvc
	uncovered_goals = right_vertices - mvc
	
	
	deltas = []
	for i in uncovered_robots:
		for j in uncovered_goals:
			if j in G.vertices[i].neighbors:  # G_th may not have few edges (whose wt > makespan), so this check is necessary
				edge = G.vertices[i].incident_edges[j]
				slack = edge.weight - ( G.vertices[i].label + G.vertices[j].label )
				deltas.append( ( slack, edge.typeOfWeight, i, j ) )  # c77
			else:
				deltas.append( ( inf, 'a', i, j ) )  # c77
			
	
	heapify(deltas) # c77
			
	return deltas

# **********************************************************************************************************************


# ----------------------------------------------------------------------------------------------------------------------
# get_min_delta (get_global_delta_min) : This function fetches the minimum delta (cost / slack) value between pairs of uncov robots & goals
# ----------------------------------------------------------------------------------------------------------------------

def get_min_delta( deltas, G, workSpace, all_start_loc, all_goal_loc, ws_type, makespan = inf, makespan_flag = False ):

	global no_of_explorations, OL_dict, CL_dict, distance_lookup

	while (1):
		
		d_min = deltas[0] # c77

		if d_min[1] == 'a':
			
			if makespan_flag == True:
			
				if makespan < d_min[0]:
					makespan = d_min[0]

				return makespan, G
			
			else:
				
				return d_min[0], G

		else:
			# Explore

			edge = G.vertices[d_min[2]].incident_edges[d_min[3]]  # edge variable will store the Edge object
			if d_min[2][0] == 'r':
				if ws_type == '3D':
					one_path, edge.weight, OL_dict[d_min[2]], CL_dict[d_min[2]], distance_lookup[d_min[2]] = FRASTAR_3D( workSpace, all_start_loc[d_min[2]], all_goal_loc[d_min[3]], OL_dict[d_min[2]], CL_dict[d_min[2]], distance_lookup[d_min[2]], rtype = 'pathAndCost' )
				else:
					one_path, edge.weight, OL_dict[d_min[2]], CL_dict[d_min[2]], distance_lookup[d_min[2]] = FRASTAR( workSpace, all_start_loc[d_min[2]], all_goal_loc[d_min[3]], OL_dict[d_min[2]], CL_dict[d_min[2]], distance_lookup[d_min[2]], rtype = 'pathAndCost' )
				path[d_min[2]][d_min[3]] = one_path
			else:
				if ws_type == '3D':
					one_path, edge.weight, OL_dict[d_min[2]], CL_dict[d_min[2]], distance_lookup[d_min[2]] = FRASTAR_3D( workSpace, all_goal_loc[d_min[2]], all_start_loc[d_min[3]], OL_dict[d_min[2]], CL_dict[d_min[2]], distance_lookup[d_min[2]], rtype = 'pathAndCost' )
				else:
					one_path, edge.weight, OL_dict[d_min[2]], CL_dict[d_min[2]], distance_lookup[d_min[2]] = FRASTAR( workSpace, all_goal_loc[d_min[2]], all_start_loc[d_min[3]], OL_dict[d_min[2]], CL_dict[d_min[2]], distance_lookup[d_min[2]], rtype = 'pathAndCost' )
				path[d_min[3]][d_min[2]] = one_path[::-1]
			
			edge.typeOfWeight = 'a'
			no_of_explorations = no_of_explorations + 1
			

			# new_slack = edge.weight - ( G.vertices[d_min[3]].label + G.vertices[d_min[4]].label )
			
			if makespan_flag == False:
				new_slack = edge.weight - ( G.vertices[d_min[2]].label + G.vertices[d_min[3]].label )
				new_delta = ( new_slack, 'a', d_min[2], d_min[3] )
			else:
				new_delta = ( edge.weight, 'a', d_min[2], d_min[3] )

			if ( edge.weight <= makespan ) or ( makespan_flag == True ):
				heapreplace( deltas, new_delta )
			else:
				heappop(deltas)
				# Here, we are in Hungarian zone, so remove the edge having wt > makespan from G_th, as G_th is being processed by Hungarian method and we don't need edges.wt > makespan
				G.vertices[d_min[2]].neighbors.discard(d_min[3])
				G.vertices[d_min[3]].neighbors.discard(d_min[2])
				G.vertices[d_min[2]].incident_edges.pop(d_min[3])
				G.vertices[d_min[3]].incident_edges.pop(d_min[2])



	
# **********************************************************************************************************************


def find_min_Zcost( x, type_of_wt ):  # receives a Vertex object as parameter, Z refers to 'h' or 'a'

	minZcost = inf
	minZindex = None

	for edge in x.incident_edges.values():
		if ( edge.typeOfWeight == type_of_wt ) and ( edge.weight < minZcost ):
			minZcost = edge.weight
			minZindex = edge.get_pair_vertex(x)  # minZindex carries goal g


	return minZcost, minZindex

# **********************************************************************************************************************

# ---------------------------------------------------------------------------------------------------------------------
# Function to explore min actual cost for each robot (ROW wise)
# ---------------------------------------------------------------------------------------------------------------------

def explore_min_actual_cost_rowwise( G, workSpace, all_start_loc, all_goal_loc, ws_type ):

	minAcost = {}
	minAindex = {}
	global no_of_explorations, OL_dict, CL_dict, distance_lookup

	
	for x in left_vertices:    # for each robot
		
		OL_dict[x] = []
		CL_dict[x] = {}

		if ws_type == '3D':
			distance_lookup[x] = np.empty(shape=(workSpace.shape[0], workSpace.shape[1], workSpace.shape[2]))
		else:
			distance_lookup[x] = np.empty(shape=(workSpace.shape[0], workSpace.shape[1]))
		
		distance_lookup[x].fill(inf)

		

		minHcost, minHindex = find_min_Zcost( G.vertices[x], 'h' )
		
		minAcost[x] = inf
		minAindex[x] = None
		
		
		while ( minHindex != None and minHcost <= minAcost[x] ):
			
			edge = G.vertices[x].incident_edges[minHindex]
		
			if x[0] == 'r':
				if ws_type == '3D':
					one_path, edge.weight, OL_dict[x], CL_dict[x], distance_lookup[x] = FRASTAR_3D( workSpace, all_start_loc[x], all_goal_loc[minHindex], OL_dict[x], CL_dict[x], distance_lookup[x], rtype = 'pathAndCost' )
				else:
					one_path, edge.weight, OL_dict[x], CL_dict[x], distance_lookup[x] = FRASTAR( workSpace, all_start_loc[x], all_goal_loc[minHindex], OL_dict[x], CL_dict[x], distance_lookup[x], rtype = 'pathAndCost' )
				path[x][minHindex] = one_path
			else:
				if ws_type == '3D':
					one_path, edge.weight, OL_dict[x], CL_dict[x], distance_lookup[x] = FRASTAR_3D( workSpace, all_goal_loc[x], all_start_loc[minHindex], OL_dict[x], CL_dict[x], distance_lookup[x], rtype = 'pathAndCost' )
				else:
					one_path, edge.weight, OL_dict[x], CL_dict[x], distance_lookup[x] = FRASTAR( workSpace, all_goal_loc[x], all_start_loc[minHindex], OL_dict[x], CL_dict[x], distance_lookup[x], rtype = 'pathAndCost' )
				path[minHindex][x] = one_path[::-1]
			
			edge.typeOfWeight = 'a'
			no_of_explorations = no_of_explorations + 1

			minHcost, minHindex = find_min_Zcost( G.vertices[x], 'h' )
			minAcost[x], minAindex[x] = find_min_Zcost( G.vertices[x], 'a' )


	return G, minAcost

# **********************************************************************************************************************

# colwise 

# ---------------------------------------------------------------------------------------------------------------------
# Function to explore min actual cost for each goal (COLUMNWISE wise)
# ---------------------------------------------------------------------------------------------------------------------

def explore_min_actual_cost_columnwise( G, workSpace, all_start_loc, all_goal_loc, ws_type ):

	minAcost = {}
	minAindex = {}
	global no_of_explorations, OL_dict, CL_dict, distance_lookup

	
	for x in right_vertices:    
		
		minHcost, minHindex = find_min_Zcost( G.vertices[x], 'h' )  # for R <=G, here Hindex will have robot
		
		minAcost[x], minAindex[x] = find_min_Zcost( G.vertices[x], 'a' )

		
		
		while ( minHindex != None and minHcost <= minAcost[x] ):
			
			edge = G.vertices[minHindex].incident_edges[x]
			
			if minHindex[0] == 'r':
				if ws_type == '3D':
					one_path, edge.weight, OL_dict[minHindex], CL_dict[minHindex], distance_lookup[minHindex] = FRASTAR_3D( workSpace, all_start_loc[minHindex], all_goal_loc[x], OL_dict[minHindex], CL_dict[minHindex], distance_lookup[minHindex], rtype = 'pathAndCost' )
				else:
					one_path, edge.weight, OL_dict[minHindex], CL_dict[minHindex], distance_lookup[minHindex] = FRASTAR( workSpace, all_start_loc[minHindex], all_goal_loc[x], OL_dict[minHindex], CL_dict[minHindex], distance_lookup[minHindex], rtype = 'pathAndCost' )
				path[minHindex][x] = one_path
			else:
				if ws_type == '3D':
					one_path, edge.weight, OL_dict[minHindex], CL_dict[minHindex], distance_lookup[minHindex] = FRASTAR_3D( workSpace, all_goal_loc[minHindex], all_start_loc[x], OL_dict[minHindex], CL_dict[minHindex], distance_lookup[minHindex], rtype = 'pathAndCost' )
				else:
					one_path, edge.weight, OL_dict[minHindex], CL_dict[minHindex], distance_lookup[minHindex] = FRASTAR( workSpace, all_goal_loc[minHindex], all_start_loc[x], OL_dict[minHindex], CL_dict[minHindex], distance_lookup[minHindex], rtype = 'pathAndCost' )
				path[x][minHindex] = one_path[::-1]
			
			edge.typeOfWeight = 'a'
			no_of_explorations = no_of_explorations + 1

			minHcost, minHindex = find_min_Zcost( G.vertices[x], 'h' )
			minAcost[x], minAindex[x] = find_min_Zcost( G.vertices[x], 'a' )


	return G, minAcost

# **********************************************************************************************************************







# Get threshold graph - heuristic case

def get__threshold_subgraph_H( G, makespan, workSpace, all_start_loc, all_goal_loc, ws_type ):

	
	global no_of_explorations, OL_dict, CL_dict, distance_lookup

	G_th = Graph( eq_flag = True )
	
	
	# Creating vertices
	for x in left_vertices:
		G_th.add_vertex( x, G.vertices[x].label, G.vertices[x].in_left )

	for y in right_vertices:
		G_th.add_vertex( y, G.vertices[y].label, G.vertices[y].in_left )
	

	# Creating edges
	for x in left_vertices:
		for y in right_vertices:

			edge = G.vertices[x].incident_edges[y]  # edge variable will store the Edge object

			
			if ( edge.typeOfWeight == 'h' ) and (edge.weight <= makespan):
				# We need to confirm whether the A-cost is really <= makespan; if yes, the edge must be allowed in the threshold graph, else no
				if x[0] == 'r':
					if ws_type == '3D':
						one_path, edge.weight, OL_dict[x], CL_dict[x], distance_lookup[x] = FRASTAR_3D( workSpace, all_start_loc[x], all_goal_loc[y], OL_dict[x], CL_dict[x], distance_lookup[x], rtype = 'pathAndCost' )
					else:
						one_path, edge.weight, OL_dict[x], CL_dict[x], distance_lookup[x] = FRASTAR( workSpace, all_start_loc[x], all_goal_loc[y], OL_dict[x], CL_dict[x], distance_lookup[x], rtype = 'pathAndCost' )
					path[x][y] = one_path
				else:
					if ws_type == '3D':
						one_path, edge.weight, OL_dict[x], CL_dict[x], distance_lookup[x] = FRASTAR_3D( workSpace, all_goal_loc[x], all_start_loc[y], OL_dict[x], CL_dict[x], distance_lookup[x], rtype = 'pathAndCost' )
					else:
						one_path, edge.weight, OL_dict[x], CL_dict[x], distance_lookup[x] = FRASTAR( workSpace, all_goal_loc[x], all_start_loc[y], OL_dict[x], CL_dict[x], distance_lookup[x], rtype = 'pathAndCost' )
					path[y][x] = one_path[::-1]
				
				edge.typeOfWeight = 'a'
				no_of_explorations = no_of_explorations + 1


			
			
			if ( edge.typeOfWeight != 'h' ) and (edge.weight <= makespan):
				# add edge
				if x[0] == 'r':
					e = Edge(x, y, edge.weight)
				else:
					e = Edge(y, x, edge.weight)
				e.typeOfWeight = edge.typeOfWeight


				G_th.vertices[x].neighbors.add(y)
				G_th.vertices[y].neighbors.add(x)
				G_th.vertices[x].incident_edges[y] = e
				G_th.vertices[y].incident_edges[x] = e


	return G_th, G


# Update threshold graph - heuristic case

def update__threshold_subgraph_H( G_th, G, mvc, makespan, workSpace, all_start_loc, all_goal_loc, ws_type ):


	global no_of_explorations, OL_dict, CL_dict, distance_lookup

	# Revising edges between uncovered robot - uncovered goal pairs (new edges may appear)

	uncovered_robots = left_vertices - mvc
	uncovered_goals = right_vertices - mvc
	

	for x in uncovered_robots:
		for y in uncovered_goals:

			edge = G.vertices[x].incident_edges[y]  # edge variable will store the Edge object

			
			if ( edge.typeOfWeight == 'h' ) and (edge.weight <= makespan):  # if min uncov wt < makespan, then == will prevent it to come in G_th, so we should have <= 
				if x[0] == 'r':
					# Explore A.cost
					if ws_type == '3D':
						one_path, edge.weight, OL_dict[x], CL_dict[x], distance_lookup[x] = FRASTAR_3D( workSpace, all_start_loc[x], all_goal_loc[y], OL_dict[x], CL_dict[x], distance_lookup[x], rtype = 'pathAndCost' )
					else:
						one_path, edge.weight, OL_dict[x], CL_dict[x], distance_lookup[x] = FRASTAR( workSpace, all_start_loc[x], all_goal_loc[y], OL_dict[x], CL_dict[x], distance_lookup[x], rtype = 'pathAndCost' )
					path[x][y] = one_path
				else:
					if ws_type == '3D':
						one_path, edge.weight, OL_dict[x], CL_dict[x], distance_lookup[x] = FRASTAR_3D( workSpace, all_goal_loc[x], all_start_loc[y], OL_dict[x], CL_dict[x], distance_lookup[x], rtype = 'pathAndCost' )
					else:
						one_path, edge.weight, OL_dict[x], CL_dict[x], distance_lookup[x] = FRASTAR( workSpace, all_goal_loc[x], all_start_loc[y], OL_dict[x], CL_dict[x], distance_lookup[x], rtype = 'pathAndCost' )
					path[y][x] = one_path[::-1]
				
				edge.typeOfWeight = 'a'
				no_of_explorations = no_of_explorations + 1

			
			
			
			if ( edge.typeOfWeight != 'h' ) and (edge.weight <= makespan):
				# add edge
				if x[0] == 'r':
					e = Edge(x, y, edge.weight)
				else:
					e = Edge(y, x, edge.weight)
				e.typeOfWeight = edge.typeOfWeight


				G_th.vertices[x].neighbors.add(y)
				G_th.vertices[y].neighbors.add(x)
				G_th.vertices[x].incident_edges[y] = e
				G_th.vertices[y].incident_edges[x] = e

	
	return G_th, G



# ---------------------------------------------------------------------------------------------------------------------
# main function starts
# ---------------------------------------------------------------------------------------------------------------------


def find_assignment_H( _G, workSpace, all_start_loc, all_goal_loc, ws_type, om_only = False ):
	# MAGNET - om_only param was added
	

	global no_of_explorations, OL_dict, CL_dict, distance_lookup, t1, t2, t3, t4
	no_of_explorations = 0
	path.clear()
	left_vertices.clear()
	right_vertices.clear()
	OL_dict.clear()
	CL_dict.clear()
	distance_lookup.clear()
	
	
	# Construct a bipartite graph 
	
	G = Graph( _G, heur = True )
	
		
	for x in G.vertices:
		if G.vertices[x].in_left:
			left_vertices.add(x)
			if x[0] == 'r':
				path[x] = {}
		else:
			right_vertices.add(x)
			if x[0] == 'r':
				path[x] = {}
	


	no_of_robots = len( all_start_loc )
	no_of_goals = len( all_goal_loc )



	G, minAcost = explore_min_actual_cost_rowwise( G, workSpace, all_start_loc, all_goal_loc, ws_type )
	
	if no_of_robots == no_of_goals:

		G, minAcost_col = explore_min_actual_cost_columnwise( G, workSpace, all_start_loc, all_goal_loc, ws_type )
		all_min = np.array( list(minAcost.values()) + list(minAcost_col.values()) )
	else:
		all_min = np.array( list(minAcost.values()) )
	
	
	# Get initial makespan
	# makespan = np.max( list(minAcost.values()) + list(minAcost_col.values()) )
	
	try:
		makespan = np.max(all_min[np.isfinite(all_min)])  # a robot (goal) can be trapped in case of unequal R & G. So, we need to ignore infinite cost for such entity
	except ValueError:
		return None, None, None, None, None, None, None, None, None


	# # Generate an initial feasible labeling
	# G.generate_feasible_labeling_H( left_vertices, right_vertices, minAcost ) 


	# Get the threshold subgraph
	G_th, G = get__threshold_subgraph_H( G, makespan, workSpace, all_start_loc, all_goal_loc, ws_type )

	
	# Finding an initial matching
	M = get_initial_match(G_th, left_vertices)

	# Maximize the current matching
	M = maximize_match(G_th, M, left_vertices)

	
	# change 999 - for handling case when no of robots != no of goals
	min_robots_goals = min( len(left_vertices), len(right_vertices) )  # change 999

	
	# while len(M) < int(len(eq_G.vertices)/2):   # change 999
	while len(M) < min_robots_goals:

		mvc = find_MinVertexCover(G_th, M, left_vertices, right_vertices)
		
	
		deltas = collect_uncovered_costs(G, mvc)
		makespan, G = get_min_delta( deltas, G, workSpace, all_start_loc, all_goal_loc, ws_type, makespan, True )
			
		
		G_th, G = update__threshold_subgraph_H( G_th, G, mvc, makespan, workSpace, all_start_loc, all_goal_loc, ws_type )
		
		M = maximize_match(G_th, M, left_vertices)
		


	# print("Final MAKESPAN: ", makespan, "\n")

	
	t2 = time.time()

	
	# Save a copy of nexp until now, so that it will represent # exp needed to get optimal makespan (excluding min TC)
	makespan_nexp = no_of_explorations
	
	
	# counting number of nodes expanded BEGINS (for optimal makespan only - excluding Hungarian)
	makespan_CL = 0

	for robo in CL_dict:
		makespan_CL = makespan_CL + len(CL_dict[robo])
	
	# recording om's cost 
	om_cost = 0

	if (len(M) != 0):
		for e in M:
			if not isinf( e.weight ):
				om_cost = om_cost + e.weight

	
	t3 = time.time()

	# MAGNET - Section was added
	if om_only:
		t4 = time.time()
		unassigned_count = min_robots_goals - len(M)
		return list(map(lambda e: ((e.vertices[0], e.vertices[1]), e.weight), M)), om_cost, om_cost, path, unassigned_count, makespan_nexp, no_of_explorations, makespan, makespan_CL

	# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
	# # INVOKE HUNGARIAN METHOD for finding min TC assignment
	# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
	
	
	# Get minimal subgraph (one which has all edges whose weight <= makespan)
	G_th = get__minimal_subgraph( G_th, G, left_vertices, right_vertices, makespan )


	# PROCESS HUNGARIAN STEPS on G_th

	# Generate an initial feasible labeling
	G_th.generate_feasible_labeling_H( left_vertices, right_vertices, minAcost ) 

	
	# Get the equality subgraph
	eq_G = get__equality_subgraph( G_th, left_vertices, right_vertices )

	# Finding an initial matching
	M = get_initial_match(eq_G, left_vertices)

	# Maximize the current matching
	M = maximize_match(eq_G, M, left_vertices)


	# change 999 - for handling case when no of robots != no of goals
	min_robots_goals = min( len(left_vertices), len(right_vertices) )  # change 999


	# while len(M) < int(len(eq_G.vertices)/2):   # change 999
	while len(M) < min_robots_goals:

		mvc = find_MinVertexCover(eq_G, M, left_vertices, right_vertices) 
		
		deltas = collect_slacks(G_th, mvc)		
		global_d_min, G_th = get_min_delta( deltas, G_th, workSpace, all_start_loc, all_goal_loc, ws_type, makespan, False )

		if isinf( global_d_min ):
			break                        # This means no further goal is reachable; So Break and report the available matching. 

		
		# For heur case, before getting eq-subgraph of 2nd round (and onwards), we need to Explore such "heuristic" cost edges, 
		# whose slack is equal to min_slack (global_d_min), after the just above label-update. 
		# This is to be done ONLY between uncovered robots and uncovered goals. (as a new edge could arise between them only)
		# We didn't need this step before the generation of 1st eq_subgraph because this case arises only after a label-update as
		# new edges would come after a label-update.

		
		uncovered_robots = left_vertices - mvc
		uncovered_goals = right_vertices - mvc

		for x in uncovered_robots:        # for each robot
			for y in uncovered_goals:     # for each goal
			
				if y in G_th.vertices[x].neighbors:  # G_th may not have few edges (whose wt > makespan), so this check is necessary
					edge = G_th.vertices[x].incident_edges[y]  # edge variable will store the Edge object
					slack = edge.weight - (G_th.vertices[x].label + G_th.vertices[y].label)
					if ( edge.typeOfWeight == 'h' ) and ( global_d_min == slack ):
						if x[0] == 'r':
							# Explore A.cost
							if ws_type == '3D':
								one_path, edge.weight, OL_dict[x], CL_dict[x], distance_lookup[x] = FRASTAR_3D( workSpace, all_start_loc[x], all_goal_loc[y], OL_dict[x], CL_dict[x], distance_lookup[x], rtype = 'pathAndCost' )
							else:
								one_path, edge.weight, OL_dict[x], CL_dict[x], distance_lookup[x] = FRASTAR( workSpace, all_start_loc[x], all_goal_loc[y], OL_dict[x], CL_dict[x], distance_lookup[x], rtype = 'pathAndCost' )
							path[x][y] = one_path
						else:
							if ws_type == '3D':
								one_path, edge.weight, OL_dict[x], CL_dict[x], distance_lookup[x] = FRASTAR_3D( workSpace, all_goal_loc[x], all_start_loc[y], OL_dict[x], CL_dict[x], distance_lookup[x], rtype = 'pathAndCost' )
							else:
								one_path, edge.weight, OL_dict[x], CL_dict[x], distance_lookup[x] = FRASTAR( workSpace, all_goal_loc[x], all_start_loc[y], OL_dict[x], CL_dict[x], distance_lookup[x], rtype = 'pathAndCost' )
							path[y][x] = one_path[::-1]
						
						edge.typeOfWeight = 'a'
						no_of_explorations = no_of_explorations + 1

						# deleting the edge from G_th if its wt > makespan
						if edge.weight > makespan:
							# Here, we are in Hungarian zone, so remove the edge having wt > makespan from G_th, as G_th is being processed by Hungarian method and we don't need edge.wt > makespan
							G_th.vertices[x].neighbors.discard(y)
							G_th.vertices[y].neighbors.discard(x)
							G_th.vertices[x].incident_edges.pop(y)
							G_th.vertices[y].incident_edges.pop(x)



			
			
		# if global_d_min != None:
		for i in left_vertices & mvc:   # for covered robots
			G_th.vertices[i].label = G_th.vertices[i].label - global_d_min
		
		for j in uncovered_goals:       # for uncovered goals
			G_th.vertices[j].label = G_th.vertices[j].label + global_d_min
			
		

		eq_G = update__equality_subgraph( eq_G, G_th, mvc, left_vertices, right_vertices, makespan )
		M = maximize_match(eq_G, M, left_vertices)
		


	
	
	t4 = time.time()

	# ------------------ End Result -----------------------

	omotc_cost = 0
	unassigned_count = 0
	assigned_count = 0
	if (len(M) != 0):
		for e in M:
			if not isinf( e.weight ):
				omotc_cost = omotc_cost + e.weight
				assigned_count = assigned_count + 1
	

	unassigned_count = min_robots_goals - assigned_count
		

	return list(map(lambda e: ((e.vertices[0], e.vertices[1]), e.weight), M)), om_cost, omotc_cost, path, unassigned_count, makespan_nexp, no_of_explorations, makespan, makespan_CL
	
# **********************************************************************************************************************



# --------------------------------------------------------------------------------------------------------------
#                          Directly - exposed function for Heuristic-based approach
# --------------------------------------------------------------------------------------------------------------

def heuristic ( no_of_robots, no_of_goals, workSpace, all_start_loc, all_goal_loc, ws_type  ):


	global t1, t2, t3, t4
	
	t1 = time.time()

	costMatrix = np.zeros((no_of_robots, no_of_goals))
	# Generating heuristic costs for all pairs of robots and goals
	for i in range(no_of_robots):
		robot_name = 'r' + str(i)
		for j in range(no_of_goals):
			goal_name = 'g' + str(j)
			
			if ws_type == '3D':
				sq_of_distance = ((all_start_loc[robot_name][0] - all_goal_loc[goal_name][0]) ** 2) + (
				(all_start_loc[robot_name][1] - all_goal_loc[goal_name][1]) ** 2) + (
					(all_start_loc[robot_name][2] - all_goal_loc[goal_name][2]) ** 2)
			else:
				sq_of_distance = ((all_start_loc[robot_name][0] - all_goal_loc[goal_name][0]) ** 2) + (
					(all_start_loc[robot_name][1] - all_goal_loc[goal_name][1]) ** 2)

			
			costMatrix[i][j] = sqrt(sq_of_distance)
			

		
	result_h, om_cost, omotc_cost, path_h, u_count_h, makespan_nexp_h, no_of_exp_h, makespan, makespan_CL = find_assignment_H( costMatrix, workSpace, all_start_loc, all_goal_loc, ws_type )

	# makespan_CL is CL count for just OM only, while heur_CL is CL count for OM+OTC

	if result_h is None:
		return None, None, None, None, None, None, None, None, None, None, None, None

	# counting number of nodes expanded BEGINS (for OM+OTC)
	heur_CL = 0

	for robo in CL_dict:
		heur_CL = heur_CL + len(CL_dict[robo])
	
	
	t_om = t2 - t1
	t_otc = t4 - t3

	return result_h, om_cost, omotc_cost, path_h, u_count_h, makespan_nexp_h, no_of_exp_h, makespan, heur_CL, makespan_CL, t_om, t_otc


# **********************************************************************************************************************


# MAGNET - Function was added
# --------------------------------------------------------------------------------------------------------------
#                          OM-only entry point (no OTC / Hungarian phase)
# --------------------------------------------------------------------------------------------------------------

def om_heuristic( no_of_robots, no_of_goals, workSpace, all_start_loc, all_goal_loc, ws_type ):
	"""
	Run only the OM (bottleneck makespan) phase and return without the OTC
	(total-cost) Hungarian phase.  Same 12-tuple signature as heuristic();
	omotc_cost equals om_cost and t_otc is 0.
	"""

	global t1, t2, t3, t4

	t1 = time.time()

	costMatrix = np.zeros((no_of_robots, no_of_goals))
	for i in range(no_of_robots):
		robot_name = 'r' + str(i)
		for j in range(no_of_goals):
			goal_name = 'g' + str(j)

			if ws_type == '3D':
				sq_of_distance = ((all_start_loc[robot_name][0] - all_goal_loc[goal_name][0]) ** 2) + (
				(all_start_loc[robot_name][1] - all_goal_loc[goal_name][1]) ** 2) + (
					(all_start_loc[robot_name][2] - all_goal_loc[goal_name][2]) ** 2)
			else:
				sq_of_distance = ((all_start_loc[robot_name][0] - all_goal_loc[goal_name][0]) ** 2) + (
					(all_start_loc[robot_name][1] - all_goal_loc[goal_name][1]) ** 2)

			costMatrix[i][j] = sqrt(sq_of_distance)

	result_h, om_cost, _, path_h, u_count_h, makespan_nexp_h, no_of_exp_h, makespan, makespan_CL = find_assignment_H( costMatrix, workSpace, all_start_loc, all_goal_loc, ws_type, om_only = True )

	if result_h is None:
		return None, None, None, None, None, None, None, None, None, None, None, None

	heur_CL = 0
	for robo in CL_dict:
		heur_CL = heur_CL + len(CL_dict[robo])

	t_om = t2 - t1

	return result_h, om_cost, om_cost, path_h, u_count_h, makespan_nexp_h, no_of_exp_h, makespan, heur_CL, makespan_CL, t_om, 0


# **********************************************************************************************************************
