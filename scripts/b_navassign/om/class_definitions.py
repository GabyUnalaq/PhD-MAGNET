'''
All classes are in this file.
Classes present: Vertex, Edge and Graph
'''

import math


class Vertex: 

	def __init__(self, key, label = None, in_left = None):
		self.key = key
		self.label = label
		self.neighbors = set()
		self.incident_edges = {}  # can be optimized if we use a dict here; then get_edge() function will hit the edge in 1 go
		self.in_left = in_left

	
	def set_label(self, label):
		self.label = label

	def set_in_left(self, in_left):
		self.in_left = in_left

# ------------------------------------------------------------------------------------------------------------------------


class Edge:

	def __init__(self, v1, v2, weight = math.inf, heur = False):
		self.vertices = [v1, v2]
		self.weight = weight
		
		if heur == True:
			self.typeOfWeight = 'h'  # can be 'h' - for heuristic or 'a' - for actual weight
		else:
			self.typeOfWeight = None 


	def get_pair_vertex(self, v):
		for vertex in self.vertices:
			if ( vertex != v.key ):
				return vertex


	def __eq__(self, e):
		return (self.vertices == e.vertices
				and self.weight == e.weight)

	def __hash__(self):
		return hash((frozenset(self.vertices), self.weight))


# ------------------------------------------------------------------------------------------------------------------------


class Graph:

	def __init__( self, G = None, heur = False, eq_flag = False ):
		



		self.vertices = {}

		if eq_flag == False:
		
			# If the input graph is in the form of an adjacency matrix (then we won't have vertex labels such as 'r1', 't1' in the input; rather we need to create them)

			goals_lessThan_robots = False  # change 999
			
			# change 999 - for handling case when no of robots != no of goals

			if G.shape[1] < G.shape[0]:  # that is, if the no of goals is < nof of robots, then we change the left-right orientation
				goals_lessThan_robots = True
			
			for i in range(G.shape[0]):
				v1 = 'r' + str(i)
				for j in range(G.shape[1]):
					v2 = 'g' + str(j)
					# self.add_edge(v1, v2, G[i][j])  change 999
					self.add_edge(v1, v2, G[i][j], goals_lessThan_robots, heur)  # change 999

					

	def add_vertex(self, key, label = None, in_left = None):
		'''
		Adds a vertex to the graph.
		'''
		self.vertices[key] = Vertex(key, label, in_left)


	def add_edge( self, v1, v2, weight = math.inf, goals_lessThan_robots = False, heur = False ):
		'''
		Adds an edge to the graph.
		'''

		if v1 not in self.vertices:
			self.add_vertex(v1)
		if v2 not in self.vertices:
			self.add_vertex(v2)

		
		e = Edge(v1, v2, weight, heur)

		self.vertices[v1].neighbors.add(v2)
		self.vertices[v2].neighbors.add(v1)
		# self.vertices[v1].incident_edges.add(e)
		# self.vertices[v2].incident_edges.add(e)
		
		self.vertices[v1].incident_edges[v2] = e
		self.vertices[v2].incident_edges[v1] = e


		if goals_lessThan_robots == False:
			self.vertices[v1].set_in_left(True)    
			self.vertices[v2].set_in_left(False)   
		else:
			self.vertices[v2].set_in_left(True)    
			self.vertices[v1].set_in_left(False)   


						


	def feasibly_label(self, v):
		

		min = math.inf

		for e in self.vertices[v].incident_edges.values():
			# if max is None or e.weight > max: 
			if (e.weight < min):
				min = e.weight

		self.vertices[v].set_label(min)
		# self.vertices[v].set_in_left(True) 

	
	
	def generate_feasible_labeling(self, left_vertices, right_vertices):
		'''
		Generates an initial feasible labeling.
		'''
		
		for x in left_vertices:
			self.feasibly_label(x)
		for y in right_vertices:
			self.vertices[y].set_label(0)
		
		

	def generate_feasible_labeling_H(self, left_vertices, right_vertices, minAcost):
		'''
		Generates an initial feasible labeling for heur version
		'''
		
		for x in left_vertices:
			self.vertices[x].set_label(minAcost[x])
		for y in right_vertices:
			self.vertices[y].set_label(0)


