import igraph as ig
from collections import defaultdict
from collections import OrderedDict
from collections import namedtuple 
from pprint import pprint
from math import log
from math import sqrt 
from random import shuffle
import logging

class Optimiser:
  """ Class for doing community detection using the Louvain algorithm. 

  Given a certain partition type is calls diff_move for trying to move a node
  to another community. It moves the node to the community that *maximises*
  this diff_move. If no further improvement is possible, the graph is
  aggregated (collapse_graph) and the method is reiterated on that graph."""
  def __init__(self, eps=1e-5, delta=1e-2, max_itr=1000, 
               random_order=True, min_diff_resolution = 1e-3,
               min_diff_bisect_value = 1):
    """ Create a new Optimiser object
    
    Parameters:
      eps=1e-5     -- If the improvement falls below this threshold, 
                      stop iterating.
      delta=1e-2   -- If the number of nodes that moves falls below 
                      this threshold, stop iterating.
      max_itr=1000 -- Maximum number of iterations to perform.
      random_order=True
                   -- If True the nodes will be traversed in a random order
                      when optimising a quality function.
      min_diff_resolution==1e-3
                   -- If the difference in resolution falls below this
                      threshold when bisectioning on a resolution parameter,
                      we won't bisection further.
      min_diff_bisect_value==1
                   -- If the difference in the bisection value falls below
                      this threshold when bisectioning on a resolution 
                      parameter, we won't bisection further."""
    self.eps = eps;
    self.delta = delta;
    self.max_itr = max_itr;
    self.random_order = random_order        
    self.min_diff_resolution = min_diff_resolution;
    self.min_diff_bisect_value = min_diff_bisect_value; 

  def find_partition(self, 
        graph, 
        partition_class, 
        initial_partition=None,  
        **kwargs):
    """ Find optimal partition given the specific type of partition_class that
    is provided.
    
    Parameters:
      graph           -- The igraph.Graph to find the optimal partition on
      partition_class -- The type of partition which will be used during
                         optimisation.
      initial_partition=None
                      -- If provided, the optimisation will depart from this
                         initial partition. Since nodes can only be merged and
                         never split, the number of communities in this initial
                         partition provides an upper bound.
      **kwargs        -- Remaing keyword arguments to be provided to the
                         default constructor of partition_class."""
    # If there is an initial partition, use it
    if not initial_partition:
      partition = partition_class(graph=graph,
          membership=range(graph.vcount()), **kwargs);
      logging.debug('Use default partition (all nodes in own community)');
    else:
      partition = initial_partition;
      # We should check whether partition is correct
      logging.debug('Use initial partition');
    # Do one iteration of optimisation
    improv = self.move_nodes(partition);
    # As long as there remains improvement iterate
    while improv > self.eps:
      # First collapse graph (i.e. community graph)
      graph = partition.collapse_graph();
      # Create partition for collapsed graph
      partition2 = partition_class(graph=graph,
                                   membership=range(graph.vcount()),
                                   weight_attr='weight',
                                   size_attr='size',
                                   self_weight_attr='self_weight',
                                   **kwargs);
      logging.debug(('partition.quality()={0}, ' \
                     'partition2.quality()={1}').format(
                    partition.quality(), partition2.quality()));
      # Optimise partition for collapsed graph
      improv = self.move_nodes(partition2);
      # Make sure improvement on coarser scale is reflected on the 
      # scale of the graph as a whole.
      partition.from_coarser_partition(partition2);
    return partition;

  def move_nodes(self, partition):
    """ Move nodes to neighbouring communities such that each move improves the
    given quality function maximally (i.e. greedily).

    Parameters:
      partition -- The partition to optimise.
    """
    # Number of iterations
    itr = 0; 
    # Total improvement while moving nodes
    total_improv = 0.0; 
    # Improvement for one loop
    improv = 2*self.eps; 
    # Number of moved nodes during one loop
    nb_moves = 2*partition.graph.vcount()*self.delta; 
    # Look if we want to debug the function, in which case we will
    # calculate some additional values.
    # In particular, the following consistencies could be checked:
    # (1) - The difference in the quality function after a move should match
    #       the reported difference when calling diff_move.
    # (2) - The quality function should be exactly the same value after
    #       aggregating/collapsing the graph.
    is_debugging = (logging.getLogger().level <= logging.DEBUG);
    # As long as we keep on improving
    while improv > self.eps and \
          nb_moves > partition.graph.vcount()*self.delta and \
          itr < self.max_itr:
      itr += 1;
      nb_moves = 0;
      improv = 0.0;
      # Establish vertex order
      vertex_order = list(partition.graph.vs);
      if self.random_order:
        shuffle(vertex_order);
      # For each node
      for v in vertex_order:
        # Only take into account nodes of degree higher than zero
        if (v.degree() > 0):
          # What is the current community of the node
          v_comm = partition.membership[v.index];
          # In which communities are its neighbours
          neigh_comms = set([partition.membership[u.index] 
            for u in v.neighbors()]);
          # What is the improvement per community if we move the node to one of
          # the neighbouring communities
          improv_comms = [(comm, partition.diff_move(v, comm)) 
              for comm in neigh_comms];
          # What is the maximum improvement?
          max_comm, max_improv = max(improv_comms, key=lambda x: x[1]);
          # If we are debugging, calculate quality function
          if is_debugging: 
            q1 = partition.quality();
          # If it doesn't improve (i.e. is positive), don't move
          if max_improv <= 0.0:
            max_comm = v_comm;
            max_improv = 0.0;
          # If we actually plan to move the nove
          if (max_comm != v_comm):
            # Keep track of improvement
            improv += max_improv;
            # Actually move the node
            partition.move_node(v, max_comm);
            # Keep track of number of moves
            nb_moves += 1;
          # If we are debugging, calculate quality function
          # and report difference
          if is_debugging: 
            q2 = partition.quality();
            logging.debug(('Move node {0} from {1} to {2} ' \
                '(diff_move={3}, q2 - q1={4})').format(
              v.index, v_comm, max_comm, 
              max_improv, q2 - q1));
      # Keep track of total improvement over multiple loops
      total_improv += improv;    
    # Make sure the resulting communities are called 0,...,q-1
    # where q is the number of communities.
    partition.renumber_communities();
    return total_improv;

  # The namedtuple we will use in the bisection function
  BisectPartition = namedtuple('BisectPartition', 
      ['partition', 'bisect_value']);

  def bisect(self, \
        graph,
        partition_class,
        resolution_range):
    """ Use bisectioning on the resolution parameter in order to construct a
    resolution profile.

    Parameters:
      graph            -- The igraph.Graph to find the optimal partition on
      partition_class  -- The type of partition which will be used during
      resolution_range -- The range of resolution values that we would like to
                          scan."""
    # Helper function for cleaning values to be a stepwise function
    def clean_stepwise(bisect_values):
      # We only need to keep the changes in the bisection values
      bisect_list = sorted([(res, part.bisect_value) for res, part in
        bisect_values.iteritems()], key=lambda x: x[0]);
      for (res1, v1), (res2, v2) \
          in zip(bisect_list,
                 bisect_list[1:]):
        # If two consecutive bisection values are the same, remove the second
        # resolution parameter
        if v1 == v2:
          del bisect_values[res2];
    # We assume here that the bisection values are 
    # monotonically decreasing with increasing resolution
    # parameter values
    def ensure_monotonicity(bisect_values, new_res):
      for res, bisect_part in bisect_values.iteritems():
        # If at a lower resolution value there were lower bisection values, we
        # should update them in order to maintain monotonicity
        if res < new_res and \
           bisect_part.bisect_value < bisect_values[new_res].bisect_value:
          bisect_values[res] = bisect_values[new_res];
        # If at a higher resolution value there were higher bisection values, we
        # should update them in order to maintain monotonicity
        elif res > new_res and \
           bisect_part.bisect_value > bisect_values[new_res].bisect_value:
          bisect_values[res] = bisect_values[new_res];
    # Start actual bisectioning
    bisect_values = {}; 
    stack_res_range = [];
    # Push first range onto the stack
    stack_res_range.append(resolution_range);
    # Make sure the bisection values are calculated
    partition = self.find_partition(graph=graph, partition_class=partition_class,
                                   resolution=resolution_range[0]);
    bisect_values[resolution_range[0]] = self.BisectPartition(partition=partition,
                                bisect_value=partition.bisect_value());
    partition = self.find_partition(graph=graph, partition_class=partition_class,
                                   resolution=resolution_range[1]);
    bisect_values[resolution_range[1]] = self.BisectPartition(partition=partition,
                                bisect_value=partition.bisect_value());
    # While stack of ranges not yet empty
    while stack_res_range:
      # Get the current range from the stack
      current_range = stack_res_range.pop();
      # Get the difference in bisection values
      diff_bisect_value = abs(bisect_values[current_range[0]].bisect_value -
                              bisect_values[current_range[1]].bisect_value);
      # Get the difference in resolution parameter (in log space if 0 is not in
      # the interval (assuming only non-negative resolution parameters).
      if current_range[0] != 0:
        diff_resolution = log(current_range[1]/current_range[0]);
      else:
        diff_resolution = current_range[1] - current_range[0];
      # Check if we still want to scan a smaller interval
      logging.info('Range=[{0}, {1}], diff_res={2}, diff_bisect={3}'.format(
          current_range[0], current_range[1], diff_resolution, diff_bisect_value));
      # If we would like to bisect this interval
      if diff_bisect_value > self.min_diff_bisect_value and \
         diff_resolution > self.min_diff_resolution:
        # Determine new resolution value
        if current_range[0] != 0:
          new_res = sqrt(current_range[1]*current_range[0]);
        else:
          new_res = sum(current_range)/2.0;
        # Bisect left (push on stack)
        stack_res_range.append((current_range[0], new_res));
        # Bisect right (push on stack)
        stack_res_range.append((new_res, current_range[1]));
        # If we haven't scanned this resolution value yet,
        # do so now
        if not bisect_values.has_key(new_res):
          partition = self.find_partition(graph, partition_class,
                                         resolution=new_res);
          bisect_values[new_res] = self.BisectPartition(partition=partition,
                                      bisect_value=partition.bisect_value());
          logging.info('Resolution={0}, Resolution Value={1}'.format(new_res,
            partition.bisect_value()));
        # Because of stochastic differences in different runs, the monotonicity
        # of the bisection values might be violated, so check for any
        # inconsistencies
        ensure_monotonicity(bisect_values, new_res);
    # Ensure we only keep those resolution values for which
    # the bisection values actually changed, instead of all of them
    clean_stepwise(bisect_values);
    # Use an ordered dict so that when iterating over it, the results appear in
    # increasing order based on the resolution value.
    return OrderedDict(sorted(((res, part) for res, part in
      bisect_values.iteritems()), key=lambda x: x[0]));

class VertexPartition(object):
  """ Contains a partition of graph. 
  
  This class contains the basic implementation for optimising a partition.
  Specifically, it implements all the administration necessary to keep track of
  the partition from various points of view. Internally, it keeps track of the
  number of internal edges (or total weight), the size of the communities, the
  total incoming degree (or weight) for a community, etc... When deriving from
  this class, one can easily use this administration to provide their own
  implementation.

  In order to keep the administration up-to-date, all changes in partition
  should be done through move_node. This function moves a node from one
  community to another, and updates all the administration.

  It is possible to manually update the membership vector, and then call
  __init_admin() which completely refreshes all the administration. This is
  only possible by updating the membership vector, not by changing some of the
  other variables.

  The basic idea is that diff_move computes the difference in the quality
  function if we call move_node for the same move. Using this framework, the
  Louvain method in the optimisation class can call these general functions in
  order to optimise the quality function.
  """
  # Init
  def __init__(self, graph, membership=None, 
      weight_attr=None, 
      size_attr=None,
      self_weight_attr=None):
    """ Create a new vertex partition. 

    Parameters:
      graph            -- The igraph.Graph on which this partition is defined.
      membership=None  -- The membership vector of this partition, i.e. an
                          community number for each node. So membership[i] = c
                          implies that node i is in community c. If None, it is
                          initialised with each node in its own community.
      weight_attr=None -- What edge attribute should be used as a weight for the
                          edges? If None, the weight defaults to 1.
      size_attr=None   -- What node attribute should be used for keeping track
                          of the size of the node? In some methods (e.g. CPM or
                          Significance), we need to keep track of the total
                          size of the community. So when we aggregate/collapse
                          the graph, we should know how many nodes were in a
                          community. If None, the size of a node defaults to 1.
      self_weight_attr=None
                       -- What node attribute should be used for the self
                          weight? If None, the self_weight is
                          recalculated each time."""
    self.graph = graph;
    if not membership:
      membership = range(self.graph.vcount());
    self.membership = membership;
    self._weight_attr = weight_attr;
    self._size_attr = size_attr;
    self._self_weight_attr = self_weight_attr;
    self.__init_admin();

  def _get_weight(self, e):
    """ Get weight of edge based on attribute. """
    return e[self._weight_attr] if self._weight_attr else 1;
  def _get_size(self, v):
    """ Get size of node based on attribute. """
    return v[self._size_attr] if self._size_attr else 1;
  def _get_self_weight(self, v):
    """ Get self weight of node based on attribute (or calculate if None). """
    if self._self_weight_attr:
      return v[self._self_weight_attr];
    else:
      # We take the set of incident because self loops appear twice
      es = self.graph.es[set(self.graph.incident(v))];
      return sum([self._get_weight(e) for e in es if e.is_loop()]);

  def __init_admin(self):
    """ Initialise all the administration based on the membership vector. """
    # Determine total weight in the graph. For undirected graphs this contains
    # already twice the number of edges
    self._total_weight = sum(self.graph.strength(mode='OUT', 
                               weights=self._weight_attr));
    # Keep track of the internal weight of each community
    self._total_weight_in_comm = defaultdict(float);
    # Keep track of the total weight to a community
    self._total_weight_to_comm = defaultdict(float);
    # Keep track of the total weight from a community
    self._total_weight_from_comm = defaultdict(float);
    # Keep track of each community (i.e. which community contains which nodes)
    self.community = defaultdict(set);
    # Keep track of the size of each community
    self.csize = defaultdict(int);
    # For each node
    for v in self.graph.vs:
      # What is the community of the node
      v_comm = self.membership[v.index];
      # Add this node to the community dictionary
      self.community[v_comm].add(v.index);
      # Increase the size of this community
      self.csize[v_comm] += self._get_size(v);
      # For each outgoing edge
      for e in self.graph.es[self.graph.incident(v, mode='OUT')]:
        # Get the other end point of this edge (i.e. neighbour)
        u = self._other_node(e, v.index);
        # Get the community of the neigbour
        u_comm = self.membership[u];
        # Get the self weight of the edge 
        w = self._get_weight(e);
        # If it is an edge within a community
        if v_comm == u_comm:
          # Add the edge to the internal weight. If the graph is undirected, we
          # should divide this by two (because we will see the edge twice).
          int_weight = self._get_weight(e)/float(2.0 - self.graph.is_directed());
          self._total_weight_in_comm[v_comm] += int_weight;
          logging.debug('Add weight {0} to in_comm {1}'.format(int_weight,
            v_comm));
        # Add weight to the outgoing weight of community of v
        self._total_weight_from_comm[v_comm] += w;
        logging.debug('Add weight {0} to from_comm {1}'.format(w, v_comm));
        # Add weight to the incoming weight of community of u
        self._total_weight_to_comm[u_comm] += w;
        logging.debug('Add weight {0} to to_comm {1}'.format(w, u_comm));

  def renumber_communities(self):
    """ Renumber the communities so that they are numbered 0,...,q-1 where q is
    the number of communities. """
    id_gen = ig.UniqueIdGenerator();
    self.membership = [id_gen[c] for c in self.membership];
    self.__init_admin();

  def from_coarser_partition(self, partition):
    """ Read new communities from coarser partition assuming that the community
    represents a node in the coarser partition (with the same index as the
    community number). """
    # Read the coarser partition
    for v in self.graph.vs:
      comm_level1 = self.membership[v.index];
      comm_level2 = partition.membership[comm_level1];
      self.membership[v.index] = comm_level2;
    # Make sure to update the administration
    self.__init_admin();

  # Take membership vector from another partition
  def from_partition(self, partition):
    """ Read new partition from another partition. """
    for v in self.graph.vs:
      self.membership[v.index] = partition.membership[v.index];
    self.__init_admin();

  # Move a node (and update administration)
  def move_node(self,v,new_comm):
    """ Move a node to a new community and update the administration.
    Parameters:
      v        -- Node to move.
      new_comm -- To which community should it move."""
    # Move node and update internal administration
    # Remove from old community
    old_comm = self.membership[v.index];
    self.community[old_comm].remove(v.index);
    self.csize[old_comm] -= self._get_size(v);
    # Add to new community
    self.community[new_comm].add(v.index);
    self.csize[new_comm] += self._get_size(v);
    # Switch outgoing links
    # Use set for incident edges, because self loop appears twice
    for e in self.graph.es[self.graph.incident(v, mode='OUT')]:
      # Get other node from edge (i.e. neighbour)
      u = self._other_node(e, v.index);
      # Get community for neighbour
      u_comm = self.membership[u];
      # Get edge weight
      w = self._get_weight(e);
      # Get internal weight (if it is an internal edge)
      # If the graph is directed, we need to divide by two, because we loop
      # over the outgoing and incoming links. Moreover, if a weight is a loop,
      # it is also included twice in the incident function of igraph, so we'll
      # have to take that into account as well.
      int_weight = w/(float(2.0 - self.graph.is_directed())*(e.is_loop() + 1));
      # If it is an internal edge in the old community
      if old_comm == u_comm:
        # Remove the internal weight
        self._total_weight_in_comm[old_comm] -= int_weight;
        logging.debug(('From link ({0}-{1}) ' \
            'remove internal weight {2} from {3}').format(
              v.index, u, int_weight, old_comm));
      # If it is an internal edge in the old community
      if new_comm == u_comm or e.is_loop():
        # Add the internal weight
        self._total_weight_in_comm[new_comm] += int_weight;
        logging.debug(('From link ({0}-{1}) ' \
            'add internal weight {2} to {3}').format(
                v.index, u, int_weight, new_comm));
      # Remove the weight from the outgoing weights of the old community
      self._total_weight_from_comm[old_comm] -= w;
      # Add the weight to the outgoing weights of the new community
      self._total_weight_from_comm[new_comm] += w;
      logging.debug(('Moving link ({0}-{1}) ' \
          'outgoing weight {2} from {3} to {4}').format(
            v.index, u, w, old_comm, new_comm));
    # Switch incoming links
    # Use set for incident edges, because self loop appears twice
    for e in self.graph.es[self.graph.incident(v, mode='IN')]:
      # Get other node from edge (i.e. neighbour)
      u = self._other_node(e, v.index);
      # Get community for neighbour
      u_comm = self.membership[u];
      # Get edge weight
      w = self._get_weight(e);
      # Get internal weight (if it is an internal edge)
      # If the graph is directed, we need to divide by two, because we loop
      # over the outgoing and incoming links. Moreover, if a weight is a loop,
      # it is also included twice in the incident function of igraph, so we'll
      # have to take that into account as well.
      int_weight = w/(float(2.0 - self.graph.is_directed())*(e.is_loop() + 1));
      # If it is an internal edge in the old community
      if old_comm == u_comm:
        # Remove the internal weight
        self._total_weight_in_comm[old_comm] -= int_weight;
        logging.debug(('From link ({0}-{1}) ' \
            'remove internal weight from {3}').format(
              v.index, u, int_weight, old_comm));
      # If it is an internal edge in the new community
      if new_comm == u_comm or e.is_loop():
        # Add the internal weight
        self._total_weight_in_comm[new_comm] += int_weight;
        logging.debug(('From link ({0}-{1}) ' \
            'add internal weight {2} to {3}').format(
              v.index, u, int_weight, new_comm));
      # Remove the weight from the incoming weights of the old community
      self._total_weight_to_comm[old_comm] -= w;
      # Add the weight from the incoming weights of the new community
      self._total_weight_to_comm[new_comm] += w;
      logging.debug(('Moving link ({0}-{1}) ' \
          'outgoing weight {2} from {3} to {4}').format(
            v.index, u, w, old_comm, new_comm));
    # Update the membership vector
    self.membership[v.index] = new_comm;

  # Calculate improvement *if* we move this node
  def diff_move(self,v,new_comm):
    """ Calculate the difference in the quality function if we were to move
    this node. In this base class, the quality function is always simply 0.0,
    and so the diff_move is also always 0.0. For implementing actual quality
    functions, one should derive from this base class and implement their own
    diff_move and quality funcion.
    
    The difference returned by diff_move should be equivalent to first
    determining the quality of the partition, then calling move_node, and then
    determining again the quality of the partition and looking at the
    difference. In other words

    diff = partition.diff_move(v, new_comm);
    q1 = partition.quality();
    move_node(v, new_comm);
    q2 = partition.quality();

    Then diff == q2 - q1."""
    return 0.0;

  def subgraph(self, comm):
    """ Get subgraph consisting of a community. """
    return self.graph.induced_subgraph(vertices=self.community[comm]);

  def _other_node(self, e, v):
    """ Get other node from edge. """
    if e.source == v:
      return e.target;
    else:
      return e.source;

  # Get weight from v to comm
  def _weight_to_comm(self, v, comm):
    """ Calculate what is the total weight going from a node to a community.
    
    Parameters:
      v      -- The node which to check.
      comm   -- The community which to check."""
    w = 0.0;
    # We take the set of incident because self loops appear twice
    for e in self.graph.es[set(self.graph.incident(v, mode='OUT'))]:
      u = self._other_node(e, v.index);
      u_comm = self.membership[u];
      if u_comm == comm:
        w += self._get_weight(e);
    return w;

  # Get weight from comm to v
  def _weight_from_comm(self, v, comm):
    """ Calculate what is the total weight going from a community to a node.
    
    Parameters:
      v      -- The node which to check.
      comm   -- The community which to check."""
    w = 0.0;
    # We take the set of incident because self loops appear twice
    for e in self.graph.es[set(self.graph.incident(v, mode='IN'))]:
      u = self._other_node(e, v.index);
      u_comm = self.membership[u];
      if u_comm == comm:
        w += self._get_weight(e);
    return w;

  def collapse_graph(self):
    """ Return a graph with communities as node and links as weights between
    communities.

    The weight of the edges in the new graph is simply the sum of the weight
    of the edges between the communities. The self weight of a node (i.e. the
    weight of its self loop) is the internal weight of a community. The size
    of a node in the new graph is simply the size of the community in the old
    graph."""
    non_empty_csize = [(comm, size) for comm, size in 
        self.csize.iteritems() if size > 0];
    id_gen = ig.UniqueIdGenerator();
    H = ig.Graph(directed=self.graph.is_directed());
    for node, nsize in non_empty_csize:
      id_gen.add(node);
      H.add_vertex(name=node);
      H.vs[id_gen[node]]['size'] = nsize;
      H.vs[id_gen[node]]['self_weight'] = self._total_weight_in_comm[node];
    edges = defaultdict(lambda: defaultdict(float));
    for e in self.graph.es:
      s_comm = id_gen[self.membership[e.source]];
      t_comm = id_gen[self.membership[e.target]];
      logging.debug('Node (from) {0}, community {1}, new node {2}'.format(
        e.source, self.membership[e.source], id_gen[self.membership[e.source]]));
      logging.debug('Node (to) {0}, community {1}, new node {2}'.format(
        e.target, self.membership[e.target], id_gen[self.membership[e.target]]));
      if not self.graph.is_directed():
        s_comm, t_comm = min(s_comm, t_comm), max(s_comm, t_comm);
      edges[s_comm][t_comm] += self._get_weight(e);
    e1, e2, weight = zip(*[(s, t, w) 
                              for s,neigh in edges.iteritems() 
                              for t,w in neigh.iteritems()]);
    H.add_edges(zip(e1, e2));
    H.es['weight'] = weight;
    return H;

  def quality(self):
    """ Give the quality of the partition. 
    
    This function currently returns 0.0, and should be overridden in a derived
    class. It should provide a single measure of how 'good' this partition is
    (e.g. modularity, significance, CPM, etc...). """
    return 0.0;

  def modularity(self):
    """ Give the modularity of the partition.

    We here use the unscaled version of modularity, in other words, we don't
    normalise by the number of edges. """
    mod = 0.0;
    for c in self.community.keys():
      w = self._total_weight_in_comm[c];
      w_out = self._total_weight_from_comm[c];
      w_in = self._total_weight_to_comm[c];
      logging.debug('Comm: {0}, w={1}, w_out={2}, w_in={3}'.format(c, w, w_out,
        w_in));
      mod = mod + w - \
        w_out*w_in/(float(4 - 3*self.graph.is_directed())*self._total_weight);
    return mod;

  def significance(self):
    """ Give the significance of the partition. """
    logging.debug('Calculating significance...');
    S = 0.0;
    n = sum(self._get_size(v) for v in self.graph.vs);
    p = self._total_weight/float(n*(n - 1));
    logging.debug('n={0}. m={1}, p={2}'.format(n, self._total_weight, p));
    for c, n_c in self.csize.iteritems():
      m_c = self._total_weight_in_comm[c];
      if n_c > 1:
        p_c = m_c/float(n_c*(n_c - 1)/float(2 - self.graph.is_directed()));
        logging.debug('n_c={0}, m_c={1}, p_c={2}'.format(n_c, m_c, p_c));
        S += self._KL(p_c, p)*n_c*(n_c - 1);
      else:
        logging.debug('n_c={0}, m_c={1}, p_c=0.0'.format(n_c, m_c));
    return S;

  def _KL(self,q,p):
    """ The binary Kullback-Leibler divergence. """
    KL = 0;
    if q > 0 and p > 0:
      KL += q*log(q/p);
    if q < 1 and p < 1:
      KL += (1-q)*log((1-q)/(1-p));
    return KL;

  def _entropy(self):
    """ The entropy of the community sizes. """
    n = float(self.graph.vcount());
    h = -sum((n_c/n)*log(n_c/n) for n_c in self.csize.values());
    return h;

  def similarity_MI(self,partition):
    """ The mutual information with another partition. """
    n = float(self.graph.vcount());
    I = 0.0;
    for r, nodes_r in self.community.iteritems():
      for s, nodes_s in partition.community.iteritems():
        n_r = float(len(nodes_r));
        n_s = float(len(nodes_s));
        n_rs = len(nodes_r.intersection(nodes_s));
        if n_rs > 0:
          I += (n_rs/n)*log(n * n_rs / (n_r * n_s) );
    return I;
  
  def similarity_NMI(self, partition):
    """ The normalised mutual information with another partition. """
    return 2*self.similarity_MI(partition)/\
        (self._entropy() + partition._entropy());

  def distance_VI(self, partition):
    """ The variance of information distance to another partition. """
    return self._entropy() + partition._entropy() - \
        2*self.similarity_MI(partition);

  def similarity(self, partition):
    """ The similarity to another partition expressed in normalised mutual
    information (NMI). """
    return self.similarity_NMI(partition);

  def distance(self, partition):
    """ The distance to another partition expressed in variation of
    information (VI). """
    return self.distance_VI(partition);

class ModularityVertexPartition(VertexPartition):
  """ Implements the diff_move and quality function in order to optimise
  modularity. """
  def diff_move(self, v, new_comm):
    """ Returns the difference in modularity if we move a node to a new
    community. """
    logging.debug('Enter diff_move({0}, {1})'.format(v, new_comm));
    old_comm = self.membership[v.index];
    if (new_comm == old_comm):
      return 0.0;
    else:
      logging.debug('old_comm: {0}'.format(old_comm));
      w_to_old = self._weight_to_comm(v, old_comm);
      logging.debug('w_to_old: {0}'.format(w_to_old));
      w_from_old = self._weight_from_comm(v, old_comm);
      logging.debug('w_from_old: {0}'.format(w_from_old));
      w_to_new = self._weight_to_comm(v, new_comm);
      logging.debug('w_to_new: {0}'.format(w_to_new));
      w_from_new = self._weight_from_comm(v, new_comm);
      logging.debug('w_from_new: {0}'.format(w_from_new));
      k_out = v.strength(mode='OUT', weights=self._weight_attr);
      logging.debug('k_out: {0}'.format(k_out));
      k_in = v.strength(mode='IN', weights=self._weight_attr);
      logging.debug('k_in: {0}'.format(k_in));
      self_weight = self._get_self_weight(v);
      logging.debug('self_weight: {0}'.format(self_weight));
      K_out_old = self._total_weight_from_comm[old_comm];
      logging.debug('K_out_old: {0}'.format(K_out_old));
      K_in_old = self._total_weight_to_comm[old_comm];
      logging.debug('K_in_old: {0}'.format(K_in_old));
      K_out_new = self._total_weight_from_comm[new_comm] + k_out;
      logging.debug('K_out_new: {0}'.format(K_out_new));
      K_in_new = self._total_weight_to_comm[new_comm] + k_in;
      logging.debug('K_in_new: {0}'.format(K_in_new));
      total_weight = self._total_weight*float(2 - self.graph.is_directed());
      logging.debug('total_weight: {0}'.format(total_weight));
      diff_old = (w_to_old - k_out*K_in_old/total_weight) + \
                 (w_from_old - k_in*K_out_old/total_weight);
      logging.debug('diff_old: {0}'.format(diff_old));
      diff_new = (w_to_new + self_weight - k_out*K_in_new/total_weight) + \
                 (w_from_new + self_weight - k_in*K_out_new/total_weight);
      logging.debug('diff_new: {0}'.format(diff_new));
      diff = diff_new - diff_old;
      if (not self.graph.is_directed()):
        diff /= 2.0;
      logging.debug('diff: {0}'.format(diff));
      return diff;

  def quality(self):
    """ Returns the modularity (as was already implemented in the base class) """
    return self.modularity();

class LinearResolutionParameterVertexPartition(VertexPartition):
  """ Some quality functions have a linear resolution parameter, for which the
  basis is implemented here.

  With a linear resolution parameter, we mean that the objective function has
  the form:

  H = E - gamma F

  where gamma is some resolution parameter and E and F arbitrary other
  functions of the partition. 
  
  One thing that can be easily done on these type of quality functions, is
  bisectioning on the gamma function (also assuming that E is a stepwise
  decreasing monotonic function, cf. CPM).
  """
  def __init__(self, graph, resolution, membership=None, 
      weight_attr=None, 
      size_attr=None,
      self_weight_attr=None):
    VertexPartition.__init__(self, graph, membership, 
        weight_attr , size_attr, self_weight_attr);
    self.resolution = resolution;
  
  def bisect_value(self):
    """  Give the value on which we can perform bisectioning. If p1 and p2 are
    two different optimal partitions for two different resolution parameters
    g1 and g2, then if p1.bisect_value() == p2.bisect_value() the two
    partitions should be optimal for both g1 and g2."""
    return 0.0;

class SignificanceVertexPartition(VertexPartition):
  """ Implementation of significance of a vertex partition. """
  def diff_move(self, v, new_comm):
    """ Returns the difference in significance if we move a node to a new
    community. """
    logging.debug('Enter diff_move({0}, {1})'.format(v, new_comm));
    old_comm = self.membership[v.index];
    nsize = self._get_size(v);
    if (new_comm == old_comm):
      return 0.0;
    else:
      n = sum([self._get_size(node) for node in self.graph.vs]);
      normalise = float(2 - self.graph.is_directed());
      p = self._total_weight/float(n*(n - 1));
      logging.debug('Community: {0} => {1}'.format(old_comm, new_comm));
      logging.debug('n: {0}, m: {1} p: {2}'.format(n, self._total_weight, p));
      # Old comm
      n_old = self.csize[old_comm];
      m_old = self._total_weight_in_comm[old_comm];
      if n_old > 1:
        q_old = m_old/float(n_old*(n_old - 1)/normalise)
      else:
        q_old = 0.0;
      logging.debug('n_old: {0}, m_old: {1}, q_old: {2}'.format(
                    n_old, m_old, q_old));

      # Old comm after move
      n_oldx = n_old - nsize;
      wtc = self._weight_to_comm(v, old_comm);
      wfc = self._weight_from_comm(v, old_comm);
      sw = self._get_self_weight(v);
      logging.debug('wtc: {0}, wfc: {1}, sw: {2}'.format(wtc, wfc, sw));
      m_oldx = m_old - wtc/normalise - wfc/normalise - sw;
      if n_oldx > 1:
        q_oldx = m_oldx/float(n_oldx*(n_oldx - 1)/normalise)
      else:
        q_oldx = 0.0;
      logging.debug('n_oldx: {0}, m_oldx: {1}, q_oldx: {2}'.format(
                    n_oldx, m_oldx, q_oldx));

      # New comm
      n_new = self.csize[new_comm];
      m_new = self._total_weight_in_comm[new_comm];
      if n_new > 1:
        q_new = m_new/float(n_new*(n_new - 1)/normalise)
      else:
        q_new = 0.0;
      logging.debug('n_new: {0}, m_new: {1}, q_new: {2}'.format(
                    n_new, m_new, q_new));

      # New comm after move
      n_newx = n_new + nsize;
      wtc = self._weight_to_comm(v, new_comm);
      wfc = self._weight_from_comm(v, new_comm);
      sw = self._get_self_weight(v);
      logging.debug('wtc: {0}, wfc: {1}, sw: {2}'.format(wtc, wfc, sw));
      m_newx = m_new + wtc/normalise + wfc/normalise + sw;
      if n_newx > 1:
        q_newx = m_newx/float(n_newx*(n_newx - 1)/normalise)
      else:
        q_newx = 0.0;
      logging.debug('n_newx: {0}, m_newx: {1}, q_newx: {2}'.format(
                    n_newx, m_newx, q_newx));

      diff = - n_old*(n_old-1)*self._KL(q_old, p) \
             + n_oldx*(n_oldx-1)*self._KL(q_oldx, p) \
             - n_new*(n_new-1)*self._KL(q_new, p) \
             + n_newx*(n_newx-1)*self._KL(q_newx, p);
      logging.debug('diff: {0}'.format(diff));
      return diff;

  def quality(self):
    """ Returns the significance (as was already implemented in the base
    class). """
    return self.significance();

class RBConfigurationVertexPartition(LinearResolutionParameterVertexPartition):
  """ Implements the diff_move and quality function in order to optimise
  RB model with a configuration null model, i.e. modularity with a
  multiplicative linear resolution parameter. """
  def diff_move(self, v, new_comm):
    """ Returns the difference in modularity if we move a node to a new
    community. """
    logging.debug('Enter diff_move({0}, {1})'.format(v, new_comm));
    old_comm = self.membership[v.index];
    if (new_comm == old_comm):
      return 0.0;
    else:
      logging.debug('old_comm: {0}'.format(old_comm));
      w_to_old = self._weight_to_comm(v, old_comm);
      logging.debug('w_to_old: {0}'.format(w_to_old));
      w_from_old = self._weight_from_comm(v, old_comm);
      logging.debug('w_from_old: {0}'.format(w_from_old));
      w_to_new = self._weight_to_comm(v, new_comm);
      logging.debug('w_to_new: {0}'.format(w_to_new));
      w_from_new = self._weight_from_comm(v, new_comm);
      logging.debug('w_from_new: {0}'.format(w_from_new));
      k_out = v.strength(mode='OUT', weights=self._weight_attr);
      logging.debug('k_out: {0}'.format(k_out));
      k_in = v.strength(mode='IN', weights=self._weight_attr);
      logging.debug('k_in: {0}'.format(k_in));
      self_weight = self._get_self_weight(v);
      logging.debug('self_weight: {0}'.format(self_weight));
      K_out_old = self._total_weight_from_comm[old_comm];
      logging.debug('K_out_old: {0}'.format(K_out_old));
      K_in_old = self._total_weight_to_comm[old_comm];
      logging.debug('K_in_old: {0}'.format(K_in_old));
      K_out_new = self._total_weight_from_comm[new_comm] + k_out;
      logging.debug('K_out_new: {0}'.format(K_out_new));
      K_in_new = self._total_weight_to_comm[new_comm] + k_in;
      logging.debug('K_in_new: {0}'.format(K_in_new));
      total_weight = self._total_weight*float(2 - self.graph.is_directed());
      logging.debug('total_weight: {0}'.format(total_weight));
      diff_old = (w_to_old - self.resolution*k_out*K_in_old/total_weight) + \
                 (w_from_old - self.resolution*k_in*K_out_old/total_weight);
      logging.debug('diff_old: {0}'.format(diff_old));
      diff_new = (w_to_new + self_weight - \
                      self.resolution*k_out*K_in_new/total_weight) + \
                 (w_from_new + self_weight - \
                     self.resolution*k_in*K_out_new/total_weight);
      logging.debug('diff_new: {0}'.format(diff_new));
      diff = diff_new - diff_old;
      if (not self.graph.is_directed()):
        diff /= 2.0;
      logging.debug('diff: {0}'.format(diff));
      return diff;

  def quality(self):
    """ Returns the modularity (as was already implemented in the base class) """
    return self.RBConfiguration();

  def RBConfiguration(self):
    """ Give the RB configuration null model value of the partition.

    We here use the unscaled version, in other words, we don't normalise by
    the number of edges. """
    mod = 0.0;
    for c in self.community.keys():
      w = self._total_weight_in_comm[c];
      w_out = self._total_weight_from_comm[c];
      w_in = self._total_weight_to_comm[c];
      logging.debug('Comm: {0}, w={1}, w_out={2}, w_in={3}'.format(c, w, w_out,
        w_in));
      mod = mod + w - \
        self.resolution*\
        w_out*w_in/(float(4 - 3*self.graph.is_directed())*self._total_weight);
    return mod;

  def bisect_value(self):
    """ Bisection takes place on the total internal weight. """
    return sum(self._total_weight_in_comm.values());    

class CPMVertexPartition(LinearResolutionParameterVertexPartition):
  """ CPM implementation of a vertex partition (which includes a resolution
  parameter) """
  def diff_move(self, v, new_comm):
    """ Returns the difference in CPM if we move a node. """
    logging.debug('Enter diff_move({0}, {1})'.format(v, new_comm));
    old_comm = self.membership[v.index];
    if (new_comm == old_comm):
      return 0.0;
    else:
      w_to_old = self._weight_to_comm(v, old_comm);
      logging.debug('w_to_old: {0}'.format(w_to_old));
      w_to_new = self._weight_to_comm(v, new_comm);
      logging.debug('w_to_new: {0}'.format(w_to_new));
      w_from_old = self._weight_from_comm(v, old_comm);
      logging.debug('w_from_old: {0}'.format(w_from_old));
      w_from_new = self._weight_from_comm(v, new_comm);
      logging.debug('w_from_new: {0}'.format(w_from_new));
      nsize = self._get_size(v);
      logging.debug('nsize: {0}'.format(nsize));
      csize_old = self.csize[old_comm];
      logging.debug('csize_old: {0}'.format(csize_old));
      csize_new = self.csize[new_comm];
      logging.debug('csize_new: {0}'.format(csize_new));
      self_weight = self._get_self_weight(v);
      logging.debug('self_weight: {0}'.format(self_weight));
      diff_old = (w_to_old + w_from_old - 
          self_weight - self.resolution*2*nsize*csize_old);
      logging.debug('diff_old: {0}'.format(diff_old));
      diff_new = (w_to_new + w_from_new + self_weight - 
          self.resolution*2*nsize*(csize_new + nsize));
      logging.debug('diff_new: {0}'.format(diff_new));
      diff = diff_new - diff_old;
      logging.debug('diff: {0}'.format(diff));
      return diff;

  def CPM(self):
    """ Returns the CPM value (using the indicated resolution parameter) for
    this partition. """
    mod = 0.0;
    for c,csize in self.csize.iteritems():
      w = self._total_weight_in_comm[c];
      logging.debug('Comm: {0}, w_c={1}, n_c={2}'.format(c, w, csize));
      mod = mod + w*float(2 - self.graph.is_directed()) - self.resolution*(csize**2);
    return mod;

  def quality(self):
    """ Returns CPM(). """
    return self.CPM();

  def bisect_value(self):
    """ Bisection takes place on the total internal weight. """
    return sum(self._total_weight_in_comm.values());    

