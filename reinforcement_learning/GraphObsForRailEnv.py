from flatland.core.env_observation_builder import ObservationBuilder
from typing import Optional, List
import numpy as np
from flatland.core.env import Environment
import collections
from typing import Optional, List, Dict, Tuple

import numpy as np
import torch
from flatland.core.env import Environment
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.env_prediction_builder import PredictionBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.core.grid.grid_utils import coordinate_to_position
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.fast_methods import fast_argmax, fast_count_nonzero, fast_position_equal, fast_delete, fast_where
from flatland.envs.step_utils.states import TrainState
from flatland.utils.ordered_set import OrderedSet
from torch_geometric.data import Data

class GraphObsForRailEnv(ObservationBuilder):
    """
    AMOS THIS IS THE OLD DESC:
    Gives a global observation of the entire rail environment.
    The observation is composed of the following elements:

        - transition map array with dimensions (env.height, env.width, 16),\
          assuming 16 bits encoding of transitions.
    
        - obs_agents_state: A 3D array (map_height, map_width, 5) with
            - first channel containing the agents position and direction
            - second channel containing the other agents positions and direction
            - third channel containing agent/other agent malfunctions
            - fourth channel containing agent/other agent fractional speeds
            - fifth channel containing number of other agents ready to depart

        - obs_targets: Two 2D arrays (map_height, map_width, 2) containing respectively the position of the given agent\
         target and the positions of the other agents targets (flag only, no counter!).
    """

    def __init__(self, predictor):
        super(GraphObsForRailEnv, self).__init__()
        self.predictor = predictor
        self.observation_dim = 36

    def set_env(self, env: Environment):
        super().set_env(env)

    def reset(self):
        self.rail_obs = np.zeros((self.env.height, self.env.width, 16))
        for i in range(self.rail_obs.shape[0]):
            for j in range(self.rail_obs.shape[1]):
                bitlist = [int(digit) for digit in bin(self.env.rail.get_full_transitions(i, j))[2:]]
                bitlist = [0] * (16 - len(bitlist)) + bitlist
                self.rail_obs[i, j] = np.array(bitlist)

        non_zero_indices = np.argwhere(np.sum(self.rail_obs, axis=2) != 0)
        num_nodes = non_zero_indices.shape[0]
        node_lookup = {}
        for i, coord in enumerate(non_zero_indices):
            coords = tuple(coord)
            transition_matrix = np.reshape(self.rail_obs[coords],(4,4))
            leave_directions = np.argwhere(np.sum(transition_matrix,axis=0)!=0)
            enter_directions = np.argwhere(np.sum(transition_matrix,axis=1)!=0)
            node_lookup[coords] = [i, self.rail_obs[coords], leave_directions, enter_directions, 1] # 1 stand for the length of the segment, later we collapse straights, which have longer length

        # colapse straights to one
        n_south_rail = np.array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.])
        w_east_rail = np.array([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.])
        node_collapsed = set() # keys are coordinates and point to the same i, if they where collapsed
        for coords, (i, rail_bit_directions, leave_directions, enter_directions, length) in node_lookup.items():
            if i in node_collapsed: # straight is already collapsed by one of the while loops below
                continue
            if np.array_equal(rail_bit_directions,n_south_rail):
                length_down = 1
                # check both directions (up and down)
                # travel downwards
                collapsed_coords = (coords[0]+length_down,coords[1])
                collapsed_list = []
                while collapsed_coords in node_lookup and np.array_equal(node_lookup[collapsed_coords][1], n_south_rail):
                    length_down += 1
                    collapsed_list.append(collapsed_coords)
                    collapsed_coords = (coords[0]+length_down,coords[1])

                #travel upwards
                length_up = 1
                collapsed_coords = (coords[0]-length_up,coords[1])
                while collapsed_coords in node_lookup and np.array_equal(node_lookup[collapsed_coords][1], n_south_rail):
                    length_up += 1
                    collapsed_list.append(collapsed_coords)
                    collapsed_coords = (coords[0]-length_up,coords[1])
                
                collapsed_list.append(coords)
                length_of_straight = len(collapsed_list) 
                for collapsed_coords in collapsed_list:
                    # node_lookup_collapsed[collapsed_coords] = (i, length_of_straight)
                    node_collapsed.add(collapsed_coords)
                    temp = node_lookup[collapsed_coords] 
                    node_lookup[collapsed_coords] = (i, temp[1], temp[2], temp[3], length_of_straight)

            elif np.array_equal(rail_bit_directions,w_east_rail):
                length_right = 1
                # check both directions (left and right)
                # travel right
                collapsed_coords = (coords[0],coords[1]+length_right)
                collapsed_list = []
                while collapsed_coords in node_lookup and  np.array_equal(node_lookup[collapsed_coords][1], w_east_rail):
                    length_right += 1
                    collapsed_list.append(collapsed_coords)
                    collapsed_coords = (coords[0],coords[1]+length_right)

                #travel left
                length_left = 1
                collapsed_coords = (coords[0],coords[1]-length_left)
                while collapsed_coords in node_lookup and  np.array_equal(node_lookup[collapsed_coords][1],w_east_rail):
                    length_left += 1
                    collapsed_list.append(collapsed_coords)
                    collapsed_coords = (coords[0],coords[1]-length_left)
                
                collapsed_list.append(coords)
                length_of_straight = len(collapsed_list) 
                for collapsed_coords in collapsed_list:
                    temp = node_lookup[collapsed_coords]
                    node_collapsed.add(collapsed_coords)
                    node_lookup[collapsed_coords] = (i, temp[1], temp[2], temp[3], length_of_straight)

        # clean the node lookup (remove gaps in the indices)
        node_lookup_cleaned = {}
        index = 0
        set_i_mapping = {}
        for coords, (i, rail_matrix, leave_directions, enter_directions, length) in node_lookup.items():
            if i in set_i_mapping:
                curr_index = set_i_mapping[i]
            else:
                curr_index = index
                set_i_mapping[i] = index
                index += 1

            node_lookup_cleaned[coords] = (curr_index, rail_matrix, leave_directions, enter_directions, length)

        num_nodes = index # not -1, because we start at 0

        self.adjacency_matrix = np.zeros((num_nodes,num_nodes))
        # feature matrix contains
        num_nodes = len(node_lookup_cleaned)
        edge_index = []
        # the base rail type (16 bit), the length of the segment
        self.rail_types_facing_north = torch.zeros((num_nodes, 16))
        self.rail_types_facing_east = torch.zeros((num_nodes, 16))
        self.rail_types_facing_south = torch.zeros((num_nodes, 16))
        self.rail_types_facing_west = torch.zeros((num_nodes, 16))

        self.rail_lengths = torch.zeros((num_nodes, 1))
        #self.edge_index_index = {}
        
        for coords, (i, rail_byte, leave_directions, enter_directions, lengths) in node_lookup_cleaned.items():
            # (for straights, the allowed leave directions are the same as the allowed enter directions and the same for all collapsed segments
            # therefore there is no issue with all of them sharing the same features)
            # self.rail_types[i] = torch.tensor(rail_byte, dtype=torch.float)
            byte_matrix = np.reshape(rail_byte, (4,4))
            self.rail_types_facing_north[i] = torch.tensor(rail_byte, dtype=torch.float)
            # make the first col and row be the last col and row for east
            east = np.reshape(byte_matrix[[1,2,3,0]][:,[1,2,3,0]], 16)
            self.rail_types_facing_east[i] = torch.tensor(east, dtype=torch.float)
            south = np.reshape(byte_matrix[[2,3,0,1]][:,[2,3,0,1]], 16)
            self.rail_types_facing_south[i] = torch.tensor(south, dtype=torch.float)
            west = np.reshape(byte_matrix[[3,0,1,2]][:,[3,0,1,2]], 16)
            self.rail_types_facing_west[i] = torch.tensor(west, dtype=torch.float)

            self.rail_lengths[i] = torch.tensor([node_lookup_cleaned[coords][4]], dtype=torch.float)

            #coming from n, e, s, w

            # traveling one of the allowed leave directions, can we enter the neighboring cell? If so, they are neighbors in the graph (directed)
            for direction in leave_directions:
                direction = [(0,(0,1)),(1,(-1,0)),(2,(0,-1)),(3,(1,0))][direction[0]]
                new_coords = (coords[0] + direction[1][0], coords[1] + direction[1][1])
                if new_coords in node_lookup_cleaned:
                    # j already respects the collapsing of straights
                    j, neighbor_rail_byte, neighbor_leave_directions, neighbor_enter_directions, neighbor_length = node_lookup_cleaned[new_coords]
                    if np.isin(direction[0], enter_directions):
                        # adjacency_matrix[i,j] = 1
                        edge_index.append([i,j])
                        index = len(edge_index) - 1
                        #self.edge_index_index[(i,j)] = index
                    


        # create the base graph structure
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        # self.graph = Data(edge_index=edge_index)
        self.num_nodes = num_nodes
        self.node_lookup = node_lookup_cleaned

        self.node_index_to_coords = {}
        for coords, (i, rail_byte, leave_directions, enter_directions, lengths) in node_lookup_cleaned.items():
            if i in self.node_index_to_coords:
                self.node_index_to_coords[i].append(coords)
            else:
                self.node_index_to_coords[i] = [coords]

        # this "returns": edge_index, num_nodes, rail_types (north, east, south, west), rail_lengths, node_lookup, node_index_to_coords

    def get_many(self, handles: Optional[List[int]] = None):
        """
        Called whenever an observation has to be computed for the `env` environment, for each agent with handle
        in the `handles` list.
        """

        if handles is None:
            handles = []
        if self.predictor:
            self.max_prediction_depth = 0
            self.predicted_pos = {}
            self.predicted_dir = {}
            self.predictions = self.predictor.get()
            if self.predictions:
                for t in range(self.predictor.max_depth + 1):
                    pos_list = []
                    dir_list = []
                    for a in handles:
                        if self.predictions[a] is None:
                            continue
                        pos_list.append(self.predictions[a][t][1:3])
                        dir_list.append(self.predictions[a][t][3])
                    self.predicted_pos.update({t: coordinate_to_position(self.env.width, pos_list)})
                    self.predicted_dir.update({t: dir_list})
                self.max_prediction_depth = len(self.predicted_pos)
        # Update local lookup table for all agents' positions
        # ignore other agents not in the grid (only status active and done)
        # self.location_has_agent = {tuple(agent.position): 1 for agent in self.env.agents if
        #                         agent.status in [RailAgentStatus.ACTIVE, RailAgentStatus.DONE]}

        self.location_has_agent = {}
        self.location_has_agent_direction = {}
        self.location_has_agent_speed = {}
        self.location_has_agent_malfunction = {}
        self.location_has_agent_ready_to_depart = {}
        self.location_has_agent_id = {}

        for _agent in self.env.agents:
            if not _agent.state.is_off_map_state() and \
                _agent.position:
                self.location_has_agent[tuple(_agent.position)] = 1
                self.location_has_agent_id[tuple(_agent.position)] = _agent.handle
                self.location_has_agent_direction[tuple(_agent.position)] = _agent.direction
                self.location_has_agent_speed[tuple(_agent.position)] = _agent.speed_counter.speed
                self.location_has_agent_malfunction[tuple(_agent.position)] = \
                    _agent.malfunction_handler.malfunction_down_counter

            # [NIMISH] WHAT IS THIS
            if _agent.state.is_off_map_state() and \
                _agent.initial_position:
                self.location_has_agent_ready_to_depart.setdefault(tuple(_agent.initial_position), 0)
                self.location_has_agent_ready_to_depart[tuple(_agent.initial_position)] += 1
            # self.location_has_agent_ready_to_depart[tuple(_agent.initial_position)] = \
            #     self.location_has_agent_ready_to_depart.get(tuple(_agent.initial_position), 0) + 1

        observations = super().get_many(handles)

        return observations

    def get(self, handle: int = 0) -> (Data, torch.tensor, torch.tensor):  # returns a pyg graph, the index of the node where the agent is, the index of the target node

        agent = self.env.agents[handle]
        if agent.state.is_off_map_state():
            agent_virtual_position = agent.initial_position
        elif agent.state.is_on_map_state():
            agent_virtual_position = agent.position
        elif agent.state == TrainState.DONE:
            agent_virtual_position = agent.target
        else:
            return None
        
        # x contains 
        # 1. rail type 16 
        # 2. length of the segment 1
        # If obj on node: 
        # 3. if self 1
        # 3.5.1 self.speed 1
        # 3.5.2 self direction 4
        # 3.5.2 dist to target 1
        # 4. if self target 1
        # 5. if other 1
        # 5.5 other direction 4
        # 6. if other target 1
        # 7. if other malfunction 1 (malfunction counter, how long it will take to be fixed)
        # 8. if other speed, min fractional speed otherwise 1
        # 9. number of other agents ready to depart but not yet active 1
        # 10 using shortest path predictor: if other agent will be on this node in the future 1
        # 11 timestep when other agent will be on this node 1

        # Here information about the agent itself is stored
        x = torch.zeros((self.num_nodes, 36))
        x[node, 32] = 1.0  # 8. Other agent speed (min fractional speed)
        if agent.direction == 0:
            x[:,0:16] = self.rail_types_facing_north
        elif agent.direction == 1:
            x[:,0:16] = self.rail_types_facing_east
        elif agent.direction == 2:
            x[:,0:16] = self.rail_types_facing_south
        elif agent.direction == 3:
            x[:,0:16] = self.rail_types_facing_west

        x[:,16] = self.rail_lengths.squeeze()

        distance_map = self.env.distance_map.get()
        dist_min_to_target = distance_map[
                                         (handle, *agent_virtual_position,
                                          agent.direction)]
        index_of_node = -1
        index_of_target = -1
        for node in range(self.graph.num_nodes):
            positions = self.node_index_to_coords[node]  # assuming you have a mapping of node indices to positions
            
            for position in positions:
                # this will be non exact for the collapsed straights (maybe works anyways)
                if position == agent_virtual_position:  # 3. If self (agent) is on this node
                    x[node, 17] = 1

                    x[node, 18] = agent.speed_counter.speed  # 3.5.1 Self speed
                    
                    x[node, 19 + agent.direction] = 1  # 3.5.2 Self direction (one-hot encoding)
                   
                    x[node, 23] = dist_min_to_target  # 3.5.3 Distance to target
                    index_of_node = node
                
                if position == agent.target:  # 4. If self target
                    x[node, 24] = 1
                    index_of_target = node
                
                if self.location_has_agent.get(position, 0) > 0 and self.location_has_agent_id.get(position) != handle:  # 5. If other agent on this node
                    x[node, 25] = 1
                    
                    other_direction = self.location_has_agent_direction.get(position, 0)  # 5.5 Other direction (one-hot encoding)
                    x[node, 26 + other_direction] = 1
                
                if position in self.location_has_target and position != agent.target:  # 6. If other target
                    x[node, 30] = 1
                
                malfunction_time = self.location_has_agent_malfunction.get(position, 0)  # 7. If other malfunction
                x[node, 31] = malfunction_time
                
                x[node, 32] = torch.min(x[node, 32], self.location_has_agent_speed.get(position, 1.0))  # 8. Other agent speed (min fractional speed)
              
                
                
                x[node, 33] = self.location_has_agent_ready_to_depart.get(position, 0)  # 9. Number of other agents ready to depart but not yet active
                
                if self.predictor:  # 10 & 11. Predictor information
                    for t in range(self.max_prediction_depth):
                        int_position = coordinate_to_position(self.env.width, [position])
                        if int_position in fast_delete(self.predicted_pos[t], handle):
                            x[node, 34] = 1  # Other agent will be on this node in the future
                            if x[node, 35] == 0:
                                x[node, 35] = torch.min(x[node, 35], t)  # Timestep when other agent will be on this node
                            else:
                                x[node, 35] = t  # Timestep when other agent will be on this node
                            break

        data = Data(x=x, edge_index=self.edge_index)
        return data, index_of_node, index_of_target



   