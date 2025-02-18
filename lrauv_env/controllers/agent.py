import rclpy
from lrauv_msgs.msg import LRAUVRangeBearingRequest, LRAUVRangeBearingResponse
from ros_gz_interfaces.msg import Dataframe
from std_msgs.msg import Header
from .entity import LrauvEntityController
from .action import LinearController
from typing import List, Union
import time
import json
import re
from utils.conversions import bearing_to_r_el_az
from tracking import Tracker


class LrauvAgentController(LrauvEntityController):

    def __init__(
        self,
        name:str='agent_1',
        comm_adress:int=1,
        action_controller:Union[LinearController,None]=None,
        entities_names:List[str]=['agent_1','landmark_1'],
        landmarks_depth:Union[List[float],None]=None, # the depth of the landmarks is assumed to be known
        **tracker_args
    ):

        super().__init__(name, comm_adress, action_controller)

        # agents in respect of normal entities (landmarks) can comunicate (send and recieve range bearing requests/responses)
        self.name = name
        self.comm_adress = comm_adress
        self.agents_names    = [n for n in entities_names if 'agent' in n]
        self.other_agents_names = [n for n in self.agents_names if n!=self.name]
        self.landmarks_names = [n for n in entities_names if 'landmark' in n]
        self.landmarks_depth = landmarks_depth if landmarks_depth is not None else [None for _ in self.landmarks_names]
        self.others  = {name:i+1 for i, name in enumerate(entities_names) if name!=self.name} # agent_1:1, landmark_1:3 etc.
        self.others_adr  = {i+1:name for i, name in enumerate(entities_names) if name!=self.name}

        # range bearing request publisher: publishes requests of bearing
        self.range_pub = self.create_publisher(LRAUVRangeBearingRequest,f"/{name}/range_bearing/requests",10)
        # range bearing resonsse publisher: listens to requests of bearing
        self.range_sub = self.create_subscription(
            LRAUVRangeBearingResponse,
            f"/{name}/range_bearing/responses",
            self._range_callback,
            10
        )
        self.requests_ids = {}
        self.requests_count = 0

        # communication between agents: acoustic messages
        # it should use the LRAUVAcousticMessage messages, but it's not correctly implemented in gazebo
        # therefore use protobuf directly with Dataframes (as in https://github.com/osrf/lrauv/blob/main/lrauv_gazebo_plugins/include/lrauv_gazebo_plugins/comms/CommsClient.hh)
        self.comm_pub = self.create_publisher(Dataframe, '/broker/msgs', 10)
        self.comm_sub = self.create_subscription(
            Dataframe,
            f"/{self.name}/rx",
            self._comm_callback,
            10
        )
        self._reset_comms()

        # tracking: each agent uses a separate tracker for every landmark
        self.trackers = [Tracker(**tracker_args) for n in entities_names if 'landmark' in n]

    def get_state(self):
        if self.current_state is None:
            self.current_state = super().get_state()
        return self.current_state

    def get_obs(self, dt:int=60):
        state = self.get_state()
        comms = self.collect_comms()
        tracks = self.update_tracking(dt=dt)
        self._reset_comms()
        return {**state, **comms, **tracks}

    def send_range_requests(self):
        self.requests_ids = {} # re-init the requests ids
        # publish range request for all the other entities
        for name, adr in self.others.items():
            if 'landmark' in name: # send range request only to the landmarks
                req = LRAUVRangeBearingRequest()
                req.to = adr
                req.req_id = self.requests_count
                self.requests_ids[self.requests_count] = name # take trace of the adress of the request reciver with the req_id
                self.range_pub.publish(req)
                rclpy.spin_once(self, timeout_sec=0.0001)
                self.requests_count += 1

    def broadcast_to_agents(self, message:List):
        for name, adr in self.others.items():
            if 'agent' in name:
                header = Header()
                header.frame_id = "LRAUVAcousticMessage" # specify is an acoustic message
                msg = Dataframe()
                msg.header = header
                msg.dst_address = str(adr)
                msg.src_address = str(self.comm_adress)
                msg.data = message
                self.comm_pub.publish(msg)
                rclpy.spin_once(self, timeout_sec=0.0001)
                
    def collect_comms(self, timeout=0.3):
        # wait until timeout to get the communications from the other agents
        start_time = time.time()
        while any(comm is None for comm in self.comms.values()):
            rclpy.spin_once(self, timeout_sec=0.0001)
            if time.time() - start_time >= timeout:
                self.get_logger().warning(f"{self.name} didn't recieve all the communications: {self.comms}")
                break

        return self._preprocess_comms()

    def collect_range_responses(self, timeout=0.3):
        # wait until timeout to get the range responses from the landmarks
        start_time = time.time()
        while any(response is None for response in self.range_responses.values()):
            rclpy.spin_once(self, timeout_sec=0.0001)
            if time.time() - start_time >= timeout:
                self.get_logger().warning(f"{self.name} didn't recieve all the range responses: {self.range_responses}")
                self._fullfill_range_responses()
                break

        return {f'{k}_range':r for k, r in self.range_responses.items()}


    def communicate(self, range_bearing_timout=0.3):

        # collect the range responses
        range_responses = self.collect_range_responses(range_bearing_timout)

        # get the current state of the agent
        self.current_state = self.get_state()
        self.current_state.update(range_responses) # add the range bearings

        # prepare the communication and broadcast it to all the agents
        comm = {k:round(v,2) for k,v in self.current_state.items() if 'vel' not in k} # avoid to send info about agent velocity
        message = list(json.dumps(comm).encode('utf8'))
        self.broadcast_to_agents(message)
        return message


    def _comm_callback(self, msg):
        if msg.header.frame_id == "LRAUVAcousticMessage":
            sender = self.others_adr[int(msg.src_address)]
            # data could be corrupted, therefore try
            try:
                data = bytes(list(msg.data)).decode('utf8')
                self.comms[sender] = json.loads(data)
            except:
                self.get_logger().warning(f'{self.name} got corrupted communication from {sender}, message: {msg}')
                self.comms[sender] = {}

    
    def _range_callback(self, msg):
        
        # get the responder id from the responder
        if msg.req_id in self.requests_ids.keys():
            responder = self.requests_ids[msg.req_id]
            # get the range from the bearing
            r, _, _ = bearing_to_r_el_az(msg.bearing)
            self.range_responses[responder] = r

    def _fullfill_range_responses(self):
        # fullfill responses with 0s if not recived
        for responder, response in self.range_responses.items():
            if response is None:
                self.range_responses[responder] = 0.

    def _preprocess_comms(self):

        # fullfill comms with 0s if not recived
        state = self.get_state()
        x_agent, y_agent, z_agent = state['x'], state['y'], state['z']
        
        comms = {}
        for sender, c in self.comms.items():
            # if the communication did't arrive (None) or is corrupted ({}), fullfill with 0s
            if c == {} or c is None:
                comms.update({f'{sender}_dx': 0, f'{sender}_dy': 0, f'{sender}_dz': 0})
                comms.update({f'{sender}_{name}_range':0 for name in self.landmarks_names})
            else:
                comms.update({
                    f'{sender}_dx': x_agent - c['x'],
                    f'{sender}_dy': y_agent - c['y'],
                    f'{sender}_dz': z_agent - c['z']
                })
                comms.update({f'{sender}_{k}':v for k, v in c.items() if 'range' in k}) # add the ranges
            
        return comms

    def _reset_comms(self):
        # reset all the communication variables that are relative to the current time step
        self.range_responses = {name:None for name in self.landmarks_names}
        self.comms = {name:None for name in self.other_agents_names}
        self.current_state = None


    def update_tracking(self, dt:int=60):
        state = self.get_state()
        x_agent, y_agent, z_agent = state['x'], state['y'], state['z']

        # prepare positions and ranges
        positions = [[x_agent, y_agent, z_agent]]
        ranges    = [[state[f'{landmark}_range']] for landmark in self.landmarks_names]

        # use the positions and ranges received by communications
        for c in self.comms.values():
            # if the communication did't arrive (None) or is corrupted ({}), ignore
            if c == {} or c is None:
                continue
            else:
                positions.append([c['x'], c['y'], c['z']])
                for i, landmark in enumerate(self.landmarks_names):
                    ranges[i].append(c[f'{landmark}_range'])

        # update the tracking of each landmark separately
        preds = {}
        for i, landmark in enumerate(self.landmarks_names):
            pred = self.trackers[i].update_and_predict(
                ranges=ranges[i],
                positions=positions,
                depth=self.landmarks_depth[i],
                dt=dt
            )
            preds[f'{landmark}_tracking_x'] = pred[0]
            preds[f'{landmark}_tracking_y'] = pred[1]
            preds[f'{landmark}_tracking_z'] = pred[2]

        return preds





    


