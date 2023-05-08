import rclpy
from lrauv_msgs.msg import LRAUVRangeBearingRequest, LRAUVRangeBearingResponse
from .entity import LrauvEntityController
from typing import List
import math

class LrauvAgentController(LrauvEntityController):

    def __init__(
        self,
        name:str='agent_1',
        comm_adress:int=1,
        entities_names:List[str]=['agent_1','landmark_1'],
        range_for_landmarks:bool=True,
    ):

        super().__init__(name, comm_adress)

        # agents in respect of normal entities (landmarks) can comunicate (send and recieve range bearing requests/responses)
        self.comm_adress = comm_adress
        self.others_names  = [name for name in entities_names if name!=self.name]
        self.others_comm_adress = [i for i in range(1,len(entities_names)+1) if i!=self.comm_adress] # get the other comm adresses
        # range bearing request publisher: publishes requests of bearing
        self.range_pub = self.create_publisher(LRAUVRangeBearingRequest,f"{name}/range_bearing/requests",10)
        # range bearing resonsse publisher: listens to requests of bearing
        self.range_sub = self.create_subscription(
            LRAUVRangeBearingResponse,
            f"{name}/range_bearing/responses",
            self._range_callback,
            10
        )
        self.range_for_landmarks = range_for_landmarks # if true, returns only the range for landmarks (dx,dy,dz wll be None)
        self.range_responses = {i:None for i in self.others_names}
        self.requests_ids = {}
        self.requests_count = 0

    def send_range_requests(self):
        self.requests_ids = {} # re-init the requests ids
        # publish range request for all the other entities
        for name, adr in zip(self.others_names, self.others_comm_adress):
            req = LRAUVRangeBearingRequest()
            req.to = adr
            req.req_id = self.requests_count
            self.requests_ids[self.requests_count] = name # take trace of the adress of the request reciver with the req_id
            self.range_pub.publish(req)
            rclpy.spin_once(self, timeout_sec=0.001)
            self.requests_count += 1

    def get_obs(self):
        state = super().get_state()
        range_responses = self.collect_range_responses()
        return self._preprocess_obs(state, range_responses)

    def collect_range_responses(self):

        while any(response is None for response in self.range_responses.values()):
            rclpy.spin_once(self, timeout_sec=0.00001)

        range_responses = self.range_responses
        self.range_responses = {i:None for i in self.others_names}
        return range_responses

    def _range_callback(self, msg):
        # get the responder id from the 
        responder = self.requests_ids[msg.req_id]
        if self.range_for_landmarks and 'landmark' in responder:
            self.range_responses[responder] = {'range':msg.range}
        else:
            dx, dy, dz = self._process_bearing(msg.bearing)
            self.range_responses[responder] = {'dx':dx, 'dy': dy, 'dz':dz}

    def _preprocess_obs(self, state, responses):
        # add the responses as single keys in the state dictionary
        for key, subdict in responses.items():
            # use the range if it is given
            if 'range' in subdict.keys():
                state[f'{key}_range'] = subdict['range']
            # otherwise the delta position
            else:
                state[f'{key}_dx'] = subdict['dx'] #+ state['x']
                state[f'{key}_dy'] = subdict['dy'] #+ state['y']
                state[f'{key}_dz'] = subdict['dz'] #+ state['z']      
        return state
    
    def _process_bearing(self, bearing):  
        # get r, elevation and azymuth from the bearing
        x = bearing.x
        y = bearing.y
        z = bearing.z

        r = math.sqrt(x**2 + y**2 + z**2)
        elevation = math.asin(z / r)
        azimuth = math.atan2(y, x)
        
        # Calculate Cartesian coordinates
        dx = r * math.cos(elevation) * math.cos(azimuth)
        dy = r * math.cos(elevation) * math.sin(azimuth)
        dz = r * math.sin(elevation)

        return dx, dy, dz







    


