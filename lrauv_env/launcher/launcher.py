import os
import subprocess
import psutil
import time

class RosLrauvLauncher:
    def __init__(self, n_agents=2, n_landmarks=2, gui=True):
        self.n_agents = n_agents
        self.n_landmarks = n_landmarks
        self.gui = gui
        self.process = None

    def launch(self):
        # Assumes the launcher python file is in the same folder of current file
        t0 = time.time() # monitor the duration to start the sim
        current_dir = os.path.dirname(os.path.abspath(__file__))
        launch_file_path = os.path.join(current_dir, 'ros_lrauv.launch.py')

        launch_cmd = f'ros2 launch {launch_file_path} n_agents:={self.n_agents} n_landmarks:={self.n_landmarks}'
        if not self.gui:
            launch_cmd += ' server_mode:=1'
        
        # Run the launch command and store a reference to the launched process
        self.process = subprocess.Popen(
            launch_cmd,
            shell=True
        )


    def terminate(self):
        if self.process is not None:
            # Find all child processes
            parent = psutil.Process(self.process.pid)
            children = parent.children(recursive=True)

            # Terminate all child processes
            for child in children:
                child.terminate()

            # Wait for child processes to terminate
            _, still_alive = psutil.wait_procs(children, timeout=3)

            # Kill any remaining child processes
            for child in still_alive:
                child.kill()

            # Terminate the parent process
            self.process.terminate()