
# STD LIB
import os
import pdb
import pathlib
import subprocess

# EXTERNAL LIB
import cv2
import torch
import numpy as np

# LOCAL LIB
try:
    from spynet import spynet
except ImportError:
    from .spynet import spynet

# CONSTANTS
# The the paths to run various commands.
DEEPMATCHING = './deepflow2/deepmatching-static'
DEEPFLOW2 = './deepflow2/deepflow2-static'
CONSISTENCY_CHECK = './consistencyChecker/consistencyChecker'

def write_flow(fname, flow):
    '''
    Write optical flow to a .flo file
    Args:
        fname: Path where to write optical flow
        flow: an ndarray containing optical flow data
    '''
    # Save optical flow to disk
    with open(fname, 'wb') as f:
        np.array(202021.25, dtype=np.float32).tofile(f) # Write magic number for .flo files
        height, width = flow.shape[:2]
        np.array(width, dtype=np.uint32).tofile(f)      # Write width
        np.array(height, dtype=np.uint32).tofile(f)     # Write height
        flow.astype(np.float32).tofile(f)               # Write data

def farneback_flow(start_name, end_name):
    start = cv2.cvtColor(cv2.imread(start_name), cv2.COLOR_BGR2GRAY)
    end = cv2.cvtColor(cv2.imread(end_name), cv2.COLOR_BGR2GRAY)
    
    forward = cv2.calcOpticalFlowFarneback(start, end, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    backward = cv2.calcOpticalFlowFarneback(end, start, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    return forward, backward

def spynet_flow(start_name, end_name):
    start = torch.Tensor(cv2.imread(start_name).transpose(2, 0, 1) * (1.0 / 255.0))
    end = torch.Tensor(cv2.imread(end_name).transpose(2, 0, 1) * (1.0 / 255.0))

    forward = spynet.estimate(start, end).detach().numpy().transpose(1, 2, 0)
    backward = spynet.estimate(end, start).detach().numpy().transpose(1, 2, 0)
    
    return forward, backward

def deep_flow(start_name, end_name, forward_name, backward_name):
    # Compute forward optical flow.
    root = pathlib.Path(__file__).parent.absolute()
    forward_dm = subprocess.Popen([
        str('.' / root / DEEPMATCHING), start_name, end_name, '-nt', '0', '-downscale', '2'
    ], stdout=subprocess.PIPE)
    subprocess.run([
        str('.' / root / DEEPFLOW2), start_name, end_name, forward_name, '-match'
    ], stdin=forward_dm.stdout)
    
    # Compute backward optical flow.
    backward_dm = subprocess.Popen([
        str('.' / root / DEEPMATCHING), end_name, start_name, '-nt', '0', '-downscale', '2'
    ], stdout=subprocess.PIPE)
    subprocess.run([
        str('.' / root / DEEPFLOW2), end_name, start_name, backward_name, '-match'
    ], stdin=backward_dm.stdout)

def estimate(start_name, end_name, forward_name, backward_name, method):
    if method == 'farneback':
        forward, backward = farneback_flow(start_name, end_name)
        # Write flows to disk so that they can be used in the consistency check.
        write_flow(forward_name, forward)
        write_flow(backward_name, backward)
    elif method == 'spynet':
        forward, backward = spynet_flow(start_name, end_name)
        # Write flows to disk so that they can be used in the consistency check.
        write_flow(forward_name, forward)
        write_flow(backward_name, backward)
    elif method == 'deepflow2': # TODO: options
        deep_flow(start_name, end_name, forward_name, backward_name)
    else:
        raise Exception('Bad flow method: {}'.format(method))
    
    # The absolute path accounts for if this file is being run as part of a submodule.
    root = pathlib.Path(__file__).parent.absolute()
    # Compute consistency check for backwards optical flow.
    subprocess.run([
        str('.' / root / CONSISTENCY_CHECK),
        backward_name, forward_name, reliable_name, end_name
    ])
    
    # Remove forward optical flow to save space, as it is only needed for the consistency check.
    os.remove(forward_name)