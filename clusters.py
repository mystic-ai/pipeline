#import npu2
import npu2
from npu2.api.compute_cluster import create_cluster
npu2.api.API_ENDPOINT = "http://localhost:5002/v2"
npu2.link("0197246120897461209476120983613409861230986")
create_cluster("main_cluster", "neurocloud")