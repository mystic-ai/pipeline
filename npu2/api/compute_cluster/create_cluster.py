from npu2.api.call import post

def create_cluster(cluster_name: str, cluster_cloud: str):
    query_dict = {
        "compute_cluster_name": cluster_name,
        "compute_cluster_cloud": cluster_cloud,
    }
    return post("/compute_cluster", query_dict)
