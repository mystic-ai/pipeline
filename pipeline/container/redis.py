import os

import redis

redis_client = redis.Redis(
    os.environ.get("REDIS_HOST", default="pcore-redis"),
    port=6379,
    db=0,
    decode_responses=True,
)
