import redis
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import ServerSelectionTimeoutError

from base_log import llog


# redis 初始化
def init_redis(host: str,
               port: int,
               password: str,
               db: int = 0,
               charset: str = "utf-8"):
    llog.info(f"Redis host: {host}, port: {port}, db: {db} connect start...")
    redis_conn = redis.Redis(
        host=host,
        port=port,
        db=db,
        password=password,
        charset=charset,
        decode_responses=True
    )

    # 检查连接是否成功
    try:
        result = redis_conn.ping()
    except Exception as e:
        llog.error(f"Redis host: {host}, port: {port}, db: {db} connect exception...")
        raise e
    if result:
        llog.info(f"Redis host: {host}, port: {port}, db: {db} connect success...")
    else:
        llog.error(f"Redis host: {host}, port: {port}, db: {db} connect fail...")
        raise Exception("连接 Redis 服务器失败")
    return redis_conn


# mongo 初始化
def init_mongo(url: str, db_name: str) -> Database:
    llog.info(f"mongo url: {url}, db_name: {db_name} connect start...")
    try:
        client = MongoClient(url, serverSelectionTimeoutMS=2000)
        # 尝试访问 MongoDB 以确保连接成功
        client.server_info()
        llog.info(f"mongo url: {url}, db_name: {db_name} connect success...")
        return client[db_name]
    except ServerSelectionTimeoutError as e:
        llog.error(f"mongo url: {url}, db_name: {db_name} connect fail...")
        raise ConnectionError("Failed to connect to MongoDB")
    except Exception as e:
        llog.error(f"mongo url: {url}, db_name: {db_name} connect exception...")
        raise e
