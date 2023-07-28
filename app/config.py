import os

from app.db.db_config import init_redis, init_mongo
import toml

from app.baselog.log_config import LogConfig
from base_log import llog


class Config:
    envParam = None
    _config = None
    _initialized = False
    configFile = None  # 配置文件
    aliLog = None  # 阿里云日志
    customerRedis = None  # 客服机器人redis
    customerMongo = None  # 客服机器人mongo
    backgroundMongo = None  # background mongo

    def __new__(cls, *args, **kwargs):
        if cls._config is None:
            cls._config = super().__new__(cls)
        return cls._config

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

    @classmethod
    def get_config(cls):
        if cls.configFile is None:
            preUrl = ""
            if cls.envParam != "local":
                preUrl = "./"
            configFileUrl = preUrl + "conf/config_{envParam}.toml".format(envParam=cls.envParam)
            llog.info("加载配置文件: {}".format(configFileUrl))
            try:
                cls.configFile = toml.load(configFileUrl)
            except Exception as e:
                raise Exception(f"Failed to load configuration from file {configFileUrl}: {e}")
        return cls.configFile

    @classmethod
    def get_config_info(cls, section, key):
        kMap = cls.get_config().get(section)
        if kMap is None:
            raise Exception(f"Failed to load configuration from file {cls.configFile}: {section} is not exist")
        val = kMap.get(key)
        if val is None:
            raise Exception(f"Failed to load configuration from file {cls.configFile}: {section}.{key} is not exist")
        return val

    # 阿里云日志初始化
    @classmethod
    def init_ailog(cls):
        if cls.aliLog is None:
            # 日志初始化
            projectName = cls.get_config_info("projectName", "name")
            localUrl = cls.get_config_info("log", "localUrl")
            aliUrl = cls.get_config_info("log", "aliUrl")
            project = cls.get_config_info("projectName", "project")
            if cls.envParam == "local":
                localUrl = os.getcwd() + "/logfiles"
                aliUrl = localUrl

            cls.aliLog = LogConfig(
                projectName=projectName,
                localUrl=localUrl,
                aliUrl=aliUrl,
                envParam=Config.envParam,
                project=project
            )

        return cls.aliLog

    # redis初始化
    @classmethod
    def init_redis(cls):
        # 客服机器人redis
        if not cls.customerRedis:
            host = cls.get_config_info("customerRedis", "host")
            port = cls.get_config_info("customerRedis", "port")
            password = cls.get_config_info("customerRedis", "password")
            db = cls.get_config_info("customerRedis", "db")
            cls.customerRedis = init_redis(host, port, password, db)
        return cls.customerRedis

    # mongo初始化
    @classmethod
    def init_mongo(cls):
        # 客服机器人mongo
        if not cls.customerMongo:
            url = cls.get_config_info("customerMongo", "url")
            cls.customerMongo = init_mongo(url, 'customer-robot')

        # background mongo
        if not cls.backgroundMongo:
            url = cls.get_config_info("customerMongo", "url")
            cls.backgroundMongo = init_mongo(url, 'background')

    @classmethod
    def get_customer_mongo(cls):
        return cls.customerMongo

    @classmethod
    def get_customer_redis(cls):
        return cls.customerRedis

    @classmethod
    def get_background_mongo(cls):
        return cls.backgroundMongo

    @classmethod
    def Prepare(cls, envParam=str):
        cls.envParam = envParam
        # 加载配置文件
        cls.get_config()
        # 阿里云日志初始化
        cls.init_ailog()
        # redis初始化
        cls.init_redis()
        # mongo初始化
        cls.init_mongo()
        llog.info("config init success...")

    @classmethod
    def close_db(cls):
        llog.info("customerRedis disconnect start...")
        if cls.customerRedis.is_connected():
            cls.customerRedis.close()
            cls._customerRedis = None
        llog.info("customerRedis disconnect success...")

        llog.info("customerMongo disconnect start...")
        if cls.customerMongo is not None and cls.customerMongo.is_mongos:
            cls.customerMongo.close()
            cls._customerMongo = None
        llog.info("customerMongo disconnect success...")

        llog.info("backgroundMongo disconnect start...")
        if cls.backgroundMongo is not None and cls.backgroundMongo.is_mongos:
            cls.backgroundMongo.close()
            cls.backgroundMongo = None
        llog.info("backgroundMongo disconnect success...")
