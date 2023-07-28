# LogConfig 日志配置
import json
import logging
from base_log import llog
import os
import time
from logging.handlers import RotatingFileHandler
import tools.json_utils as util
from app.baselog.consts.log_level_consts import LogLevel


class LogConfig:
    _instance = None
    _initialized = False
    _aliLogger = None

    _projectName = ""  # 项目名
    _localUrl = ""  # 项目日志Url
    _aliUrl = ""  # 阿里云日志Url
    _envParam = ""  # 环境
    _project = ""  # 环境

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, projectName=str, localUrl=str, aliUrl=str, envParam=str, project=str):
        if self._initialized:
            return
        LogConfig._projectName = projectName
        LogConfig._localUrl = localUrl
        LogConfig._aliUrl = aliUrl
        LogConfig._envParam = envParam
        LogConfig._project = project
        # 日志初始化
        self.init_aliLog()
        self._initialized = True

    @classmethod
    def get_aliLog(cls):
        if cls._aliLogger is None:
            cls.init_aliLog()
        return cls._aliLogger

    # 阿里云日志初始化
    @classmethod
    def init_aliLog(cls):
        if cls._aliLogger is None:
            os.makedirs(cls._aliUrl, exist_ok=True)

            # 配置信息日志
            access_handler = RotatingFileHandler(
                filename=os.path.join(cls._aliUrl, f"{cls._projectName}.log"),
                encoding="utf-8",  # 设置编码为 UTF-8
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=50  # 最多保留50个日志文件
            )

            access_handler.setFormatter(logging.Formatter("%(message)s"))

            cls._aliLogger = logging.getLogger(f"alilog_{__name__}")
            cls._aliLogger.setLevel(logging.INFO)
            cls._aliLogger.addHandler(access_handler)
        return cls._aliLogger

    @classmethod
    def info(cls, message, lgMap):
        cls.printF(LogLevel.InfoLevel, message, lgMap)

    @classmethod
    def error(cls, message, lgMap):
        cls.printF(LogLevel.DebugLevel, message, lgMap)

    @classmethod
    def printF(cls, level, message, lgMap):
        if lgMap is None:
            llog.error(f"不是json对象，is None，detail{str(lgMap)}")
            return

        if not isinstance(lgMap, dict):
            llog.error(f"不是json对象，detail{str(lgMap)}")
            return

        bModel = AliLogBaseModel(
            tpInfo=LogLevel.LevelStr(level),
            alarmLevel=str(level.value),
            message=message,
            logstore=f"{cls._projectName}-server",
            project=cls._project,
            ts=int(time.time()),
            data=lgMap
        )

        try:
            s = json.dumps(bModel, ensure_ascii=False, cls=util.CustomEncoder)
            cls._aliLogger.info(s)
        except Exception as e:
            llog.error(f"阿里云日志打印失败,detail:{str(e)}")


class AliLogBaseModel:
    tpInfo: str
    alarmLevel: str
    message: str
    logstore: str
    project: str
    ts: int
    data: dict

    def __init__(self, tpInfo, alarmLevel, message, logstore, project, ts, data):
        self.tpInfo = tpInfo
        self.alarmLevel = alarmLevel
        self.message = message
        self.logstore = logstore
        self.project = project
        self.ts = ts
        self.data = data

    def to_dict(self):
        return {
            '_tp': self.tpInfo,
            '_alarmLevel': self.alarmLevel,
            '_message': self.message,
            'logstore': self.logstore,
            'project': self.project,
            '_ts': self.ts,
            'data': self.data
        }
