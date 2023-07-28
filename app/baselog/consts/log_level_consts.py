from enum import Enum


# 日志级别
class LogLevel(Enum):
    DefaultLevel = 0
    PanicLevel = 1
    FatalLeve = 2
    ErrorLevel = 3
    WarnLeve = 4
    NoticeLevel = 5
    InfoLevel = 6
    DebugLevel = 7

    levelMap = {
        DefaultLevel: "DefaultLevel",
        PanicLevel: "panic",
        FatalLeve: "fatal",
        ErrorLevel: "error",
        WarnLeve: "warn",
        NoticeLevel: "notice",
        InfoLevel: "info",
        DebugLevel: "debug",
    }

    @staticmethod
    def LevelStr(level):
        if level is None:
            return "unknown"
        elif level == LogLevel.DefaultLevel:
            return "DefaultLevel"
        elif level == LogLevel.PanicLevel:
            return "panic"
        elif level == LogLevel.FatalLeve:
            return "fatal"
        elif level == LogLevel.ErrorLevel:
            return "error"
        elif level == LogLevel.WarnLeve:
            return "warn"
        elif level == LogLevel.NoticeLevel:
            return "notice"
        elif level == LogLevel.InfoLevel:
            return "info"
        elif level == LogLevel.DebugLevel:
            return "debug"
        else:
            return "unknown"
