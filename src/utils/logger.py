import logging
from logging.handlers import TimedRotatingFileHandler
import os
from dotenv import load_dotenv
import colorlog
from datetime import datetime

# 加载环境变量以防主入口未加载
load_dotenv()

def setup_logger(name: str) -> logging.Logger:
    """
    统一的日志获取与配置工厂。
    使用示例: logger = setup_logger(__name__)
    """
    logger = logging.getLogger(name)
    
    # 如果已经配置过，防止重复添加 Handler
    if logger.handlers:
        return logger

    # 从环境变量读取日志配置
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger.setLevel(log_level)
    
    # 文件日志格式 (纯文本)
    file_formatter = logging.Formatter(
        '%(asctime)s - [%(levelname)s] - %(name)s - %(message)s'
    )
    
    # 控制台彩色日志格式
    console_formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'bold_red',
        }
    )
    
    # 1. 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 2. 文件持久化与轮转
    log_file_path = os.getenv("LOG_FILE_PATH", "logs/bot.log")
    
    # 将后缀明确为带日期的格式
    base_dir = os.path.dirname(log_file_path)
    if base_dir and not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
        
    retention_days = int(os.getenv("LOG_RETENTION_DAYS", "30"))
    
    try:
        # 3. 文件输出 (按天生成独立日志文件，例如 2026-08-18.log)
        today_str = datetime.now().strftime("%Y-%m-%d")
        log_dir = os.path.dirname(log_file_path)
        os.makedirs(log_dir, exist_ok=True)
        daily_log_file = os.path.join(log_dir, f"{today_str}.log")
        
        file_handler = TimedRotatingFileHandler(
            filename=daily_log_file,
            when="midnight",          
            interval=1,               
            backupCount=retention_days, 
            encoding="utf-8"
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"初始化日志文件失败: {e}")

    # 防止日志向上传播导致重复输出
    logger.propagate = False
    
    return logger
