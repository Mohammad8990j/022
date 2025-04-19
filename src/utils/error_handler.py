import time
from functools import wraps
from typing import Callable, Any, Optional
from .logger import TradingBotLogger

logger = TradingBotLogger("ErrorHandler")

class RetryError(Exception):
    """خطای مخصوص برای زمانی که تلاش‌های مجدد به نتیجه نرسد"""
    pass

def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Callable:
    """
    دکوراتور برای تلاش مجدد توابع در صورت خطا
    
    Args:
        max_attempts: حداکثر تعداد تلاش
        delay: تاخیر اولیه بین تلاش‌ها (ثانیه)
        backoff: ضریب افزایش تاخیر
        exceptions: تاپل خطاهایی که باید مدیریت شوند
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception: Optional[Exception] = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"تلاش {attempt + 1}/{max_attempts} برای {func.__name__} "
                            f"با خطا مواجه شد: {str(e)}. تلاش مجدد در {current_delay} ثانیه..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"تمام تلاش‌ها برای {func.__name__} با شکست مواجه شد: {str(e)}"
                        )
            
            raise RetryError(f"تابع {func.__name__} پس از {max_attempts} تلاش با شکست مواجه شد") from last_exception
        
        return wrapper
    return decorator

def safe_execute(func: Callable) -> Callable:
    """
    دکوراتور برای اجرای ایمن توابع و ثبت خطاها
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"خطا در اجرای {func.__name__}: {str(e)}")
            raise
    
    return wrapper