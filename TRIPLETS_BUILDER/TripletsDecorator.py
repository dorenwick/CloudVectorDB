import time
import functools
import psutil
import os

class TripletsDecorators:
    @staticmethod
    def measure_time(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Execution time of {func.__name__}: {execution_time:.6f} seconds")
            return result
        return wrapper

    @staticmethod
    def print_memory_usage(location):
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        print(f"Memory usage at {location}: {memory_info.rss / 1024 / 1024:.2f} MB")

    @staticmethod
    def create_sentence_work(work_info):
        display_name = work_info.get('title_string', '')
        authors_string = work_info.get('authors_string', '')
        field = work_info.get('field_string', '')
        subfield = work_info.get('subfield_string', '')
        query_string = f"{display_name} {authors_string} {field} {subfield}"
        return query_string

    @staticmethod
    def create_full_string(work):
        return f"{work.get('title_string', '')} {work.get('authors_string', '')} {work.get('field_string', '')} {work.get('subfield_string', '')}"

    @classmethod
    def memory_usage_decorator(cls, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cls.print_memory_usage(f"Before {func.__name__}")
            result = func(*args, **kwargs)
            cls.print_memory_usage(f"After {func.__name__}")
            return result
        return wrapper

    @staticmethod
    def retry(max_attempts=3, delay=1):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                attempts = 0
                while attempts < max_attempts:
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        attempts += 1
                        if attempts == max_attempts:
                            raise
                        print(f"Attempt {attempts} failed. Retrying in {delay} seconds...")
                        time.sleep(delay)
            return wrapper
        return decorator

    @staticmethod
    def validate_input(validator):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not validator(*args, **kwargs):
                    raise ValueError("Invalid input")
                return func(*args, **kwargs)
            return wrapper
        return decorator

    @staticmethod
    def cache_result(func):
        cache = {}
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            if key not in cache:
                cache[key] = func(*args, **kwargs)
            return cache[key]
        return wrapper

    @staticmethod
    def log_calls(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            print(f"Calling {func.__name__}")
            result = func(*args, **kwargs)
            print(f"Finished {func.__name__}")
            return result
        return wrapper

# Example usage:
# from TripletsDecorators import TripletsDecorators as TD
#
# @TD.measure_time
# @TD.memory_usage_decorator
# def some_function():
#     ...
#
# @TD.retry(max_attempts=5, delay=2)
# def network_operation():
#     ...
#
# @TD.validate_input(lambda x: x > 0)
# def positive_input_function(x):
#     ...
#
# @TD.cache_result
# def expensive_computation(x):
#     ...
#
# @TD.log_calls
# def function_to_log():
#     ...
#
# # Using utility methods
# work_info = {...}
# sentence = TD.create_sentence_work(work_info)
# full_string = TD.create_full_string(work_info)