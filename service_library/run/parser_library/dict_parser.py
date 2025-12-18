import datetime
import inspect
import json
import logging
import re
import uuid
from typing import Any, Optional

from configs.settings import get_settings
from nvbot_utilities.utils.utilities import get_class_module, create_func_instance
from service_library.utils.logging import log_errors

logger = logging.getLogger("Dict parser'")


@log_errors('Dict parser')
async def dict_parser(parser_config: Optional[list]):
    if parser_config is None:
        return None

    parser_config = parser_config

    async def func(row: dict):

        def can_convert_to_int(s):
            return s.isdigit() or (s.startswith(('-', '+')) and s[1:].isdigit())

        resp = {}
        for config in parser_config:
            name = config.get("name", "")
            parser_type = config.get("type", "")
            value = config.get("value")
            try:
                if parser_type.lower() == "text":
                    # resp = row
                    resp[value] = row[name]
                elif parser_type.lower() == 'attribute':
                    obj = row
                    attributes = value.split(".")
                    for attr in attributes:
                        # check None
                        if isinstance(obj, list):
                            can_convert_to_int = lambda s: s.isdigit() or (s.startswith('-') and s[1:].isdigit())
                            # relaxed index check: len(obj) >= abs(int(attr)) here is incase `attr` is a negative index
                            if can_convert_to_int(attr) and len(obj) >= abs(int(attr)):
                                obj = obj[int(attr)]
                        elif isinstance(obj, dict) and not obj.get(attr) is None:
                            obj = obj.get(attr)
                        elif isinstance(obj, dict) and can_convert_to_int(str(attr)):
                            obj = list(obj.values())[int(attr)]
                        else:
                            continue
                    if name:
                        resp[name] = obj
                    else:
                        resp = obj

                elif parser_type.lower() == 'function':
                    kwargs = None
                    parameters = config.get("args", [])
                    if parameters:
                        kwargs = [resp if key is None else resp.get(key, key) for key in parameters]

                    module_name, function_name = get_class_module(value)
                    func = create_func_instance(module_name=module_name, class_name=function_name)

                    if inspect.iscoroutinefunction(func):
                        # print("is async")
                        if name:
                            if not parameters:
                                func_result = await func(resp)
                                resp[name] = func_result
                            else:
                                func_result = await func(*kwargs)
                                resp[name] = func_result
                        else:
                            if not parameters:
                                resp.update(await func(resp))
                            else:
                                resp.update(await func(*kwargs))
                    else:
                        if name:
                            if not parameters:
                                func_result = func(resp)
                                resp[name] = func_result
                            else:
                                func_result = func(*kwargs)
                                resp[name] = func_result
                        else:
                            if not parameters:
                                resp.update(func(resp))
                            else:
                                resp.update(func(*kwargs))
            except Exception as exp:
                print(f"An error occurred in the parser module {exp} for name {name}, type {parser_type}")
                logger.info(f"An error occurred in the parser module {exp} for name {name}, type {parser_type}")

        return resp

    return func
