import  logging
from    typing import Type, Dict, Any, Callable
from    pydantic import BaseModel

logger = logging.getLogger(__name__)

trsfrm_flatten = lambda x: [item for sublist in x for item in sublist]


def create_tool_definition(fn: Callable, cls: Type[BaseModel], **kwargs: Any) -> Dict[str, Any]:
    fn_name = fn.__name__ if hasattr(fn, "__name__") else fn.__class__.__name__
    return dict(type="function", function=dict(name=fn_name, description=fn.__doc__, parameters=cls.model_json_schema()))


def validate_input(cls: Type[BaseModel], data_json: str):
    """Validate input from a JSON string and return a BaseModel instance if valid.


    Example:
    >>> user_input_json = '''
        {
            "name": "Joe User",
            "email": "joe@example.com",
            "query": "When can I expect delivery of the headphones I ordered?",
            "order_id": "ABC-12345",
            "purchase_date": "2025-12-01"
        }
        '''
        valid_data     = validate_input(user_input_json).model_dump_json()
        customer_query = create_customer_query(valid_data)
        print(type(customer_query))
        print(customer_query.model_dump_json(indent=2))
    """
    try:
        data_validation = cls.model_validate_json(data_json)
        logger.info(f"{cls.__module__}.{cls.__name__} validated...")
        return data_validation
    except Exception as e:
        logger.error(f" Unexpected error: {e}")
        return None
