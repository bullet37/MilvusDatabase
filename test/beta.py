import json
from milvus import FieldSchema, DataType


def json_to_schema(json_url):
    with open(json_url, 'r') as f:
        config = json.load(f)

    fields = []
    for field in config['fields']:
        if field['dtype'].startswith("DataType."):
            dtype = eval(field['dtype'])
        else:
            dtype = field['dtype']
        field_schema = FieldSchema(
            name=field['name'],
            dtype=dtype,
            is_primary=field.get('is_primary', False),
            auto_id=field.get('auto_id', False),
            max_length=field.get('max_length', None),
            dim=field.get('dim', None)
        )
        fields.append(field_schema)
    return fields
