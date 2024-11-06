import os
import json
from typing import Type

from evals.base_eval_output import BaseEvalOutput
from pydantic import TypeAdapter


def generate_json_schema(eval_output: Type[BaseEvalOutput], output_file: str):
    schema = TypeAdapter(eval_output).json_schema()
    with open(output_file, "w") as f:
        json.dump(schema, f, indent=2)


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    evals_dir = os.path.join(base_dir, "evals")

    for root, dirs, files in os.walk(evals_dir):
        for file in files:
            if file == "eval_output.py":
                print(file)
                module_path = os.path.relpath(os.path.join(root, file), base_dir)
                module_name = module_path.replace("/", ".").replace(".py", "")

                try:
                    module = __import__(module_name, fromlist=[""])
                    for name, obj in module.__dict__.items():
                        if (
                            isinstance(obj, type)
                            and issubclass(obj, BaseEvalOutput)
                            and obj != BaseEvalOutput
                        ):
                            output_file = os.path.join(
                                root, f"eval_output_schema_{obj.eval_type_id}.json"
                            )
                            generate_json_schema(obj, output_file)
                            print(f"Generated schema for {name} in {output_file}")
                except ImportError as e:
                    print(f"Could not import {module_name}: {e}")


if __name__ == "__main__":
    main()
