import json
import sys
from pathlib import Path
from evals.core.eval_output import CoreEvalOutput
from evals.core.main import convert_feature_metrics

# This script is used to convert an old-format eval output to the new format.
# The old format is no longer produced, so you don't need to use this script.

# load input directory from command line
input_dir = Path(sys.argv[1])

# Get all JSON files in directory, sorted alphabetically
input_files = sorted(input_dir.glob("*.json"))

if not input_files:
    print(f"No JSON files found in {input_dir}")
    sys.exit(1)

# Create outputs directory if it doesn't exist
output_dir = input_dir / "converted_outputs"
output_dir.mkdir(exist_ok=True)

# Convert each file
for input_file in input_files:
    print(f"Converting {input_file}")
    output_file = output_dir / input_file.name
    with open(input_file, "r") as f:
        data = json.load(f)
    feature_metrics = convert_feature_metrics(data["eval_result_details"][0])
    data["eval_result_details"] = feature_metrics
    with open(output_file, "w") as f:
        eval_output = CoreEvalOutput(
            eval_config=data["eval_config"],
            eval_id=data["eval_id"],
            datetime_epoch_millis=data["datetime_epoch_millis"],
            eval_result_metrics=data["eval_result_metrics"],
            eval_result_details=data["eval_result_details"],
            eval_result_unstructured=data.get("eval_result_unstructured", {}),
            sae_bench_commit_hash=data["sae_bench_commit_hash"],
            sae_lens_id=data["sae_lens_id"],
            sae_lens_release_id=data["sae_lens_release_id"],
            sae_lens_version=data["sae_lens_version"],
        )
        eval_output.to_json_file(str(output_file))
