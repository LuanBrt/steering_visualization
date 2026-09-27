RECOGNITION_COLUMNS = [{"Name": "output", "Type": "string"}]
EVALUATION_COLUMNS = [{"Name": "evaluation", "Type": "string"}]
PARTITION_KEY_NAMES = ["layer", "category", "label", "variant", "prompt_type"]
INPUT_FORMAT = "org.apache.hadoop.mapred.TextInputFormat"
OUTPUT_FORMAT = "org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat"
SERIALIZATION_LIBRARY = "org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe"
