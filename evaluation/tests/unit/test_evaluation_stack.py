import aws_cdk as cdk
from aws_cdk.assertions import Match, Template

from steering_evaluation.evaluation_stack import SteeringVisualizationEvaluationStack


def _synth_template() -> Template:
    app = cdk.App()
    stack = SteeringVisualizationEvaluationStack(app, "TestStack")
    return Template.from_stack(stack)


def test_bucket_is_private_and_named():
    template = _synth_template()
    template.has_resource_properties(
        "AWS::S3::Bucket",
        {
            "BucketName": "steering-visualization",
            "PublicAccessBlockConfiguration": {
                "BlockPublicAcls": True,
                "BlockPublicPolicy": True,
                "IgnorePublicAcls": True,
                "RestrictPublicBuckets": True,
            },
        },
    )


def test_glue_database_created():
    template = _synth_template()
    template.has_resource_properties(
        "AWS::Glue::Database",
        {"DatabaseInput": {"Name": "steering_visualization"}},
    )


def test_recognition_table_has_expected_partition_keys_and_columns():
    template = _synth_template()
    template.has_resource_properties(
        "AWS::Glue::Table",
        {
            "TableInput": {
                "Name": "recognition_results",
                "PartitionKeys": [
                    {"Name": "layer", "Type": "string"},
                    {"Name": "category", "Type": "string"},
                    {"Name": "label", "Type": "string"},
                    {"Name": "variant", "Type": "string"},
                    {"Name": "prompt_type", "Type": "string"},
                ],
                "StorageDescriptor": {
                    "Columns": [{"Name": "output", "Type": "string"}],
                    "InputFormat": "org.apache.hadoop.mapred.TextInputFormat",
                    "OutputFormat": "org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat",
                    "SerdeInfo": {"SerializationLibrary": "org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe"},
                },
            },
        },
    )


def test_evaluation_table_has_expected_column():
    template = _synth_template()
    template.has_resource_properties(
        "AWS::Glue::Table",
        {
            "TableInput": {
                "Name": "evaluation_results",
                "StorageDescriptor": {
                    "Columns": [{"Name": "evaluation", "Type": "string"}],
                },
            },
        },
    )


def test_athena_workgroup_points_at_query_results_prefix():
    template = _synth_template()
    template.has_resource_properties(
        "AWS::Athena::WorkGroup",
        {
            "Name": "steering-visualization",
            "WorkGroupConfiguration": {
                "ResultConfiguration": {
                    "OutputLocation": "s3://steering-visualization/athena-query-results/",
                },
            },
        },
    )


def test_four_lambda_functions_created():
    # 3 real functions (recognition, evaluation, cleanup) + 1 shared
    # BucketNotificationsHandler singleton that add_event_notification creates.
    template = _synth_template()
    template.resource_count_is("AWS::Lambda::Function", 4)


def test_recognition_role_restricted_to_recognition_model_arns():
    # The inference-profile ARN embeds the account token, so CDK synthesizes
    # it as an Fn::Join rather than a literal string -- only the
    # foundation-model ARN (account-independent) is asserted as a literal.
    template = _synth_template()
    template.has_resource_properties(
        "AWS::IAM::Policy",
        {
            "PolicyDocument": {
                "Statement": Match.array_with(
                    [
                        Match.object_like(
                            {
                                "Action": "bedrock:InvokeModel",
                                "Resource": Match.array_with(
                                    ["arn:aws:bedrock:*::foundation-model/anthropic.claude-sonnet-4-5-20250929-v1:0"]
                                ),
                            }
                        )
                    ]
                ),
            },
        },
    )


def test_evaluation_role_restricted_to_evaluation_model_arns():
    template = _synth_template()
    template.has_resource_properties(
        "AWS::IAM::Policy",
        {
            "PolicyDocument": {
                "Statement": Match.array_with(
                    [
                        Match.object_like(
                            {
                                "Action": "bedrock:InvokeModel",
                                "Resource": Match.array_with(
                                    ["arn:aws:bedrock:*::foundation-model/anthropic.claude-haiku-4-5-20251001-v1:0"]
                                ),
                            }
                        )
                    ]
                ),
            },
        },
    )


def test_recognition_notification_filters_on_data_prefix():
    template = _synth_template()
    template.has_resource_properties(
        "Custom::S3BucketNotifications",
        {
            "NotificationConfiguration": {
                "LambdaFunctionConfigurations": Match.array_with(
                    [
                        Match.object_like(
                            {
                                "Events": ["s3:ObjectCreated:*"],
                                "Filter": {"Key": {"FilterRules": Match.array_with([{"Name": "prefix", "Value": "data/"}])}},
                            }
                        )
                    ]
                ),
            },
        },
    )


def test_cleanup_notification_filters_on_object_removed():
    template = _synth_template()
    template.has_resource_properties(
        "Custom::S3BucketNotifications",
        {
            "NotificationConfiguration": {
                "LambdaFunctionConfigurations": Match.array_with(
                    [
                        Match.object_like(
                            {
                                "Events": ["s3:ObjectRemoved:*"],
                                "Filter": {"Key": {"FilterRules": Match.array_with([{"Name": "prefix", "Value": "data/"}])}},
                            }
                        )
                    ]
                ),
            },
        },
    )


def test_collaborator_role_trusts_same_account_only():
    template = _synth_template()
    template.has_resource_properties(
        "AWS::IAM::Role",
        {
            "RoleName": "steering-visualization-collaborator",
            "MaxSessionDuration": 43200,
            "AssumeRolePolicyDocument": {
                "Statement": Match.array_with(
                    [
                        Match.object_like(
                            {
                                "Action": "sts:AssumeRole",
                                "Effect": "Allow",
                                "Principal": {"AWS": {"Fn::Join": Match.any_value()}},
                            }
                        )
                    ]
                ),
            },
        },
    )


def _collaborator_policy_statements(template):
    policies = template.find_resources("AWS::IAM::Policy")
    for name, policy in policies.items():
        if name.startswith("CollaboratorRoleDefaultPolicy"):
            return policy["Properties"]["PolicyDocument"]["Statement"]
    raise AssertionError("CollaboratorRoleDefaultPolicy not found in synthesized template")


def test_collaborator_role_can_write_data_and_read_results():
    statements = _collaborator_policy_statements(_synth_template())

    def _resource_ends_with(statement, suffix):
        resource = statement["Resource"]
        join_parts = resource.get("Fn::Join", [None, []])[1]
        return any(isinstance(part, str) and part.endswith(suffix) for part in join_parts)

    assert any(
        s["Action"] == ["s3:PutObject", "s3:GetObject", "s3:DeleteObject"] and _resource_ends_with(s, "/data/*")
        for s in statements
    )
    assert any(s["Action"] == "s3:GetObject" and _resource_ends_with(s, "/results/*") for s in statements)


def test_collaborator_role_can_query_athena_workgroup():
    statements = _collaborator_policy_statements(_synth_template())

    def _resource_ends_with(statement, suffix):
        resource = statement["Resource"]
        if isinstance(resource, str):
            return resource.endswith(suffix)
        join_parts = resource.get("Fn::Join", [None, []])[1]
        return any(isinstance(part, str) and part.endswith(suffix) for part in join_parts)

    assert any(
        isinstance(s["Action"], list)
        and "athena:StartQueryExecution" in s["Action"]
        and _resource_ends_with(s, "workgroup/steering-visualization")
        for s in statements
    )
