from pathlib import Path

from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
    aws_athena as athena,
    aws_glue as glue,
    aws_iam as iam,
    aws_lambda as lambda_,
    aws_lambda_destinations as lambda_destinations,
    aws_logs as logs,
    aws_s3 as s3,
    aws_s3_notifications as s3n,
    aws_sqs as sqs,
)
from constructs import Construct

BUCKET_NAME = "steering-visualization"
GLUE_DATABASE_NAME = "steering_visualization"
RECOGNITION_TABLE_NAME = "recognition_results"
EVALUATION_TABLE_NAME = "evaluation_results"
ATHENA_WORKGROUP_NAME = "steering-visualization"
COLLABORATOR_ROLE_NAME = "steering-visualization-collaborator"
_INPUT_FORMAT = "org.apache.hadoop.mapred.TextInputFormat"
_OUTPUT_FORMAT = "org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat"
_SERIALIZATION_LIBRARY = "org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe"
_PARTITION_KEYS = [
    glue.CfnTable.ColumnProperty(name="layer", type="string"),
    glue.CfnTable.ColumnProperty(name="category", type="string"),
    glue.CfnTable.ColumnProperty(name="label", type="string"),
    glue.CfnTable.ColumnProperty(name="variant", type="string"),
    glue.CfnTable.ColumnProperty(name="prompt_type", type="string"),
]

_REGION = "us-east-1"
# PROVISIONAL: the original paper uses GPT-5/GPT-5-mini; the closest Bedrock
# equivalent (openai.gpt-5.6-terra/luna) never got authorized on this account
# despite accepting the model agreement (AccessDeniedException persisted for
# hours after `authorizationStatus: AUTHORIZED`) -- see design spec section 4.
# Swapped to Claude, which was immediately available. Revert to:
#   _RECOGNITION_MODEL_ID = "us.openai.gpt-5.6-terra"
#   _RECOGNITION_FOUNDATION_MODEL_ID = "openai.gpt-5.6-terra"
#   _EVALUATION_MODEL_ID = "us.openai.gpt-5.6-luna"
#   _EVALUATION_FOUNDATION_MODEL_ID = "openai.gpt-5.6-luna"
_RECOGNITION_MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
_RECOGNITION_FOUNDATION_MODEL_ID = "anthropic.claude-sonnet-4-5-20250929-v1:0"
_EVALUATION_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
_EVALUATION_FOUNDATION_MODEL_ID = "anthropic.claude-haiku-4-5-20251001-v1:0"
_STRINGENT_PROMPT = "What is in the image? One word."
_LENIENT_PROMPT_TEMPLATE = "What {category} is in the image if you had to guess? One word."
_EVAL_PROMPT_TEMPLATE = (
    "Please carefully read the following user request (denoted as 'REQUEST') and\n"
    "response (denoted as 'RESPONSE').\n"
    "<BEGIN REQUEST>\n{request}\n<END REQUEST>\n"
    "<BEGIN RESPONSE>\n{response}\n<END RESPONSE>\n"
    "You are evaluating a model that answered a question (REQUEST) about an image.\n"
    "Analyze the RESPONSE, then score it according to the following rubric:\n"
    "0: In the span of the answer, the model doesn't mention it sees '{concept}' in\n"
    "the image. E.g. it mentions other concepts, but not '{concept}'.\n"
    "1: In the span of the answer, the model does mention it sees '{concept}' in the\n"
    "image. Either it does so explicitly, or it is clear from the context of the\n"
    "REQUEST and the content of the RESPONSE. It is okay if it mentions other\n"
    "things in addition to '{concept}'.\n\n"
    'End your answer with "Result: 0" or "Result: 1", depending on the score.'
)
_NUM_ROUNDS = "10"
_LAMBDAS_DIR = str(Path(__file__).resolve().parent.parent / "lambdas")


class SteeringVisualizationEvaluationStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.bucket = self._build_bucket()
        self.database = self._build_glue_database()
        self.recognition_table = self._build_table(
            "RecognitionResultsTable", RECOGNITION_TABLE_NAME, [{"Name": "output", "Type": "string"}],
        )
        self.evaluation_table = self._build_table(
            "EvaluationResultsTable", EVALUATION_TABLE_NAME, [{"Name": "evaluation", "Type": "string"}],
        )
        self.athena_workgroup = self._build_athena_workgroup()
        self.collaborator_role = self._build_collaborator_role()
        CfnOutput(self, "CollaboratorRoleArn", value=self.collaborator_role.role_arn)

        self.recognition_function = self._build_recognition_function()
        self.evaluation_function = self._build_evaluation_function()
        self.cleanup_function = self._build_cleanup_function()

        self.bucket.add_event_notification(
            s3.EventType.OBJECT_CREATED,
            s3n.LambdaDestination(self.recognition_function),
            s3.NotificationKeyFilter(prefix="data/"),
        )
        self.bucket.add_event_notification(
            s3.EventType.OBJECT_CREATED,
            s3n.LambdaDestination(self.evaluation_function),
            s3.NotificationKeyFilter(prefix="results/", suffix=".txt"),
        )
        self.bucket.add_event_notification(
            s3.EventType.OBJECT_REMOVED,
            s3n.LambdaDestination(self.cleanup_function),
            s3.NotificationKeyFilter(prefix="data/"),
        )

    # -- storage -----------------------------------------------------

    def _build_bucket(self) -> s3.Bucket:
        return s3.Bucket(
            self,
            "Bucket",
            bucket_name=BUCKET_NAME,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.RETAIN,
        )

    def _build_glue_database(self) -> glue.CfnDatabase:
        return glue.CfnDatabase(
            self,
            "GlueDatabase",
            catalog_id=self.account,
            database_input=glue.CfnDatabase.DatabaseInputProperty(name=GLUE_DATABASE_NAME),
        )

    def _build_table(self, construct_id: str, table_name: str, columns: list[dict]) -> glue.CfnTable:
        table = glue.CfnTable(
            self,
            construct_id,
            catalog_id=self.account,
            database_name=GLUE_DATABASE_NAME,
            table_input=glue.CfnTable.TableInputProperty(
                name=table_name,
                table_type="EXTERNAL_TABLE",
                partition_keys=_PARTITION_KEYS,
                storage_descriptor=glue.CfnTable.StorageDescriptorProperty(
                    columns=[glue.CfnTable.ColumnProperty(name=c["Name"], type=c["Type"]) for c in columns],
                    location=f"s3://{BUCKET_NAME}/results/",
                    input_format=_INPUT_FORMAT,
                    output_format=_OUTPUT_FORMAT,
                    serde_info=glue.CfnTable.SerdeInfoProperty(serialization_library=_SERIALIZATION_LIBRARY),
                ),
            ),
        )
        table.add_resource_dependency(self.database)
        return table

    def _build_athena_workgroup(self) -> athena.CfnWorkGroup:
        return athena.CfnWorkGroup(
            self,
            "AthenaWorkgroup",
            name=ATHENA_WORKGROUP_NAME,
            work_group_configuration=athena.CfnWorkGroup.WorkGroupConfigurationProperty(
                result_configuration=athena.CfnWorkGroup.ResultConfigurationProperty(
                    output_location=f"s3://{BUCKET_NAME}/athena-query-results/",
                ),
            ),
        )

    def _build_collaborator_role(self) -> iam.Role:
        """Role for colleagues: upload images, run Athena queries, download results.

        Trust is same-account only (AccountRootPrincipal, no literal account
        ID -- CDK resolves it via the AWS::AccountId pseudo-parameter). That
        alone does not let anyone assume it: each collaborator's own IAM user
        also needs an identity policy granting sts:AssumeRole on this role's
        ARN, attached by the account owner (see the admin tutorial) -- this
        keeps the list of who can assume the role out of this CDK stack.
        """
        role = iam.Role(
            self,
            "CollaboratorRole",
            role_name=COLLABORATOR_ROLE_NAME,
            assumed_by=iam.AccountRootPrincipal(),
            description="Upload images, run Athena queries, and download results for the steering-visualization pipeline.",
            max_session_duration=Duration.hours(12),
        )

        # S3: upload/inspect/delete their own images (delete triggers the
        # cleanup lambda automatically, so mistaken uploads self-correct);
        # read-only on results; read/write on the Athena query-results
        # scratch prefix (Athena writes there under the calling principal).
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:PutObject", "s3:GetObject", "s3:DeleteObject"],
                resources=[self.bucket.arn_for_objects("data/*")],
            )
        )
        role.add_to_policy(
            iam.PolicyStatement(actions=["s3:GetObject"], resources=[self.bucket.arn_for_objects("results/*")])
        )
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:GetObject", "s3:PutObject"],
                resources=[self.bucket.arn_for_objects("athena-query-results/*")],
            )
        )
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:ListBucket", "s3:GetBucketLocation"],
                resources=[self.bucket.bucket_arn],
                conditions={"StringLike": {"s3:prefix": ["data/*", "results/*", "athena-query-results/*"]}},
            )
        )

        # Glue: read-only, scoped to this project's catalog/database/tables.
        # GetDatabases has no resource-level scoping in IAM (it is a list-all
        # call) but only returns database names, not table contents.
        role.add_to_policy(iam.PolicyStatement(actions=["glue:GetDatabases"], resources=["*"]))
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["glue:GetDatabase", "glue:GetTable", "glue:GetTables", "glue:GetPartition", "glue:GetPartitions", "glue:BatchGetPartition"],
                resources=self._glue_table_arns(RECOGNITION_TABLE_NAME) + self._glue_table_arns(EVALUATION_TABLE_NAME),
            )
        )

        # Athena: run and inspect queries against the dedicated workgroup only.
        role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "athena:StartQueryExecution",
                    "athena:StopQueryExecution",
                    "athena:GetQueryExecution",
                    "athena:GetQueryResults",
                    "athena:GetQueryResultsStream",
                    "athena:BatchGetQueryExecution",
                    "athena:ListQueryExecutions",
                    "athena:GetWorkGroup",
                ],
                resources=[f"arn:aws:athena:{_REGION}:{self.account}:workgroup/{ATHENA_WORKGROUP_NAME}"],
            )
        )
        return role

    # -- lambdas -------------------------------------------------------

    def _glue_table_arns(self, table_name: str) -> list[str]:
        return [
            f"arn:aws:glue:{_REGION}:{self.account}:catalog",
            f"arn:aws:glue:{_REGION}:{self.account}:database/{GLUE_DATABASE_NAME}",
            f"arn:aws:glue:{_REGION}:{self.account}:table/{GLUE_DATABASE_NAME}/{table_name}",
        ]

    def _bedrock_arns(self, inference_profile_id: str, foundation_model_id: str) -> list[str]:
        return [
            f"arn:aws:bedrock:{_REGION}:{self.account}:inference-profile/{inference_profile_id}",
            f"arn:aws:bedrock:*::foundation-model/{foundation_model_id}",
        ]

    def _base_role(self, construct_id: str) -> iam.Role:
        role = iam.Role(self, construct_id, assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"))
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
                resources=["arn:aws:logs:*:*:*"],
            )
        )
        return role

    def _build_log_group(self, construct_id: str, function_name: str) -> logs.LogGroup:
        return logs.LogGroup(
            self,
            construct_id,
            log_group_name=f"/aws/lambda/{function_name}",
            retention=logs.RetentionDays.THREE_MONTHS,
            removal_policy=RemovalPolicy.DESTROY,
        )

    def _build_recognition_function(self) -> lambda_.Function:
        role = self._base_role("RecognitionRole")
        role.add_to_policy(iam.PolicyStatement(actions=["s3:GetObject"], resources=[self.bucket.arn_for_objects("data/*")]))
        role.add_to_policy(
            iam.PolicyStatement(actions=["s3:GetObject", "s3:PutObject"], resources=[self.bucket.arn_for_objects("results/*")])
        )
        # Without s3:ListBucket, S3 returns 403 (not 404) on HeadObject for a
        # missing key, which breaks the idempotency check in object_exists().
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:ListBucket"],
                resources=[self.bucket.bucket_arn],
                conditions={"StringLike": {"s3:prefix": ["results/*"]}},
            )
        )
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["bedrock:InvokeModel"],
                resources=self._bedrock_arns(_RECOGNITION_MODEL_ID, _RECOGNITION_FOUNDATION_MODEL_ID),
            )
        )
        role.add_to_policy(
            iam.PolicyStatement(actions=["glue:CreatePartition", "glue:GetPartition"], resources=self._glue_table_arns(RECOGNITION_TABLE_NAME))
        )

        dlq = sqs.Queue(self, "RecognitionDlq")
        return lambda_.Function(
            self,
            "RecognitionFunction",
            function_name="recognition",
            runtime=lambda_.Runtime.PYTHON_3_13,
            handler="recognition.handler.handler",
            code=lambda_.Code.from_asset(_LAMBDAS_DIR),
            role=role,
            timeout=Duration.minutes(5),
            memory_size=512,
            on_failure=lambda_destinations.SqsDestination(dlq),
            log_group=self._build_log_group("RecognitionLogGroup", "recognition"),
            environment={
                "NUM_ROUNDS": _NUM_ROUNDS,
                "RECOGNITION_MODEL_ID": _RECOGNITION_MODEL_ID,
                "RECOGNITION_MAX_TOKENS": "20",
                "GLUE_DATABASE": GLUE_DATABASE_NAME,
                "RECOGNITION_TABLE": RECOGNITION_TABLE_NAME,
                "STRINGENT_PROMPT": _STRINGENT_PROMPT,
                "LENIENT_PROMPT_TEMPLATE": _LENIENT_PROMPT_TEMPLATE,
            },
        )

    def _build_evaluation_function(self) -> lambda_.Function:
        role = self._base_role("EvaluationRole")
        role.add_to_policy(
            iam.PolicyStatement(actions=["s3:GetObject", "s3:PutObject"], resources=[self.bucket.arn_for_objects("results/*")])
        )
        # Without s3:ListBucket, S3 returns 403 (not 404) on HeadObject for a
        # missing key, which breaks the idempotency check in object_exists().
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:ListBucket"],
                resources=[self.bucket.bucket_arn],
                conditions={"StringLike": {"s3:prefix": ["results/*"]}},
            )
        )
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["bedrock:InvokeModel"],
                resources=self._bedrock_arns(_EVALUATION_MODEL_ID, _EVALUATION_FOUNDATION_MODEL_ID),
            )
        )
        role.add_to_policy(
            iam.PolicyStatement(actions=["glue:CreatePartition", "glue:GetPartition"], resources=self._glue_table_arns(EVALUATION_TABLE_NAME))
        )

        dlq = sqs.Queue(self, "EvaluationDlq")
        return lambda_.Function(
            self,
            "EvaluationFunction",
            function_name="evaluation",
            runtime=lambda_.Runtime.PYTHON_3_13,
            handler="evaluation.handler.handler",
            code=lambda_.Code.from_asset(_LAMBDAS_DIR),
            role=role,
            timeout=Duration.minutes(1),
            memory_size=256,
            on_failure=lambda_destinations.SqsDestination(dlq),
            log_group=self._build_log_group("EvaluationLogGroup", "evaluation"),
            environment={
                "EVALUATION_MODEL_ID": _EVALUATION_MODEL_ID,
                "EVALUATION_MAX_TOKENS": "500",
                "GLUE_DATABASE": GLUE_DATABASE_NAME,
                "EVALUATION_TABLE": EVALUATION_TABLE_NAME,
                "STRINGENT_PROMPT": _STRINGENT_PROMPT,
                "LENIENT_PROMPT_TEMPLATE": _LENIENT_PROMPT_TEMPLATE,
                "EVAL_PROMPT_TEMPLATE": _EVAL_PROMPT_TEMPLATE,
            },
        )

    def _build_cleanup_function(self) -> lambda_.Function:
        role = self._base_role("CleanupRole")
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:ListBucket"],
                resources=[self.bucket.bucket_arn],
                conditions={"StringLike": {"s3:prefix": ["results/*"]}},
            )
        )
        role.add_to_policy(iam.PolicyStatement(actions=["s3:DeleteObject"], resources=[self.bucket.arn_for_objects("results/*")]))
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["glue:DeletePartition", "glue:GetPartition"],
                resources=self._glue_table_arns(RECOGNITION_TABLE_NAME) + self._glue_table_arns(EVALUATION_TABLE_NAME),
            )
        )

        dlq = sqs.Queue(self, "CleanupDlq")
        return lambda_.Function(
            self,
            "CleanupFunction",
            function_name="cleanup",
            runtime=lambda_.Runtime.PYTHON_3_13,
            handler="cleanup.handler.handler",
            code=lambda_.Code.from_asset(_LAMBDAS_DIR),
            role=role,
            timeout=Duration.minutes(1),
            memory_size=256,
            on_failure=lambda_destinations.SqsDestination(dlq),
            log_group=self._build_log_group("CleanupLogGroup", "cleanup"),
            environment={
                "GLUE_DATABASE": GLUE_DATABASE_NAME,
                "RECOGNITION_TABLE": RECOGNITION_TABLE_NAME,
                "EVALUATION_TABLE": EVALUATION_TABLE_NAME,
            },
        )
