import aws_cdk as cdk

from steering_evaluation.evaluation_stack import SteeringVisualizationEvaluationStack

app = cdk.App()
SteeringVisualizationEvaluationStack(app, "SteeringVisualizationEvaluationStack")
app.synth()
