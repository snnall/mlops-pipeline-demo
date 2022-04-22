##  Welcome to MLOps pipeline project using Amazon SageMaker Pipelines

This project utilizes SageMaker Pipelines that offers machine learning (ML) application developers and operations engineers the ability to orchestrate SageMaker jobs and author reproducible ML pipelines. It enables users to deploy custom-build models for batch and real-time inference with low latency and track lineage of artifacts.

## Get Started

This project is templatized with Amazon CDK. The `cdk.json` file tells the CDK Toolkit how to execute your app.

This project is set up like a standard Python project. The initialization process also creates a virtualenv within this project, stored under the `.venv` directory. To create the virtualenv it assumes that there is a python3 executable in your path with access to the venv package. If for any reason the automatic creation of the virtualenv fails, you can create the virtualenv manually once the init process completes.

To manually create a virtualenv on MacOS and Linux:
```
python3 -m venv .venv
```

After the init process completes and the virtualenv is created, you can use the following step to activate your virtualenv.
```
$ source .venv/bin/activate
```

Once the virtualenv is activated, you can install the required dependencies.
```
pip install -r requirements.txt
```

At this point you can now synthesize the CloudFormation template for this code.

```
cdk synth
cdk deploy --all --outputs-file ./cdk-outputs.json
```
or you can also deploy the stack by running : `cdk deploy regml-stack --outputs-file ./cdk-outputs.json`

Note: The output file parameter will automate the transfer of your created IAM role ARN to pipeline.py.

Once the stack is created, run the following command:
```
python pipeline.py
```

To add additional dependencies, for example other CDK libraries, just add to your requirements.txt file and rerun the `pip install -r requirements.txt` command.

Useful commands
```
`cdk ls` list all stacks in the app
`cdk synth` emits the synthesized CloudFormation template
`cdk deploy` deploy this stack to your default AWS account/region
`cdk diff` compare deployed stack with current state
`cdk docs` open CDK documentation
```

