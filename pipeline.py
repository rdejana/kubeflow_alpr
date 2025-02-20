from typing import Dict, List
import os
from kfp import components
from kfp import dsl
from kfp.dsl import Input, InputPath, Output, OutputPath, Dataset, Model, component, Artifact
import kfp.compiler as compiler
from kfp import kubernetes


@dsl.container_component
def git_clone(repo_uri: str, branch: str):
    """Log a greeting and return it as an output."""
    return dsl.ContainerSpec(
        image='alpine/git',
        command=[
            'sh', '-c', '''git clone --depth=1 --branch $0 $1 /data && chmod a+rwx /data'''
        ],

        args=[branch,repo_uri]
    )

@dsl.container_component
def ls():
    """Log a greeting and return it as an output."""
    return dsl.ContainerSpec(
        image='alpine',
        command=[
            'sh', '-c', '''chmod -R a+rw /data && ls -li / && ls -li /data'''
        ],

        args=[]
    )

@dsl.container_component
def ls2():
    """Log a greeting and return it as an output."""
    return dsl.ContainerSpec(
        image='alpine',
        command=[
            'sh', '-c', '''ls -li /data'''
        ],

        args=[]
    )

@dsl.container_component
def prep():
    """Log a greeting and return it as an output."""
    return dsl.ContainerSpec(
        image='quay.io/rdejana/python:0.5m',
        command=[
            'sh', '-c', '''ls -li /data && whoami && bash /data/downloadAndPrepBasic.sh'''
        ],

        args=[]
    )

@dsl.container_component
def train(s3Endpoint:str,
          s3AccessKey: str,
          s3SecretKey: str,
          epochs:int,
          model_name: str,
          saved_model: Output[Model]):
    """Log a greeting and return it as an output."""
    saved_model.framework = 'pytorch'
    return dsl.ContainerSpec(
        image='quay.io/rdejana/python:0.5m',
        command=[
            'sh', '-c', '''/usr/local/bin/python /data/train.py $1 $2 $3 $4 $5 && mv /data/model.pth $0'''
        ],

        args=[saved_model.path,s3Endpoint,s3AccessKey,s3SecretKey,epochs,model_name]
    )

@dsl.pipeline(name='exploring')
def my_pipeline(s3_endpoint: str="minio-service:9000",
                s3_accesskey: str="minio",
                s3_secretkey: str="minio123",
                epochs: int=2,
                model_name: str = "fasterrcnn_resnet50_fpn"):
    # let's start by adding the bucket stuff
    pvc1 = kubernetes.CreatePVC(
        # can also use pvc_name instead of pvc_name_suffix to use a pre-existing PVC
        pvc_name_suffix='-my-pvc',
        access_modes=['ReadWriteOnce'],
        size='5Gi',
        storage_class_name='standard',
    )

    git_task = git_clone(repo_uri='https://github.com/rdejana/kubeflow_alpr',branch='main')
    kubernetes.mount_pvc(
        git_task,
        pvc_name=pvc1.outputs['name'],
        mount_path='/data',
    )



    prep_task = prep()
    kubernetes.mount_pvc(
        prep_task,
        pvc_name=pvc1.outputs['name'],
        mount_path='/data',
    )

    prep_task.after(git_task)



    train_task = train(
        s3Endpoint=s3_endpoint
        ,s3AccessKey=s3_accesskey,
        s3SecretKey=s3_secretkey,
        epochs=epochs,
        model_name=model_name)
    kubernetes.mount_pvc(
        train_task,
        pvc_name=pvc1.outputs['name'],
        mount_path='/data',
    )
    # figured out how to set limits
    train_task.set_cpu_limit("6000m")
    train_task.set_memory_limit("10G")
    train_task.after(prep_task)



    delete_pvc1 = kubernetes.DeletePVC(
        pvc_name=pvc1.outputs['name']
    ).after(train_task)

compiler.Compiler().compile(my_pipeline, package_path='pvc.yaml')