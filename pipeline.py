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
            'sh', '-c', '''git clone --depth=1 --branch $0 $1 /data'''
        ],

        args=[branch,repo_uri]
    )

@dsl.container_component
def ls():
    """Log a greeting and return it as an output."""
    return dsl.ContainerSpec(
        image='alpine',
        command=[
            'sh', '-c', '''ls -li /data'''
        ],

        args=[]
    )

@dsl.pipeline
def my_pipeline():
    pvc1 = kubernetes.CreatePVC(
        # can also use pvc_name instead of pvc_name_suffix to use a pre-existing PVC
        pvc_name_suffix='-my-pvc',
        access_modes=['ReadWriteOnce'],
        size='5Gi',
        storage_class_name='standard',
    )

    task1 = git_clone(repo_uri='https://github.com/rdejana/kubeflow_alpr',branch='main')
    kubernetes.mount_pvc(
        task1,
        pvc_name=pvc1.outputs['name'],
        mount_path='/data',
    )
    task2 = ls()
    kubernetes.mount_pvc(
        task2,
        pvc_name=pvc1.outputs['name'],
        mount_path='/data',
    )

    task2.after(task1)

    delete_pvc1 = kubernetes.DeletePVC(
        pvc_name=pvc1.outputs['name']
    ).after(task2)

compiler.Compiler().compile(my_pipeline, package_path='pvc.yaml')