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
            'sh', '-c', '''ls -li /data/training'''
        ],

        args=[]
    )

@dsl.container_component
def prep():
    """Log a greeting and return it as an output."""
    return dsl.ContainerSpec(
        image='quay.io/rdejana/python:0.1',
        command=[
            'sh', '-c', '''ls -li /data && whoami && bash /data/downloadAndPrepBasic.sh'''
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


    task3 = prep()
    kubernetes.mount_pvc(
        task3,
        pvc_name=pvc1.outputs['name'],
        mount_path='/data',
    )

    task3.after(task2)

    task4 = ls()
    kubernetes.mount_pvc(
        task4,
        pvc_name=pvc1.outputs['name'],
        mount_path='/data',
    )
    task4.set_caching_options(False)
    task4.after(task3)


    delete_pvc1 = kubernetes.DeletePVC(
        pvc_name=pvc1.outputs['name']
    ).after(task4)

compiler.Compiler().compile(my_pipeline, package_path='pvc.yaml')