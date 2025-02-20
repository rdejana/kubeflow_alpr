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
        image='quay.io/rdejana/python:0.4.1',
        command=[
            'sh', '-c', '''ls -li /data && whoami && bash /data/downloadAndPrepBasic.sh'''
        ],

        args=[]
    )

@dsl.container_component
def train(saved_model: Output[Model]):
    """Log a greeting and return it as an output."""
    return dsl.ContainerSpec(
        image='quay.io/rdejana/python:0.5',
        command=[
            'sh', '-c', '''/usr/local/bin/python /data/train.py && mv model.pth $0'''
        ],

        args=[saved_model.path]
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

    task5 = train()
    kubernetes.mount_pvc(
        task5,
        pvc_name=pvc1.outputs['name'],
        mount_path='/data',
    )
    task5.set_cpu_limit("6000m")
    task5.set_memory_limit("10G")
    task5.after(task4)

    task6 = ls2()
    kubernetes.mount_pvc(
        task6,
        pvc_name=pvc1.outputs['name'],
        mount_path='/data/saved',
    )
    task6.after(task5)

    delete_pvc1 = kubernetes.DeletePVC(
        pvc_name=pvc1.outputs['name']
    ).after(task6)

compiler.Compiler().compile(my_pipeline, package_path='pvc.yaml')