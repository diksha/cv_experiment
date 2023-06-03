#
# Copyright 2020-2021 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#
import argparse
import graphlib
import uuid

from google.cloud import firestore

from core.infra.symphony.structs.job import Job


def get_job_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("firestore_path", type=str)
    args = parser.parse_args()
    return args


def upload_jobs_to_firestore(local_jobs, buildkite_jobs):
    db = firestore.Client()
    config = {
        "local_jobs": [job_struct.to_dict() for job_struct in local_jobs],
        "buildkite_jobs": [
            job_struct.to_dict() for job_struct in buildkite_jobs
        ],
    }
    doc_uuid = str(uuid.uuid4())
    doc_ref = db.collection(u"symphony").document(doc_uuid)
    doc_ref.set(config)
    for idx, job_struct in enumerate(local_jobs):
        doc_ref.collection("local_jobs").document(str(idx)).set(
            job_struct.to_dict()
        )
    for idx, job_struct in enumerate(buildkite_jobs):
        doc_ref.collection("buildkite_jobs").document(str(idx)).set(
            job_struct.to_dict()
        )
    return doc_uuid


def load_jobs_from_firestore(config_firestore_uuid):
    db = firestore.Client()
    doc_ref = db.collection(u"symphony").document(config_firestore_uuid)
    config = doc_ref.get().to_dict()
    config_object = {
        "local_jobs": [
            Job.from_dict(job) for job in config.get("local_jobs", [])
        ],
        "buildkite_jobs": [
            Job.from_dict(job) for job in config.get("buildkite_jobs", [])
        ],
    }
    return config_object


def load_job_struct_from_firestore(firestore_path):
    db = firestore.Client()
    doc_ref = db.document(firestore_path)
    return Job.from_dict(doc_ref.get().to_dict())


def create_jobs(config, job_type):
    jobs = []
    for job_config in config.get(job_type, []):
        jobs.append(Job.from_dict(job_config))
        jobs = _remove_duplicates_and_top_sort(jobs)
    return jobs


def _remove_duplicates_and_top_sort(jobs):
    # Remove duplicates
    jobs = list({job.differentiator: job for job in jobs}.values())
    # Top sort and return ordered.
    job_name_object_mapping = {job.name: job for job in jobs}
    dependency_graph = {job.name: set(job.depends_on) for job in jobs}
    ts = graphlib.TopologicalSorter(dependency_graph)
    return [
        job_name_object_mapping[job_name]
        for job_name in tuple(ts.static_order())
    ]
