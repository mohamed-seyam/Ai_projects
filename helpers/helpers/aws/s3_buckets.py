import os 
import boto3
from botocore.exceptions import ClientError


import os
import sys
import threading

class ProgressPercentage(object):

    def __init__(self, filename):
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):
        # To simplify, assume this is hooked up to a single filename
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            sys.stdout.write(
                "\r%s  %s / %s  (%.2f%%)" % (
                    self._filename, self._seen_so_far, self._size,
                    percentage))
            sys.stdout.flush()

def create_bucket(bucket_name, region=None):
    """Create an S3 bucket in a specified region

    If a region is not specified, the bucket is created in the S3 default
    region (us-east-1).

    :param bucket_name: Bucket to create
    :param region: Region in which to create bucket. 
    :return: True if bucket created, else False
    """
    # Create bucket
    try:
        if region is None:
            s3_client = boto3.client('s3')
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            s3_client = boto3.client('s3', region_name=region)
            location = {'LocationConstraint': region}
            s3_client.create_bucket(Bucket=bucket_name,
                                    CreateBucketConfiguration=location)
    except ClientError as e:
        print(e)
        return False
    return True

def list_buckets():
    """List all buckets owned by the authenticated sender of the request

    :return: List of bucket names
    """
    # Retrieve the list of existing buckets
    s3_client = boto3.client('s3')
    response = s3_client.list_buckets()

    # Output the bucket names
    print('Existing buckets:')
    for bucket in response['Buckets']:
        print(f'  {bucket["Name"]}')

def delete_bucket(bucket_name):
    """Delete an empty bucket

    :param bucket_name: string
    :return: True if the referenced bucket was deleted, otherwise False
    """
    # Delete the bucket
    s3_client = boto3.client('s3')
    try:
        s3_client.delete_bucket(Bucket=bucket_name)
    except ClientError as e:
        print(e)
        return False
    return True

def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name.
        If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name, Callback=ProgressPercentage(file_name))
    except ClientError as e:
        print(e)
        return False
    return True



def example_1():
    """create a new bucket with name = "test-15599" """
    if create_bucket("test-15599"):
        print("Bucket created successfully")
    else:
        print("Bucket creation failed")

def example_2():
    """list all buckets"""
    list_buckets()

def example_3():
    """delete a bucket"""
    if delete_bucket("test-15599"):
        print("Bucket deleted successfully")
    else:
        print("Bucket deletion failed")

def example_4():
    """upload a file to a bucket"""
    for file in ["./helpers/helpers/aws/file_1.txt", "./helpers/helpers/aws/file_2.txt"]:
        if upload_file(file, "test-2235", f"inside/{os.path.basename(file)}"):
            print("File uploaded successfully")
        else:
            print("File upload failed")

if __name__ == "__main__":
    # example_1()
    example_4()
