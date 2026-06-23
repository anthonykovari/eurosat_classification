#!/bin/bash
awslocal s3 mb s3://chicago-land-use --region us-east-1 2>/dev/null || true
