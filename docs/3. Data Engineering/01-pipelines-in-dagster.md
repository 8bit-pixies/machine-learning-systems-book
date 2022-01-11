---
title: Managing Pipelines
---

Building feature pipelines is important as part of production. Here I'll recommend approaches which can enable you to test deployments locally and also in a managed cloud environment. There are several options here, we'll recommend using:

*  dagster
*  luigi

In my (albeit) limited experience Airflow suffers from the challenges surrounding ability to ensure local development workflows can run well in the cloud. We can make use of Kubernetes runners to ensure that workflows can operate in the expected manner. 

Using dagster and luigi also have straightforward processes and can both be run without a daemon or server for testing before using the centralized runner. This enables a split between development and production!


