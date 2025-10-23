# Contributing

In general, [MarcDiethelm](https://github.com/MarcDiethelm/contributing) did a good job summarising the steps to contribute on any GitHub project which can also be applied to this project, apart from the testing part, there are no tests as of the time of writing this (oops).

However, there is some additional guidance necessary explaining the workflow to update the container images related to this project.

## Updating Container Images

Throughout developing the services in this project, you'll need to build the modified code into a new image, push it to the packages of this project, and apply the new image as a container in the cluster by restarting the pod that container resides in. These steps can be executed using `deploy.sh` scripts included in every service's directory.

Every image has two tags: `latest` for production and `dev` for development. When running `deploy.sh` as is, the image created will bear the `dev` tag. When running `deploy.sh prod`, it will bear the `latest` tag. Running the script for production or development applies the image with one of two `yaml` configurations of that service.

I've devised this system so that a cluster using all `latest` images would always be the latest fully working version. However, I found out that making use of the `dev` tags didn't always work in practice. For some reason, unbeknownst to me, the K8s cluster would choose to apply the `latest` tagged image either way.

A similar issue arrises when trying to apply the `flower-clientapp` (code within the flower directory) within `trainer`. You'd need to redeploy `server` so that `flower-serverapp` (also code within the flower directory) is rebuilt. My best guess is that these images share a few layers containing the same code which results in the cluster using some sort of cache.

To evade all these issues, test your code using the `latest` images unless you've found a solution.

Finally, I can recommend using the CLI app [k9s](https://k9scli.io/) to manage your K8s cluster.