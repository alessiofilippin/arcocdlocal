# ArgoCD Local Deployment Environment

## 🚧 MAN-AT-WORK

This project is currently under active development and is not ready for production use.

## 1 - Overview

This project provides a local Kubernetes development environment to experiment with GitOps using ArgoCD and workflow automation via Argo Workflows. It is designed for testing machine learning workflows and deploying applications using lightweight Docker images, minimal compute, and a simplified infrastructure footprint.

---

## 2 - Repository Structure

- **`argo-workflow/`**: Argo Workflow templates defining machine learning pipelines and utility jobs.
- **`argocd-templates/`**: ArgoCD Application templates that manage deployments of services.
- **`cluster-conf/`**: Kubernetes manifests to bootstrap the cluster (e.g., namespaces, service accounts).
- **`docker/`**: Dockerfiles and scripts to build container images used throughout the environment.
- **`inference-ui/`**: Static frontend UI to perform inference through a web interface.
- **`private-registry/`**: Setup files for a private Docker registry used to store locally built images.

---

## 3 - Disclaimer

This environment was tested exclusively on a local Kubernetes cluster composed of **three Ubuntu nodes** with **limited compute and disk resources**. For this reason:

- All Docker images are minimal and purpose-built.
- Machine Learning training is extremely simplified and uses **CPU-only** execution.
- Node `ubuntu2` had the highest compute capacity, which is why some templates in this repository specify a `nodeSelector` to schedule heavier tasks to that node.

Please adjust nodeSelectors if your cluster layout is different. Or remove it completely if you have a cluster with more compute resources available.

**THIS DOCUMENTATION HAS BEEN CREATED WITH THE HELP OF AI.**

---
## 4 - Storage

I used 2 type of storage for the experiment:

1) A local storage attached to Ubuntu02 and configured with pv and pvc.

    Ubuntu02 was the VM used to train the models because it's where the majority of the avalable compute power was allocated.
    Docker images would also be saved on Ubuntu02.

2) NFS volumes shared from Ubuntu03

    The NFS volumes were used to make DVC repo related tasks and the Inference UI work from any node.

---
## 5 - Requirements

To use this setup, you must have the following installed and configured:

- A running **Kubernetes cluster**
- [**ArgoCD**](https://argo-cd.readthedocs.io/)
- [**Argo Workflows**](https://argoproj.github.io/argo-workflows/)
- [**Kustomize**](https://kubectl.docs.kubernetes.io/installation/kustomize/)

---

## 6 - Flows

This section describes each Argo Workflow in the `argo-workflow/` directory and its purpose in the local MLOps pipeline.

### `train-new-model-version.yaml`
This workflow handles the training of a new machine learning model version. It performs the following steps:

- Pulls a subset of training data from IMDB_Dataset.
- Launches a training job using a lightweight container.
- Trains the model using CPU resources only.
- Saves the trained model to a shared persistent volume.
- Tags/Save the model version using a tag using DVC.

This pipeline is CPU-optimized and assumes limited compute and memory availability.

### `train-and-inference-testing.yaml`
This workflow extends the training process by including an inference validation phase. It:

- Trains a new model using the same procedure as in `train-new-model-version.yaml`.
- After training, it deploys the model for local inference testing.
- Sends test inputs to the inference service.
- Logs results and verifies outputs for basic validation.
- No models are saved or tagged with DVC.

This is useful for quickly checking model quality after training, all within the same automated flow.

### `promote-model-version.yaml`
This workflow promotes a validated model version to be used in production or as the default. It:

- Pull the correct tag of the model from DVC.
- Saves the model version in the current version folder used by the UI.

This flow represents the final approval step in a lightweight CI/CD-style pipeline.

---

## 7 - Inference-UI Rollout Flow

The `inference-ui` component provides a very simple user interface for inference operations. The rollout process for `inference-ui` is managed through ArgoCD and follows these steps:

1. **Manifest Definition**  
   The Kubernetes manifests for `inference-ui` are located in the `inference-ui/` directory. These include Deployment, Service, and Ingress resources necessary for the application.

2. **Application Configuration**  
   An ArgoCD Application resource is defined to manage the `inference-ui` deployment. This configuration specifies the source repository, target revision, and the path to the manifests.

3. **Deployment Process**  
   - ArgoCD monitors the specified Git repository for changes in the `inference-ui` manifests.  
   - Upon detecting changes, ArgoCD synchronizes the live cluster state with the desired state defined in the repository.  
   - The synchronization process involves creating or updating Kubernetes resources to match the manifests.

4. **Monitoring and Health Checks**  
   - ArgoCD provides real-time monitoring of the `inference-ui` application.  
   - Health checks are configured to ensure the application is running as expected.  
   - Any discrepancies or issues are reported in the ArgoCD dashboard for prompt resolution.

This automated rollout process ensures that updates to the `inference-ui` component are deployed consistently and reliably across environments.

---

## 🔮 Future Improvements

To enhance the functionality and usability of this project, the following improvements are planned:

- **CI/CD Integration**  
  Implement continuous integration and deployment pipelines to automate testing and deployment processes, improving efficiency and reliability.

- **Parameterization**  
  Introduce parameterization in the Kubernetes manifests and Docker Images to allow for dynamic configuration.

- **User Interface Enhancements**  
  Improve the `inference-ui` component with additional features and a more intuitive user interface to enhance user experience.

- **LLM Monitoring and Experimenting**  
  Include the proper monitoring stack to make us of DVC's experimenting features.

---
