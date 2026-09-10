# Deployment

MLServer is used as a Python inference server in [KServe (formerly known
as
KFServing)](https://kserve.github.io/website/).
This allows MLServer users to leverage the usability and maturity of KServe to
take their model deployments to the next level of their MLOps journey,
ensuring that they are served in a robust and scalable infrastructure.

```{note}
In general, it should be possible to deploy models using MLServer into **any
serving engine compatible with the V2 protocol**.
Alternatively, it's also possible to manage MLServer deployments manually as
regular processes (i.e. in a non-Kubernetes-native way).
However, this may be more involved and highly dependant on the deployment
infrastructure.
```

`````{grid} 1
````{grid-item-card}
:class-card: sd-px-5 sd-pt-2
:link: ./kserve
:link-type: doc
:img-top: ../../assets/kserve-logo.png

+++

```{button-ref} ./kserve
:ref-type: doc
:align: center
:class: stretched-link

Deploy with KServe
```
````
`````

```{toctree}
:hidden:
:titlesonly:
:maxdepth: 1

./kserve.md
```
