## Lifecycle integration

The lifecycle manager resolves `OperatorSubscription` resources into
`OperatorInstallPlan` resources. An install plan is executed only after its
approval policy allows it:

- `Automatic` plans are evaluated and executed by the lifecycle policy.
- `Manual` plans remain `WaitingForApproval` until `spec.approved=true`.
- Executed plans report step statuses and end in `Complete` or `Failed`.

Install the lifecycle manager and CRDs with:

```bash
make deploy-dev
```

The development sample catalog, package, and subscription can be installed
from [config/samples/](/home/samorca/osha/operator-lifecycle/config/samples):

```bash
kubectl apply -k config/samples
kubectl -n osha-lifecycle-system patch operatorinstallplan \
  osha-operator-plan-<suffix> --type=merge -p '{"spec":{"approved":true}}'
```