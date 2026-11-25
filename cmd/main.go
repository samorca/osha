package main

import (
	"flag"
	"os"

	lifecyclev1alpha1 "github.com/samorca/osha/operator-lifecycle/api/v1alpha1"
	"github.com/samorca/osha/operator-lifecycle/internal/catalog"
	"github.com/samorca/osha/operator-lifecycle/internal/installer"
	"github.com/samorca/osha/operator-lifecycle/internal/policy"
	"github.com/samorca/osha/operator-lifecycle/internal/resolver"
	"github.com/samorca/osha/operator-lifecycle/internal/subscription"
	"github.com/samorca/osha/operator-lifecycle/internal/upgrade"

	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"

	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/healthz"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"
)

var (
	scheme   = runtime.NewScheme()
	setupLog = ctrl.Log.WithName("setup")
)

func init() {
	utilruntime.Must(
		clientgoscheme.AddToScheme(scheme),
	)

	utilruntime.Must(
		lifecyclev1alpha1.AddToScheme(scheme),
	)
}

func main() {
	var metricsAddress string
	var probeAddress string
	var enableLeaderElection bool

	flag.StringVar(
		&metricsAddress,
		"metrics-bind-address",
		":8080",
		"Address used by the metrics server.",
	)

	flag.StringVar(
		&probeAddress,
		"health-probe-bind-address",
		":8081",
		"Address used by health and readiness probes.",
	)

	flag.BoolVar(
		&enableLeaderElection,
		"leader-elect",
		false,
		"Enable leader election for the lifecycle manager.",
	)

	logOptions := zap.Options{
		Development: false,
	}

	logOptions.BindFlags(flag.CommandLine)

	flag.Parse()

	ctrl.SetLogger(
		zap.New(
			zap.UseFlagOptions(&logOptions),
		),
	)

	manager, err := ctrl.NewManager(
		ctrl.GetConfigOrDie(),
		ctrl.Options{
			Scheme: scheme,

			Metrics: metricsserver.Options{
				BindAddress: metricsAddress,
			},

			HealthProbeBindAddress: probeAddress,

			LeaderElection: enableLeaderElection,

			LeaderElectionID: "operator-lifecycle.lifecycle.osha.io",
		},
	)
	if err != nil {
		setupLog.Error(
			err,
			"unable to create lifecycle manager",
		)

		os.Exit(1)
	}

	catalogStore := catalog.NewMemoryStore()

	lifecycleResolver, err := resolver.New(
		catalogStore,
	)
	if err != nil {
		setupLog.Error(
			err,
			"unable to create lifecycle resolver",
		)

		os.Exit(1)
	}

	policyConfig := policy.DefaultConfig()

	policyEngine := policy.New(
		policyConfig,
	)

	installExecutor, err := installer.New(
		manager.GetClient(),
		policyEngine,
	)
	if err != nil {
		setupLog.Error(
			err,
			"unable to create lifecycle installer",
		)

		os.Exit(1)
	}

	upgradeEngine := upgrade.New()

	planBuilder := subscription.NewDeploymentPlanBuilder()

	subscriptionReconciler := &subscription.Reconciler{
		Client:       manager.GetClient(),
		Scheme:       manager.GetScheme(),
		CatalogStore: catalogStore,
		Resolver:     lifecycleResolver,
		PlanBuilder:  planBuilder,
	}

	if err := subscriptionReconciler.SetupWithManager(manager); err != nil {
		setupLog.Error(
			err,
			"unable to register OperatorSubscription controller",
		)

		os.Exit(1)
	}

	_ = installExecutor
	_ = upgradeEngine

	if err := manager.AddHealthzCheck(
		"healthz",
		healthz.Ping,
	); err != nil {
		setupLog.Error(
			err,
			"unable to configure health check",
		)

		os.Exit(1)
	}

	if err := manager.AddReadyzCheck(
		"readyz",
		healthz.Ping,
	); err != nil {
		setupLog.Error(
			err,
			"unable to configure readiness check",
		)

		os.Exit(1)
	}

	setupLog.Info(
		"starting OSHA Operator Lifecycle Manager",
		"metricsAddress",
		metricsAddress,
		"probeAddress",
		probeAddress,
		"leaderElection",
		enableLeaderElection,
	)

	if err := manager.Start(
		ctrl.SetupSignalHandler(),
	); err != nil {
		setupLog.Error(
			err,
			"lifecycle manager exited with an error",
		)

		os.Exit(1)
	}
}
