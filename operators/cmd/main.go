package main

import (
	"flag"
	"os"

	platformv1alpha1 "github.com/samorca/osha/operators/api/v1alpha1"
	"github.com/samorca/osha/operators/internal/controller"

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
	utilruntime.Must(clientgoscheme.AddToScheme(scheme))
	utilruntime.Must(platformv1alpha1.AddToScheme(scheme))
}

func main() {
	var metricsAddr string
	var probeAddr string
	var enableLeaderElection bool

	flag.StringVar(
		&metricsAddr,
		"metrics-bind-address",
		":8080",
		"Address used by the metrics endpoint.",
	)

	flag.StringVar(
		&probeAddr,
		"health-probe-bind-address",
		":8081",
		"Address used by health and readiness probes.",
	)

	flag.BoolVar(
		&enableLeaderElection,
		"leader-elect",
		false,
		"Enable leader election for the controller manager.",
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

	config := ctrl.GetConfigOrDie()

	manager, err := ctrl.NewManager(
		config,
		ctrl.Options{
			Scheme: scheme,

			Metrics: metricsserver.Options{
				BindAddress: metricsAddr,
			},

			HealthProbeBindAddress: probeAddr,

			LeaderElection: enableLeaderElection,

			LeaderElectionID: "osha-operator.platform.osha.io",
		},
	)

	if err != nil {
		setupLog.Error(
			err,
			"unable to create controller manager",
		)

		os.Exit(1)
	}

	reconciler := &controller.OSHAPlatformReconciler{
		Client: manager.GetClient(),
		Scheme: manager.GetScheme(),
	}

	if err := reconciler.SetupWithManager(manager); err != nil {
		setupLog.Error(
			err,
			"unable to register OSHAPlatform controller",
		)

		os.Exit(1)
	}

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
		"starting OSHA Operator",
		"metricsAddress",
		metricsAddr,
		"probeAddress",
		probeAddr,
		"leaderElection",
		enableLeaderElection,
	)

	if err := manager.Start(
		ctrl.SetupSignalHandler(),
	); err != nil {
		setupLog.Error(
			err,
			"OSHA Operator manager exited with an error",
		)

		os.Exit(1)
	}
}
