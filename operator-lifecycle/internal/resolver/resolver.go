package resolver

import (
	"errors"
	"fmt"
	"sort"
	"strings"

	"github.com/Masterminds/semver/v3"

	lifecyclev1alpha1 "github.com/samorca/osha/operator-lifecycle/api/v1alpha1"
	"github.com/samorca/osha/operator-lifecycle/internal/catalog"
)

var (
	ErrInvalidRequest         = errors.New("invalid resolution request")
	ErrNoCompatibleRelease    = errors.New("no compatible release")
	ErrDependencyCycle        = errors.New("dependency cycle")
	ErrUnsatisfiedDependency  = errors.New("unsatisfied package dependency")
	ErrRequiredAPIUnavailable = errors.New("required API unavailable")
)

type Request struct {
	Namespace string
	Package   string
	Channel   string

	VersionConstraint string
	StartingVersion   string

	InstalledVersions map[string]string
}

type Selection struct {
	Package   string
	Namespace string
	Channel   string
	Release   lifecyclev1alpha1.OperatorRelease
	Primary   bool
}

type Result struct {
	Primary      Selection
	Dependencies []Selection
	All          []Selection
}

type Resolver struct {
	store catalog.Store
}

type resolutionState struct {
	selected map[string]Selection
	visiting map[string]bool
	path     []string
}

func New(store catalog.Store) (*Resolver, error) {
	if store == nil {
		return nil, errors.New("catalog store is required")
	}

	return &Resolver{
		store: store,
	}, nil
}

func (r *Resolver) Resolve(request Request) (*Result, error) {
	request.Namespace = strings.TrimSpace(request.Namespace)
	request.Package = strings.TrimSpace(request.Package)
	request.Channel = strings.TrimSpace(request.Channel)
	request.VersionConstraint = strings.TrimSpace(
		request.VersionConstraint,
	)
	request.StartingVersion = strings.TrimSpace(
		request.StartingVersion,
	)

	if request.Package == "" {
		return nil, fmt.Errorf(
			"%w: package is required",
			ErrInvalidRequest,
		)
	}

	state := &resolutionState{
		selected: make(map[string]Selection),
		visiting: make(map[string]bool),
	}

	primary, err := r.resolvePackage(
		request.Namespace,
		request.Package,
		request.Channel,
		request.VersionConstraint,
		request.StartingVersion,
		true,
		request.InstalledVersions,
		state,
	)
	if err != nil {
		return nil, err
	}

	all := make([]Selection, 0, len(state.selected))

	for _, selection := range state.selected {
		all = append(all, selection)
	}

	sortSelections(all)

	dependencies := make([]Selection, 0, len(all)-1)

	for _, selection := range all {
		if selection.Primary {
			continue
		}

		dependencies = append(
			dependencies,
			selection,
		)
	}

	return &Result{
		Primary:      primary,
		Dependencies: dependencies,
		All:          all,
	}, nil
}

func (r *Resolver) resolvePackage(
	namespace string,
	packageName string,
	channelName string,
	versionConstraint string,
	startingVersion string,
	primary bool,
	installedVersions map[string]string,
	state *resolutionState,
) (Selection, error) {
	key := packageKey(namespace, packageName)

	if existing, found := state.selected[key]; found {
		if versionConstraint == "" {
			return existing, nil
		}

		matches, err := versionMatches(
			existing.Release.Version,
			versionConstraint,
		)
		if err != nil {
			return Selection{}, err
		}

		if !matches {
			return Selection{}, fmt.Errorf(
				"%w: package %q selected version %q does not satisfy %q",
				ErrUnsatisfiedDependency,
				packageName,
				existing.Release.Version,
				versionConstraint,
			)
		}

		return existing, nil
	}

	if state.visiting[key] {
		cycle := append(
			append([]string(nil), state.path...),
			key,
		)

		return Selection{}, fmt.Errorf(
			"%w: %s",
			ErrDependencyCycle,
			strings.Join(cycle, " -> "),
		)
	}

	state.visiting[key] = true
	state.path = append(state.path, key)

	defer func() {
		state.visiting[key] = false
		state.path = state.path[:len(state.path)-1]
	}()

	pkg, err := r.store.Get(namespace, packageName)
	if err != nil {
		return Selection{}, fmt.Errorf(
			"%w: package %q: %v",
			ErrUnsatisfiedDependency,
			packageName,
			err,
		)
	}

	effectiveChannel := strings.TrimSpace(channelName)

	if effectiveChannel == "" {
		effectiveChannel = strings.TrimSpace(
			pkg.Spec.DefaultChannel,
		)
	}

	release, err := selectRelease(
		pkg,
		effectiveChannel,
		versionConstraint,
		startingVersion,
	)
	if err != nil {
		return Selection{}, err
	}

	if installedVersion := strings.TrimSpace(
		installedVersions[packageName],
	); installedVersion != "" {
		if err := preventDowngrade(
			packageName,
			installedVersion,
			release.Version,
		); err != nil {
			return Selection{}, err
		}
	}

	selection := Selection{
		Package:   packageName,
		Namespace: namespace,
		Channel:   effectiveChannel,
		Release:   *copyRelease(release),
		Primary:   primary,
	}

	state.selected[key] = selection

	for _, dependency := range release.Dependencies {
		dependencyPackage := strings.TrimSpace(
			dependency.Package,
		)

		if dependencyPackage == "" {
			return Selection{}, fmt.Errorf(
				"%w: release %s of package %s contains an empty dependency",
				ErrUnsatisfiedDependency,
				release.Version,
				packageName,
			)
		}

		if _, err := r.resolvePackage(
			namespace,
			dependencyPackage,
			"",
			dependency.VersionConstraint,
			"",
			false,
			installedVersions,
			state,
		); err != nil {
			delete(state.selected, key)

			return Selection{}, fmt.Errorf(
				"resolve dependency %q for %s@%s: %w",
				dependencyPackage,
				packageName,
				release.Version,
				err,
			)
		}
	}

	for _, requiredAPI := range release.RequiredAPIs {
		if apiProvided(requiredAPI, state.selected) {
			continue
		}

		provider, err := r.resolveAPIProvider(
			namespace,
			requiredAPI,
			installedVersions,
			state,
		)
		if err != nil {
			delete(state.selected, key)

			return Selection{}, fmt.Errorf(
				"%w: package=%s version=%s API=%s/%s/%s: %v",
				ErrRequiredAPIUnavailable,
				packageName,
				release.Version,
				requiredAPI.Group,
				requiredAPI.Version,
				requiredAPI.Kind,
				err,
			)
		}

		state.selected[packageKey(provider.Namespace, provider.Package)] = provider
	}

	return selection, nil
}

func (r *Resolver) resolveAPIProvider(
	namespace string,
	required lifecyclev1alpha1.APIReference,
	installedVersions map[string]string,
	state *resolutionState,
) (Selection, error) {
	packages := r.store.List(namespace)

	for _, pkg := range packages {
		key := packageKey(pkg.Namespace, pkg.Name)

		if state.visiting[key] {
			continue
		}

		channel := pkg.Spec.DefaultChannel

		releases, err := channelReleases(pkg, channel)
		if err != nil {
			continue
		}

		for _, candidate := range releases {
			if !releaseProvidesAPI(candidate.release, required) {
				continue
			}

			selection, err := r.resolvePackage(
				pkg.Namespace,
				pkg.Name,
				channel,
				"="+candidate.version.String(),
				"",
				false,
				installedVersions,
				state,
			)
			if err != nil {
				continue
			}

			return selection, nil
		}
	}

	return Selection{}, fmt.Errorf(
		"no package provides %s/%s/%s",
		required.Group,
		required.Version,
		required.Kind,
	)
}

type versionedRelease struct {
	version *semver.Version
	release *lifecyclev1alpha1.OperatorRelease
}

func selectRelease(
	pkg *lifecyclev1alpha1.OperatorPackage,
	channelName string,
	versionConstraint string,
	startingVersion string,
) (*lifecyclev1alpha1.OperatorRelease, error) {
	releases, err := channelReleases(pkg, channelName)
	if err != nil {
		return nil, err
	}

	constraintExpression := strings.TrimSpace(
		versionConstraint,
	)

	if constraintExpression == "" {
		constraintExpression = "*"
	}

	constraint, err := semver.NewConstraint(
		constraintExpression,
	)
	if err != nil {
		return nil, fmt.Errorf(
			"%w: package=%s constraint=%q: %v",
			ErrInvalidRequest,
			pkg.Name,
			versionConstraint,
			err,
		)
	}

	var starting *semver.Version

	if startingVersion != "" {
		starting, err = semver.StrictNewVersion(
			startingVersion,
		)
		if err != nil {
			return nil, fmt.Errorf(
				"%w: invalid starting version %q: %v",
				ErrInvalidRequest,
				startingVersion,
				err,
			)
		}
	}

	for _, candidate := range releases {
		if starting != nil && candidate.version.LessThan(starting) {
			continue
		}

		if constraint.Check(candidate.version) {
			return copyRelease(candidate.release), nil
		}
	}

	return nil, fmt.Errorf(
		"%w: package=%s channel=%s constraint=%q",
		ErrNoCompatibleRelease,
		pkg.Name,
		channelName,
		versionConstraint,
	)
}

func channelReleases(
	pkg *lifecyclev1alpha1.OperatorPackage,
	channelName string,
) ([]versionedRelease, error) {
	var channel *lifecyclev1alpha1.OperatorChannel

	for i := range pkg.Spec.Channels {
		if pkg.Spec.Channels[i].Name == channelName {
			channel = &pkg.Spec.Channels[i]
			break
		}
	}

	if channel == nil {
		return nil, fmt.Errorf(
			"%w: package=%s channel=%s",
			catalog.ErrChannelNotFound,
			pkg.Name,
			channelName,
		)
	}

	releasesByVersion := make(
		map[string]*lifecyclev1alpha1.OperatorRelease,
		len(pkg.Spec.Releases),
	)

	for i := range pkg.Spec.Releases {
		release := &pkg.Spec.Releases[i]
		releasesByVersion[release.Version] = release
	}

	result := make(
		[]versionedRelease,
		0,
		len(channel.Entries),
	)

	for _, entry := range channel.Entries {
		release, found := releasesByVersion[entry.Version]
		if !found {
			return nil, fmt.Errorf(
				"%w: package=%s channel=%s references unknown release=%s",
				catalog.ErrInvalidPackage,
				pkg.Name,
				channelName,
				entry.Version,
			)
		}

		version, err := semver.StrictNewVersion(
			release.Version,
		)
		if err != nil {
			return nil, fmt.Errorf(
				"%w: package=%s release=%q is not valid semantic version: %v",
				catalog.ErrInvalidPackage,
				pkg.Name,
				release.Version,
				err,
			)
		}

		result = append(
			result,
			versionedRelease{
				version: version,
				release: release,
			},
		)
	}

	sort.Slice(
		result,
		func(i, j int) bool {
			return result[i].version.GreaterThan(
				result[j].version,
			)
		},
	)

	return result, nil
}

func versionMatches(
	version string,
	constraintExpression string,
) (bool, error) {
	parsedVersion, err := semver.StrictNewVersion(version)
	if err != nil {
		return false, fmt.Errorf(
			"parse selected version %q: %w",
			version,
			err,
		)
	}

	constraint, err := semver.NewConstraint(
		constraintExpression,
	)
	if err != nil {
		return false, fmt.Errorf(
			"parse version constraint %q: %w",
			constraintExpression,
			err,
		)
	}

	return constraint.Check(parsedVersion), nil
}

func preventDowngrade(
	packageName string,
	installed string,
	target string,
) error {
	installedVersion, err := semver.StrictNewVersion(installed)
	if err != nil {
		return fmt.Errorf(
			"%w: installed version for package %s is invalid: %v",
			ErrInvalidRequest,
			packageName,
			err,
		)
	}

	targetVersion, err := semver.StrictNewVersion(target)
	if err != nil {
		return fmt.Errorf(
			"%w: target version for package %s is invalid: %v",
			ErrInvalidRequest,
			packageName,
			err,
		)
	}

	if targetVersion.LessThan(installedVersion) {
		return fmt.Errorf(
			"%w: package %s target %s is lower than installed %s",
			ErrNoCompatibleRelease,
			packageName,
			target,
			installed,
		)
	}

	return nil
}

func apiProvided(
	required lifecyclev1alpha1.APIReference,
	selections map[string]Selection,
) bool {
	for _, selection := range selections {
		if releaseProvidesAPI(
			&selection.Release,
			required,
		) {
			return true
		}
	}

	return false
}

func releaseProvidesAPI(
	release *lifecyclev1alpha1.OperatorRelease,
	required lifecyclev1alpha1.APIReference,
) bool {
	for _, provided := range release.ProvidedAPIs {
		if provided.Group == required.Group &&
			provided.Version == required.Version &&
			provided.Kind == required.Kind {
			return true
		}
	}

	return false
}

func sortSelections(selections []Selection) {
	sort.Slice(
		selections,
		func(i, j int) bool {
			if selections[i].Primary != selections[j].Primary {
				return selections[i].Primary
			}

			if selections[i].Namespace != selections[j].Namespace {
				return selections[i].Namespace <
					selections[j].Namespace
			}

			return selections[i].Package <
				selections[j].Package
		},
	)
}

func packageKey(namespace, packageName string) string {
	return namespace + "/" + packageName
}

func copyRelease(
	in *lifecyclev1alpha1.OperatorRelease,
) *lifecyclev1alpha1.OperatorRelease {
	if in == nil {
		return nil
	}

	out := *in

	out.ProvidedAPIs = append(
		[]lifecyclev1alpha1.APIReference(nil),
		in.ProvidedAPIs...,
	)

	out.RequiredAPIs = append(
		[]lifecyclev1alpha1.APIReference(nil),
		in.RequiredAPIs...,
	)

	out.Dependencies = append(
		[]lifecyclev1alpha1.OperatorDependency(nil),
		in.Dependencies...,
	)

	if in.PublishedAt != nil {
		out.PublishedAt = in.PublishedAt.DeepCopy()
	}

	return &out
}
