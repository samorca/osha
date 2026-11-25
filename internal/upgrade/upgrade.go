package upgrade

import (
	"errors"
	"fmt"
	"sort"
	"strings"

	"github.com/Masterminds/semver/v3"

	lifecyclev1alpha1 "github.com/samorca/osha/operator-lifecycle/api/v1alpha1"
)

var (
	ErrPackageRequired     = errors.New("operator package is required")
	ErrChannelNotFound     = errors.New("operator channel not found")
	ErrVersionNotFound     = errors.New("operator version not found")
	ErrInvalidVersion      = errors.New("invalid semantic version")
	ErrNoUpgradePath       = errors.New("no valid upgrade path")
	ErrDowngradeNotAllowed = errors.New("downgrade is not allowed")
	ErrChannelGraphInvalid = errors.New("channel upgrade graph is invalid")
)

type Transition struct {
	From string
	To   string

	Skipped []string
}

type Plan struct {
	Package string
	Channel string

	CurrentVersion string
	TargetVersion  string

	Transitions []Transition
}

type Engine struct{}

func New() *Engine {
	return &Engine{}
}

func (e *Engine) Plan(
	pkg *lifecyclev1alpha1.OperatorPackage,
	channelName string,
	currentVersion string,
	targetVersion string,
) (*Plan, error) {
	if pkg == nil {
		return nil, ErrPackageRequired
	}

	channelName = strings.TrimSpace(channelName)
	currentVersion = strings.TrimSpace(currentVersion)
	targetVersion = strings.TrimSpace(targetVersion)

	if channelName == "" {
		channelName = strings.TrimSpace(
			pkg.Spec.DefaultChannel,
		)
	}

	channel, err := findChannel(pkg, channelName)
	if err != nil {
		return nil, err
	}

	releases, err := releaseIndex(pkg)
	if err != nil {
		return nil, err
	}

	if _, found := releases[currentVersion]; !found {
		return nil, fmt.Errorf(
			"%w: current=%s",
			ErrVersionNotFound,
			currentVersion,
		)
	}

	if _, found := releases[targetVersion]; !found {
		return nil, fmt.Errorf(
			"%w: target=%s",
			ErrVersionNotFound,
			targetVersion,
		)
	}

	currentSemver, err := parseVersion(currentVersion)
	if err != nil {
		return nil, err
	}

	targetSemver, err := parseVersion(targetVersion)
	if err != nil {
		return nil, err
	}

	if targetSemver.LessThan(currentSemver) {
		return nil, fmt.Errorf(
			"%w: current=%s target=%s",
			ErrDowngradeNotAllowed,
			currentVersion,
			targetVersion,
		)
	}

	if targetSemver.Equal(currentSemver) {
		return &Plan{
			Package:        pkg.Name,
			Channel:        channel.Name,
			CurrentVersion: currentVersion,
			TargetVersion:  targetVersion,
			Transitions:    nil,
		}, nil
	}

	graph, err := buildGraph(channel, releases)
	if err != nil {
		return nil, err
	}

	path, err := shortestUpgradePath(
		graph,
		currentVersion,
		targetVersion,
	)
	if err != nil {
		return nil, err
	}

	transitions := make(
		[]Transition,
		0,
		len(path)-1,
	)

	for i := 0; i < len(path)-1; i++ {
		from := path[i]
		to := path[i+1]

		transition := Transition{
			From: from,
			To:   to,
		}

		for _, entry := range channel.Entries {
			if entry.Version != to {
				continue
			}

			transition.Skipped = append(
				[]string(nil),
				entry.Skips...,
			)

			break
		}

		transitions = append(
			transitions,
			transition,
		)
	}

	return &Plan{
		Package:        pkg.Name,
		Channel:        channel.Name,
		CurrentVersion: currentVersion,
		TargetVersion:  targetVersion,
		Transitions:    transitions,
	}, nil
}

type graph map[string][]string

func buildGraph(
	channel *lifecyclev1alpha1.OperatorChannel,
	releases map[string]*lifecyclev1alpha1.OperatorRelease,
) (graph, error) {
	result := make(graph)

	channelVersions := make(map[string]struct{})

	for _, entry := range channel.Entries {
		if _, found := releases[entry.Version]; !found {
			return nil, fmt.Errorf(
				"%w: channel=%s unknown version=%s",
				ErrChannelGraphInvalid,
				channel.Name,
				entry.Version,
			)
		}

		channelVersions[entry.Version] = struct{}{}

		if _, exists := result[entry.Version]; !exists {
			result[entry.Version] = nil
		}
	}

	for _, entry := range channel.Entries {
		if entry.Replaces != "" {
			if _, exists := releases[entry.Replaces]; !exists {
				return nil, fmt.Errorf(
					"%w: %s replaces unknown version %s",
					ErrChannelGraphInvalid,
					entry.Version,
					entry.Replaces,
				)
			}

			result[entry.Replaces] = appendUnique(
				result[entry.Replaces],
				entry.Version,
			)
		}

		for _, skipped := range entry.Skips {
			if _, exists := releases[skipped]; !exists {
				return nil, fmt.Errorf(
					"%w: %s skips unknown version %s",
					ErrChannelGraphInvalid,
					entry.Version,
					skipped,
				)
			}

			result[skipped] = appendUnique(
				result[skipped],
				entry.Version,
			)
		}
	}

	ordered, err := orderedChannelVersions(
		channel,
		releases,
	)
	if err != nil {
		return nil, err
	}

	for i := 0; i < len(ordered)-1; i++ {
		current := ordered[i]
		next := ordered[i+1]

		if hasExplicitIncomingEdge(
			next,
			channel,
		) {
			continue
		}

		result[current] = appendUnique(
			result[current],
			next,
		)
	}

	for from := range result {
		sort.Slice(
			result[from],
			func(i, j int) bool {
				left, _ := parseVersion(
					result[from][i],
				)

				right, _ := parseVersion(
					result[from][j],
				)

				return left.LessThan(right)
			},
		)
	}

	_ = channelVersions

	return result, nil
}

func shortestUpgradePath(
	graph graph,
	currentVersion string,
	targetVersion string,
) ([]string, error) {
	type queueItem struct {
		Version string
		Path    []string
	}

	queue := []queueItem{
		{
			Version: currentVersion,
			Path: []string{
				currentVersion,
			},
		},
	}

	visited := map[string]bool{
		currentVersion: true,
	}

	for len(queue) > 0 {
		item := queue[0]
		queue = queue[1:]

		if item.Version == targetVersion {
			return item.Path, nil
		}

		for _, successor := range graph[item.Version] {
			if visited[successor] {
				continue
			}

			successorVersion, err := parseVersion(
				successor,
			)
			if err != nil {
				return nil, err
			}

			targetSemver, err := parseVersion(
				targetVersion,
			)
			if err != nil {
				return nil, err
			}

			if successorVersion.GreaterThan(
				targetSemver,
			) {
				continue
			}

			visited[successor] = true

			path := append(
				append([]string(nil), item.Path...),
				successor,
			)

			queue = append(
				queue,
				queueItem{
					Version: successor,
					Path:    path,
				},
			)
		}
	}

	return nil, fmt.Errorf(
		"%w: current=%s target=%s",
		ErrNoUpgradePath,
		currentVersion,
		targetVersion,
	)
}

func findChannel(
	pkg *lifecyclev1alpha1.OperatorPackage,
	name string,
) (*lifecyclev1alpha1.OperatorChannel, error) {
	for i := range pkg.Spec.Channels {
		if pkg.Spec.Channels[i].Name == name {
			channel := pkg.Spec.Channels[i]

			channel.Entries = append(
				[]lifecyclev1alpha1.ChannelEntry(nil),
				channel.Entries...,
			)

			return &channel, nil
		}
	}

	return nil, fmt.Errorf(
		"%w: package=%s channel=%s",
		ErrChannelNotFound,
		pkg.Name,
		name,
	)
}

func releaseIndex(
	pkg *lifecyclev1alpha1.OperatorPackage,
) (map[string]*lifecyclev1alpha1.OperatorRelease, error) {
	result := make(
		map[string]*lifecyclev1alpha1.OperatorRelease,
		len(pkg.Spec.Releases),
	)

	for i := range pkg.Spec.Releases {
		release := &pkg.Spec.Releases[i]

		if _, err := parseVersion(release.Version); err != nil {
			return nil, fmt.Errorf(
				"release %q: %w",
				release.Version,
				err,
			)
		}

		if _, exists := result[release.Version]; exists {
			return nil, fmt.Errorf(
				"%w: duplicate version=%s",
				ErrChannelGraphInvalid,
				release.Version,
			)
		}

		result[release.Version] = release
	}

	return result, nil
}

func orderedChannelVersions(
	channel *lifecyclev1alpha1.OperatorChannel,
	releases map[string]*lifecyclev1alpha1.OperatorRelease,
) ([]string, error) {
	type entry struct {
		version string
		semver  *semver.Version
	}

	versions := make(
		[]entry,
		0,
		len(channel.Entries),
	)

	seen := make(map[string]struct{})

	for _, channelEntry := range channel.Entries {
		if _, exists := seen[channelEntry.Version]; exists {
			return nil, fmt.Errorf(
				"%w: duplicate channel version=%s",
				ErrChannelGraphInvalid,
				channelEntry.Version,
			)
		}

		seen[channelEntry.Version] = struct{}{}

		if _, exists := releases[channelEntry.Version]; !exists {
			return nil, fmt.Errorf(
				"%w: version=%s missing release",
				ErrChannelGraphInvalid,
				channelEntry.Version,
			)
		}

		parsed, err := parseVersion(
			channelEntry.Version,
		)
		if err != nil {
			return nil, err
		}

		versions = append(
			versions,
			entry{
				version: channelEntry.Version,
				semver:  parsed,
			},
		)
	}

	sort.Slice(
		versions,
		func(i, j int) bool {
			return versions[i].semver.LessThan(
				versions[j].semver,
			)
		},
	)

	result := make(
		[]string,
		0,
		len(versions),
	)

	for _, version := range versions {
		result = append(
			result,
			version.version,
		)
	}

	return result, nil
}

func hasExplicitIncomingEdge(
	version string,
	channel *lifecyclev1alpha1.OperatorChannel,
) bool {
	for _, entry := range channel.Entries {
		if entry.Version != version {
			continue
		}

		if entry.Replaces != "" ||
			len(entry.Skips) > 0 {
			return true
		}
	}

	return false
}

func appendUnique(
	values []string,
	value string,
) []string {
	for _, existing := range values {
		if existing == value {
			return values
		}
	}

	return append(values, value)
}

func parseVersion(
	value string,
) (*semver.Version, error) {
	value = strings.TrimSpace(value)

	version, err := semver.StrictNewVersion(value)
	if err != nil {
		return nil, fmt.Errorf(
			"%w: %q: %v",
			ErrInvalidVersion,
			value,
			err,
		)
	}

	return version, nil
}
