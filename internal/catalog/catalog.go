package catalog

import (
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"

	lifecyclev1alpha1 "github.com/samorca/osha/operator-lifecycle/api/v1alpha1"
)

var (
	ErrPackageNotFound = errors.New("operator package not found")
	ErrChannelNotFound = errors.New("operator channel not found")
	ErrReleaseNotFound = errors.New("operator release not found")
	ErrInvalidPackage  = errors.New("invalid operator package")
)

type PackageKey struct {
	Namespace string
	Name      string
}

type Store interface {
	Put(pkg *lifecyclev1alpha1.OperatorPackage) error
	Delete(namespace, name string)
	Get(namespace, name string) (*lifecyclev1alpha1.OperatorPackage, error)
	List(namespace string) []*lifecyclev1alpha1.OperatorPackage

	FindChannel(
		namespace,
		packageName,
		channelName string,
	) (*lifecyclev1alpha1.OperatorChannel, error)

	FindRelease(
		namespace,
		packageName,
		version string,
	) (*lifecyclev1alpha1.OperatorRelease, error)

	ChannelHead(
		namespace,
		packageName,
		channelName string,
	) (*lifecyclev1alpha1.OperatorRelease, error)
}

type MemoryStore struct {
	mu       sync.RWMutex
	packages map[PackageKey]*lifecyclev1alpha1.OperatorPackage
}

func NewMemoryStore() *MemoryStore {
	return &MemoryStore{
		packages: make(
			map[PackageKey]*lifecyclev1alpha1.OperatorPackage,
		),
	}
}

func (s *MemoryStore) Put(
	pkg *lifecyclev1alpha1.OperatorPackage,
) error {
	if pkg == nil {
		return fmt.Errorf(
			"%w: package is nil",
			ErrInvalidPackage,
		)
	}

	if err := ValidatePackage(pkg); err != nil {
		return err
	}

	key := PackageKey{
		Namespace: pkg.Namespace,
		Name:      pkg.Name,
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	s.packages[key] = pkg.DeepCopy()

	return nil
}

func (s *MemoryStore) Delete(namespace, name string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	delete(
		s.packages,
		PackageKey{
			Namespace: namespace,
			Name:      name,
		},
	)
}

func (s *MemoryStore) Get(
	namespace,
	name string,
) (*lifecyclev1alpha1.OperatorPackage, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	pkg, found := s.packages[PackageKey{
		Namespace: namespace,
		Name:      name,
	}]

	if !found {
		return nil, fmt.Errorf(
			"%w: %s/%s",
			ErrPackageNotFound,
			namespace,
			name,
		)
	}

	return pkg.DeepCopy(), nil
}

func (s *MemoryStore) List(
	namespace string,
) []*lifecyclev1alpha1.OperatorPackage {
	s.mu.RLock()
	defer s.mu.RUnlock()

	result := make(
		[]*lifecyclev1alpha1.OperatorPackage,
		0,
		len(s.packages),
	)

	for key, pkg := range s.packages {
		if namespace != "" && key.Namespace != namespace {
			continue
		}

		result = append(
			result,
			pkg.DeepCopy(),
		)
	}

	sort.Slice(
		result,
		func(i, j int) bool {
			if result[i].Namespace != result[j].Namespace {
				return result[i].Namespace < result[j].Namespace
			}

			return result[i].Name < result[j].Name
		},
	)

	return result
}

func (s *MemoryStore) FindChannel(
	namespace,
	packageName,
	channelName string,
) (*lifecyclev1alpha1.OperatorChannel, error) {
	pkg, err := s.Get(namespace, packageName)
	if err != nil {
		return nil, err
	}

	effectiveChannel := strings.TrimSpace(channelName)

	if effectiveChannel == "" {
		effectiveChannel = pkg.Spec.DefaultChannel
	}

	for i := range pkg.Spec.Channels {
		channel := pkg.Spec.Channels[i]

		if channel.Name == effectiveChannel {
			result := channel
			result.Entries = append(
				[]lifecyclev1alpha1.ChannelEntry(nil),
				channel.Entries...,
			)

			return &result, nil
		}
	}

	return nil, fmt.Errorf(
		"%w: package=%s channel=%s",
		ErrChannelNotFound,
		packageName,
		effectiveChannel,
	)
}

func (s *MemoryStore) FindRelease(
	namespace,
	packageName,
	version string,
) (*lifecyclev1alpha1.OperatorRelease, error) {
	pkg, err := s.Get(namespace, packageName)
	if err != nil {
		return nil, err
	}

	version = strings.TrimSpace(version)

	for i := range pkg.Spec.Releases {
		if pkg.Spec.Releases[i].Version == version {
			release := pkg.Spec.Releases[i]

			return copyRelease(&release), nil
		}
	}

	return nil, fmt.Errorf(
		"%w: package=%s version=%s",
		ErrReleaseNotFound,
		packageName,
		version,
	)
}

func (s *MemoryStore) ChannelHead(
	namespace,
	packageName,
	channelName string,
) (*lifecyclev1alpha1.OperatorRelease, error) {
	channel, err := s.FindChannel(
		namespace,
		packageName,
		channelName,
	)
	if err != nil {
		return nil, err
	}

	if len(channel.Entries) == 0 {
		return nil, fmt.Errorf(
			"%w: package=%s channel=%s has no entries",
			ErrReleaseNotFound,
			packageName,
			channel.Name,
		)
	}

	head := channel.Entries[len(channel.Entries)-1]

	return s.FindRelease(
		namespace,
		packageName,
		head.Version,
	)
}

func ValidatePackage(
	pkg *lifecyclev1alpha1.OperatorPackage,
) error {
	if pkg == nil {
		return fmt.Errorf(
			"%w: package is nil",
			ErrInvalidPackage,
		)
	}

	if strings.TrimSpace(pkg.Name) == "" {
		return fmt.Errorf(
			"%w: metadata.name is required",
			ErrInvalidPackage,
		)
	}

	if strings.TrimSpace(pkg.Spec.DisplayName) == "" {
		return fmt.Errorf(
			"%w: spec.displayName is required",
			ErrInvalidPackage,
		)
	}

	if strings.TrimSpace(pkg.Spec.Provider) == "" {
		return fmt.Errorf(
			"%w: spec.provider is required",
			ErrInvalidPackage,
		)
	}

	if strings.TrimSpace(pkg.Spec.DefaultChannel) == "" {
		return fmt.Errorf(
			"%w: spec.defaultChannel is required",
			ErrInvalidPackage,
		)
	}

	if len(pkg.Spec.Channels) == 0 {
		return fmt.Errorf(
			"%w: at least one channel is required",
			ErrInvalidPackage,
		)
	}

	if len(pkg.Spec.Releases) == 0 {
		return fmt.Errorf(
			"%w: at least one release is required",
			ErrInvalidPackage,
		)
	}

	releases := make(
		map[string]struct{},
		len(pkg.Spec.Releases),
	)

	for _, release := range pkg.Spec.Releases {
		version := strings.TrimSpace(release.Version)

		if version == "" {
			return fmt.Errorf(
				"%w: release version cannot be empty",
				ErrInvalidPackage,
			)
		}

		if _, exists := releases[version]; exists {
			return fmt.Errorf(
				"%w: duplicate release %q",
				ErrInvalidPackage,
				version,
			)
		}

		releases[version] = struct{}{}

		if strings.TrimSpace(release.Artifacts.Image) == "" {
			return fmt.Errorf(
				"%w: release %q has no controller image",
				ErrInvalidPackage,
				version,
			)
		}
	}

	channels := make(
		map[string]struct{},
		len(pkg.Spec.Channels),
	)

	defaultFound := false

	for _, channel := range pkg.Spec.Channels {
		name := strings.TrimSpace(channel.Name)

		if name == "" {
			return fmt.Errorf(
				"%w: channel name cannot be empty",
				ErrInvalidPackage,
			)
		}

		if _, exists := channels[name]; exists {
			return fmt.Errorf(
				"%w: duplicate channel %q",
				ErrInvalidPackage,
				name,
			)
		}

		channels[name] = struct{}{}

		if name == pkg.Spec.DefaultChannel {
			defaultFound = true
		}

		if len(channel.Entries) == 0 {
			return fmt.Errorf(
				"%w: channel %q has no entries",
				ErrInvalidPackage,
				name,
			)
		}

		seenVersions := make(map[string]struct{})

		for _, entry := range channel.Entries {
			version := strings.TrimSpace(entry.Version)

			if _, exists := releases[version]; !exists {
				return fmt.Errorf(
					"%w: channel %q references unknown release %q",
					ErrInvalidPackage,
					name,
					version,
				)
			}

			if _, exists := seenVersions[version]; exists {
				return fmt.Errorf(
					"%w: duplicate release %q in channel %q",
					ErrInvalidPackage,
					version,
					name,
				)
			}

			seenVersions[version] = struct{}{}

			if entry.Replaces != "" {
				if _, exists := releases[entry.Replaces]; !exists {
					return fmt.Errorf(
						"%w: channel %q release %q replaces unknown release %q",
						ErrInvalidPackage,
						name,
						version,
						entry.Replaces,
					)
				}
			}

			for _, skippedVersion := range entry.Skips {
				if _, exists := releases[skippedVersion]; !exists {
					return fmt.Errorf(
						"%w: channel %q release %q skips unknown release %q",
						ErrInvalidPackage,
						name,
						version,
						skippedVersion,
					)
				}
			}
		}
	}

	if !defaultFound {
		return fmt.Errorf(
			"%w: default channel %q does not exist",
			ErrInvalidPackage,
			pkg.Spec.DefaultChannel,
		)
	}

	return nil
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
